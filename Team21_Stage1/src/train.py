"""Training pipeline for Stage 1: Traditional Baselines + Self-designed CNN Ablation Study."""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import compute_classification_metrics, evaluate_cnn, measure_inference_latency
from model import CLASS_ORDER, SteelDefectCNN, count_parameters


NORM_MEAN = [0.5071]
NORM_STD = [0.2132]
INPUT_SIZE = [200, 200]


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SteelClassificationDataset(Dataset):
    """Grayscale image folder dataset using fixed CLASS_ORDER (cached in RAM)."""

    def __init__(self, split_dir: Path, transform=None, preload: bool = True) -> None:
        self.samples: List[Tuple[Path, int]] = []
        self.transform = transform
        self.images: List[Image.Image] = []
        for idx, cls_name in enumerate(CLASS_ORDER):
            cls_dir = split_dir / cls_name
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {cls_dir}")
            files = sorted(
                p for p in cls_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )
            for p in files:
                self.samples.append((p, idx))
                if preload:
                    with Image.open(p) as im:
                        self.images.append(im.convert("L").copy())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        if self.images:
            img = self.images[idx].copy()
        else:
            img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, str(path)


def extract_flat_features(
    dataset: SteelClassificationDataset, resize_wh: Tuple[int, int] = (64, 64)
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract flattened normalized grayscale pixels for traditional ML baselines."""
    feats = []
    labels = []
    for path, label in dataset.samples:
        img = Image.open(path).convert("L").resize(resize_wh, Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
        feats.append(arr)
        labels.append(label)
    return np.stack(feats, axis=0), np.array(labels, dtype=np.int64)


def run_traditional_baselines(data_dir: Path, seed: int = 42) -> Dict:
    """Train and evaluate PCA + SVM and PCA + KNN on 64x64 grayscale features."""
    train_ds = SteelClassificationDataset(data_dir / "train")
    val_ds = SteelClassificationDataset(data_dir / "validation")

    X_train, y_train = extract_flat_features(train_ds, (64, 64))
    X_val, y_val = extract_flat_features(val_ds, (64, 64))

    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    pca = PCA(n_components=0.95, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    pca_fit_time = time.perf_counter() - t0

    n_components = int(pca.n_components_)
    cum_var = float(np.sum(pca.explained_variance_ratio_))
    explained_curve = np.cumsum(pca.explained_variance_ratio_).tolist()

    results = {
        "feature_dim_raw": int(X_train.shape[1]),
        "pca_n_components": n_components,
        "pca_explained_variance": cum_var,
        "pca_explained_curve": explained_curve,
        "pca_fit_time_sec": float(pca_fit_time),
        "models": {},
    }

    # 1. PCA + SVM (RBF)
    t_s0 = time.perf_counter()
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed)
    svm.fit(X_train_pca, y_train)
    svm_train_time = time.perf_counter() - t_s0

    t_i0 = time.perf_counter()
    svm_val_preds = svm.predict(X_val_pca)
    svm_inf_ms = (time.perf_counter() - t_i0) * 1000.0 / len(X_val_pca)
    svm_train_preds = svm.predict(X_train_pca)

    svm_val_metrics = compute_classification_metrics(y_val.tolist(), svm_val_preds.tolist(), CLASS_ORDER)
    svm_train_metrics = compute_classification_metrics(y_train.tolist(), svm_train_preds.tolist(), CLASS_ORDER)
    results["models"]["PCA_SVM"] = {
        "method": "StandardScaler + PCA(0.95) + SVC(kernel='rbf', C=10.0, gamma='scale')",
        "train_time_sec": float(pca_fit_time + svm_train_time),
        "cpu_ms_per_img": float(svm_inf_ms),
        "train_accuracy": svm_train_metrics["accuracy"],
        "train_macro_f1": svm_train_metrics["macro_f1"],
        **svm_val_metrics,
    }

    # 2. PCA + KNN
    t_k0 = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")
    knn.fit(X_train_pca, y_train)
    knn_train_time = time.perf_counter() - t_k0

    t_ki0 = time.perf_counter()
    knn_val_preds = knn.predict(X_val_pca)
    knn_inf_ms = (time.perf_counter() - t_ki0) * 1000.0 / len(X_val_pca)
    knn_train_preds = knn.predict(X_train_pca)

    knn_val_metrics = compute_classification_metrics(y_val.tolist(), knn_val_preds.tolist(), CLASS_ORDER)
    knn_train_metrics = compute_classification_metrics(y_train.tolist(), knn_train_preds.tolist(), CLASS_ORDER)
    results["models"]["PCA_KNN"] = {
        "method": "StandardScaler + PCA(0.95) + KNeighborsClassifier(n_neighbors=5, weights='distance')",
        "train_time_sec": float(pca_fit_time + knn_train_time),
        "cpu_ms_per_img": float(knn_inf_ms),
        "train_accuracy": knn_train_metrics["accuracy"],
        "train_macro_f1": knn_train_metrics["macro_f1"],
        **knn_val_metrics,
    }

    return results


def build_transforms(use_aug: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build deterministic baseline or augmented train transform + fixed eval transform."""
    eval_tf = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE[0], INPUT_SIZE[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )
    if not use_aug:
        return eval_tf, eval_tf

    train_tf = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE[0], INPUT_SIZE[1])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )
    return train_tf, eval_tf


def train_single_cnn(
    exp_cfg: Dict,
    data_dir: Path,
    device: torch.device,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    """Train one CNN configuration and return full history + best checkpoint state."""
    set_seed(seed)
    train_tf, eval_tf = build_transforms(use_aug=exp_cfg["use_aug"])

    train_ds = SteelClassificationDataset(data_dir / "train", transform=train_tf)
    train_eval_ds = SteelClassificationDataset(data_dir / "train", transform=eval_tf)
    val_ds = SteelClassificationDataset(data_dir / "validation", transform=eval_tf)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SteelDefectCNN(
        num_classes=len(CLASS_ORDER),
        use_bn=exp_cfg["use_bn"],
        dropout_p=exp_cfg["dropout_p"],
    ).to(device)
    param_info = count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=exp_cfg["weight_decay"],
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_val_metrics = None
    best_train_metrics = None
    no_improve = 0
    patience = exp_cfg.get("patience", 20)

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_train = 0
        tr_true, tr_pred = [], []

        for imgs, targets, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            running_loss += float(loss.item()) * bs
            n_train += bs
            tr_true.extend(targets.cpu().numpy().tolist())
            tr_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())

        tr_metrics = compute_classification_metrics(tr_true, tr_pred, CLASS_ORDER)
        tr_loss = running_loss / max(n_train, 1)
        val_metrics = evaluate_cnn(model, val_loader, criterion, device, CLASS_ORDER, collect_errors=False)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(tr_loss))
        history["train_acc"].append(tr_metrics["accuracy"])
        history["train_macro_f1"].append(tr_metrics["macro_f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        improved = (val_metrics["macro_f1"] > best_f1 + 1e-6) or (
            abs(val_metrics["macro_f1"] - best_f1) <= 1e-6 and val_metrics["loss"] < best_loss
        )
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if exp_cfg["early_stopping"] and no_improve >= patience:
            break

    train_time = time.perf_counter() - t0
    last_epoch = history["epoch"][-1]

    # Measure final epoch clean train & val for overfitting diagnostic
    final_train_clean = evaluate_cnn(model, train_eval_loader, criterion, device, CLASS_ORDER)
    final_val_clean = evaluate_cnn(model, val_loader, criterion, device, CLASS_ORDER)

    # Restore best checkpoint and compute full evaluation + errors + latency
    model.load_state_dict(best_state)
    best_train_metrics = evaluate_cnn(model, train_eval_loader, criterion, device, CLASS_ORDER)
    best_val_metrics = evaluate_cnn(model, val_loader, criterion, device, CLASS_ORDER, collect_errors=True)
    latency = measure_inference_latency(model, input_shape=(1, 1, INPUT_SIZE[0], INPUT_SIZE[1]))

    exp_result = {
        "exp_id": exp_cfg["exp_id"],
        "name": exp_cfg["name"],
        "config": exp_cfg,
        "params": param_info,
        "epochs_configured": epochs,
        "epochs_ran": last_epoch,
        "best_epoch": best_epoch,
        "train_time_sec": float(train_time),
        "latency": latency,
        "best_val": best_val_metrics,
        "best_train_clean": {
            "loss": best_train_metrics["loss"],
            "accuracy": best_train_metrics["accuracy"],
            "macro_f1": best_train_metrics["macro_f1"],
        },
        "final_epoch_diagnostic": {
            "train_accuracy": final_train_clean["accuracy"],
            "train_macro_f1": final_train_clean["macro_f1"],
            "val_accuracy": final_val_clean["accuracy"],
            "val_macro_f1": final_val_clean["macro_f1"],
            "overfit_gap_f1": float(final_train_clean["macro_f1"] - final_val_clean["macro_f1"]),
        },
        "best_overfit_gap_f1": float(best_train_metrics["macro_f1"] - best_val_metrics["macro_f1"]),
        "history": history,
    }

    ckpt_payload = {
        "state_dict": best_state,
        "model_kwargs": {
            "num_classes": len(CLASS_ORDER),
            "use_bn": exp_cfg["use_bn"],
            "dropout_p": exp_cfg["dropout_p"],
        },
        "class_order": CLASS_ORDER,
        "input_size": INPUT_SIZE,
        "normalization_mean": NORM_MEAN,
        "normalization_std": NORM_STD,
        "experiment_id": exp_cfg["exp_id"],
        "experiment_name": exp_cfg["name"],
        "seed": seed,
        "best_epoch": best_epoch,
        "val_macro_f1": best_val_metrics["macro_f1"],
    }
    return exp_result, ckpt_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 Training & Ablation Runner")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/STUDENT_PUBLIC/classification",
        help="Path to classification dataset directory containing train/ and validation/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="workspace_outputs/stage1",
        help="Directory to store metrics JSONs and intermediate checkpoints",
    )
    parser.add_argument(
        "--submission_dir",
        type=str,
        default="Team21_Stage1",
        help="Directory where final stage1_model.pth and config.json live",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    submission_dir = Path(args.submission_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 1] Device={device}, Data={data_dir}, Seed={args.seed}")

    # Part A: Traditional Baselines
    print("[Stage 1] Running Traditional Baselines (PCA+SVM, PCA+KNN)...")
    trad_results = run_traditional_baselines(data_dir, seed=args.seed)
    with open(output_dir / "traditional_results.json", "w", encoding="utf-8") as f:
        json.dump(trad_results, f, indent=2)
    for k, m in trad_results["models"].items():
        print(f"  {k}: Acc={m['accuracy']:.4f}, MacroF1={m['macro_f1']:.4f}, CPU={m['cpu_ms_per_img']:.3f}ms")

    # Part A & B: Controlled CNN Ablation
    experiments = [
        {
            "exp_id": "E1",
            "name": "Baseline CNN",
            "use_bn": False,
            "use_aug": False,
            "dropout_p": 0.0,
            "weight_decay": 0.0,
            "early_stopping": False,
            "methods": [],
        },
        {
            "exp_id": "E2",
            "name": "Baseline + A (BatchNorm)",
            "use_bn": True,
            "use_aug": False,
            "dropout_p": 0.0,
            "weight_decay": 0.0,
            "early_stopping": False,
            "methods": ["Batch Normalization"],
        },
        {
            "exp_id": "E3",
            "name": "Baseline + B (DataAug)",
            "use_bn": False,
            "use_aug": True,
            "dropout_p": 0.0,
            "weight_decay": 0.0,
            "early_stopping": False,
            "methods": ["Data Augmentation"],
        },
        {
            "exp_id": "E4",
            "name": "Baseline + A + B (BN + DataAug)",
            "use_bn": True,
            "use_aug": True,
            "dropout_p": 0.0,
            "weight_decay": 0.0,
            "early_stopping": False,
            "methods": ["Batch Normalization", "Data Augmentation"],
        },
        {
            "exp_id": "E5",
            "name": "Baseline + A + B + C (BN + DataAug + Dropout + L2 + ES)",
            "use_bn": True,
            "use_aug": True,
            "dropout_p": 0.3,
            "weight_decay": 1e-4,
            "early_stopping": True,
            "patience": 20,
            "methods": [
                "Batch Normalization",
                "Data Augmentation",
                "Dropout(p=0.3)",
                "L2 Weight Decay(1e-4)",
                "Early Stopping(patience=20)",
            ],
        },
    ]

    cnn_results = {}
    checkpoints = {}
    for exp_cfg in experiments:
        print(f"[Stage 1] Training {exp_cfg['exp_id']}: {exp_cfg['name']}...")
        res, ckpt = train_single_cnn(
            exp_cfg,
            data_dir=data_dir,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
        cnn_results[exp_cfg["exp_id"]] = res
        checkpoints[exp_cfg["exp_id"]] = ckpt
        torch.save(ckpt, output_dir / "checkpoints" / f"{exp_cfg['exp_id']}.pth")
        bv = res["best_val"]
        print(
            f"  -> Best@{res['best_epoch']:2d}/{res['epochs_ran']:2d}: "
            f"Acc={bv['accuracy']:.4f}, MacroF1={bv['macro_f1']:.4f}, "
            f"Gap={res['best_overfit_gap_f1']:.4f}, Time={res['train_time_sec']:.1f}s"
        )

    # Select final improved CNN among combined-improvement candidates (E4, E5)
    # Primary: Val Macro F1; tie-breakers: late-stage Val Macro F1 stability & lowest val loss
    improved_ids = ["E4", "E5"]
    best_id = max(
        improved_ids,
        key=lambda eid: (
            round(cnn_results[eid]["best_val"]["macro_f1"], 4),
            round(sum(cnn_results[eid]["history"]["val_macro_f1"][-10:]) / 10.0, 4),
            -abs(cnn_results[eid]["final_epoch_diagnostic"]["overfit_gap_f1"]),
            -cnn_results[eid]["best_val"]["loss"],
        ),
    )

    final_ckpt_path = submission_dir / "stage1_model.pth"
    torch.save(checkpoints[best_id], final_ckpt_path)
    ckpt_size_mb = final_ckpt_path.stat().st_size / (1024.0 * 1024.0)

    summary = {
        "selected_model_id": best_id,
        "selected_model_name": cnn_results[best_id]["name"],
        "selected_val_macro_f1": cnn_results[best_id]["best_val"]["macro_f1"],
        "selected_val_accuracy": cnn_results[best_id]["best_val"]["accuracy"],
        "checkpoint_size_mb": float(ckpt_size_mb),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "normalization_mean": NORM_MEAN,
        "normalization_std": NORM_STD,
        "input_size": INPUT_SIZE,
        "experiments": cnn_results,
    }
    with open(output_dir / "cnn_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Keep config.json synchronized with winning model
    config_payload = {
        "team_id": "Team21",
        "stage": 1,
        "model_name": f"SteelDefectCNN_{best_id}",
        "input_size": [1, INPUT_SIZE[0], INPUT_SIZE[1]],
        "class_order": CLASS_ORDER,
        "normalization_mean": NORM_MEAN,
        "normalization_std": NORM_STD,
        "seed": args.seed,
        "improvement_methods": cnn_results[best_id]["config"]["methods"],
        "checkpoint_file": "stage1_model.pth",
    }
    with open(submission_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    print(
        f"[Stage 1] DONE! Selected {best_id} ({cnn_results[best_id]['name']}), "
        f"Val Macro F1={cnn_results[best_id]['best_val']['macro_f1']:.4f}, "
        f"Saved to {final_ckpt_path} ({ckpt_size_mb:.2f} MB)"
    )


if __name__ == "__main__":
    main()

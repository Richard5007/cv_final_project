"""Training pipeline for Stage 2: 2x3 Controlled Transfer Learning + Robustness Benchmark."""

import argparse
import copy
import io
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import (
    build_eval_transform,
    evaluate_clean_loader,
    evaluate_local_robustness,
    evaluate_model_on_samples,
    measure_inference_latency,
)
from model import (
    CLASS_ORDER,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PRETRAINED_SOURCES,
    build_stage2_model,
    get_layer_freeze_summary,
    get_parameter_groups,
)

INPUT_SIZE = 224


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RobustnessAwareAugmentation:
    """Unified factory-shift augmentation applied identically across all 2x3 Stage 2 runs."""

    def __init__(self, input_size: int = 224) -> None:
        self.spatial_tf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(
                    size=(input_size, input_size),
                    scale=(0.82, 1.0),
                    ratio=(0.9, 1.1),
                ),
            ]
        )
        self.color_jitter = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.45, contrast=0.45)],
            p=0.6,
        )
        self.gauss_blur = transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.2))],
            p=0.4,
        )
        self.to_tensor_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = self.spatial_tf(img)
        img = self.color_jitter(img)
        img = self.gauss_blur(img)

        arr = np.asarray(img, dtype=np.float32) / 255.0

        # Random gamma shift
        if random.random() < 0.35:
            gamma = random.uniform(0.55, 1.7)
            arr = np.clip(arr, 1e-4, 1.0) ** gamma

        # Linear illumination shading gradient
        if random.random() < 0.35:
            h, w = arr.shape[:2]
            strength = random.uniform(-0.28, 0.28)
            if random.random() < 0.5:
                grad = np.linspace(1.0 - strength, 1.0 + strength, h, dtype=np.float32)[:, None, None]
            else:
                grad = np.linspace(1.0 - strength, 1.0 + strength, w, dtype=np.float32)[None, :, None]
            arr = np.clip(arr * grad, 0.0, 1.0)

        # Directional motion blur
        if random.random() < 0.25:
            mk = random.choice([5, 7])
            kernel = np.zeros((mk, mk), dtype=np.float32)
            if random.random() < 0.5:
                kernel[mk // 2, :] = 1.0 / mk
            else:
                kernel[:, mk // 2] = 1.0 / mk
            arr_u8 = cv2.filter2D((arr * 255.0).astype(np.uint8), -1, kernel)
            arr = arr_u8.astype(np.float32) / 255.0

        # Sensor Gaussian noise
        if random.random() < 0.40:
            sigma = random.uniform(0.01, 0.07)
            noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 1.0)

        out_img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")

        # In-memory JPEG compression artifacts
        if random.random() < 0.40:
            quality = random.randint(15, 75)
            buf = io.BytesIO()
            out_img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            out_img = Image.open(buf).convert("RGB")

        return self.to_tensor_norm(out_img)


def collect_split_samples(split_dir: Path) -> List[Tuple[Path, int]]:
    """Collect ordered (path, class_idx) pairs using CLASS_ORDER."""
    samples: List[Tuple[Path, int]] = []
    for idx, cls_name in enumerate(CLASS_ORDER):
        cls_dir = split_dir / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {cls_dir}")
        files = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        for p in files:
            samples.append((p, idx))
    return samples


class CachedRGBDataset(Dataset):
    """In-memory cached RGB PIL dataset for fast reproducible training."""

    def __init__(self, samples: List[Tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform
        self.images = []
        for p, _ in samples:
            with Image.open(p) as im:
                self.images.append(im.convert("RGB").copy())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img = self.images[idx].copy()
        _, label = self.samples[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def train_single_experiment(
    exp_cfg: Dict,
    train_samples: List[Tuple[Path, int]],
    val_samples: List[Tuple[Path, int]],
    device: torch.device,
    epochs: int = 35,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    """Run one (architecture x strategy) transfer learning experiment."""
    set_seed(seed)
    arch = exp_cfg["architecture"]
    strategy = exp_cfg["strategy"]

    train_tf = RobustnessAwareAugmentation(input_size=INPUT_SIZE)
    eval_tf = build_eval_transform(input_size=INPUT_SIZE)

    train_ds = CachedRGBDataset(train_samples, transform=train_tf)
    train_eval_ds = CachedRGBDataset(train_samples, transform=eval_tf)
    val_ds = CachedRGBDataset(val_samples, transform=eval_tf)

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=gen,
    )
    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_stage2_model(
        architecture=arch,
        strategy=strategy,
        num_classes=len(CLASS_ORDER),
        pretrained=True,
    ).to(device)
    freeze_summary = get_layer_freeze_summary(model, arch, strategy)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    param_groups = get_parameter_groups(
        model,
        strategy=strategy,
        head_lr=exp_cfg["head_lr"],
        backbone_lr=exp_cfg["backbone_lr"],
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

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

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_train = 0
        tr_true, tr_pred = [], []

        for imgs, targets in train_loader:
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

        scheduler.step()

        # Compute train batch metrics cleanly
        from evaluate import compute_classification_metrics

        tr_metrics = compute_classification_metrics(tr_true, tr_pred, CLASS_ORDER)
        val_metrics = evaluate_clean_loader(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(running_loss / max(n_train, 1)))
        history["train_acc"].append(tr_metrics["accuracy"])
        history["train_macro_f1"].append(tr_metrics["macro_f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_acc"].append(val_metrics["accuracy"])

        improved = (val_metrics["macro_f1"] > best_f1 + 1e-6) or (
            abs(val_metrics["macro_f1"] - best_f1) <= 1e-6 and val_metrics["loss"] < best_loss
        )
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    train_time = time.perf_counter() - t0

    # Evaluate best checkpoint (selected strictly by clean Val Macro F1)
    model.load_state_dict(best_state)
    best_train_clean = evaluate_clean_loader(model, train_eval_loader, criterion, device)
    best_val_clean = evaluate_model_on_samples(
        model=model,
        samples=val_samples,
        device=device,
        eval_transform=eval_tf,
        criterion=criterion,
        corruption_mode=None,
        seed=seed,
        collect_errors=True,
    )
    robust_eval = evaluate_local_robustness(
        model=model,
        val_samples=val_samples,
        device=device,
        eval_transform=eval_tf,
        seed=seed,
    )
    latency = measure_inference_latency(model, input_shape=(1, 3, INPUT_SIZE, INPUT_SIZE))

    ckpt_payload = {
        "state_dict": best_state,
        "architecture": arch,
        "strategy": strategy,
        "pretrained_source": PRETRAINED_SOURCES[arch],
        "num_classes": len(CLASS_ORDER),
        "class_order": CLASS_ORDER,
        "input_size": [3, INPUT_SIZE, INPUT_SIZE],
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "seed": seed,
        "experiment_id": exp_cfg["exp_id"],
        "best_epoch": best_epoch,
        "clean_val_macro_f1": float(best_val_clean["macro_f1"]),
        "local_robust_macro_f1": float(robust_eval["robust_macro_f1"]),
    }

    result = {
        "exp_id": exp_cfg["exp_id"],
        "architecture": arch,
        "strategy": strategy,
        "pretrained_source": PRETRAINED_SOURCES[arch],
        "hyperparams": {
            "epochs": epochs,
            "batch_size": batch_size,
            "head_lr": exp_cfg["head_lr"],
            "backbone_lr": exp_cfg["backbone_lr"],
            "optimizer": "AdamW(weight_decay=1e-4)",
            "scheduler": "CosineAnnealingLR(eta_min=1e-6)",
            "label_smoothing": 0.05,
        },
        "layer_summary": freeze_summary,
        "best_epoch": best_epoch,
        "train_time_sec": float(train_time),
        "latency": latency,
        "clean_train": {
            "loss": best_train_clean["loss"],
            "accuracy": best_train_clean["accuracy"],
            "macro_f1": best_train_clean["macro_f1"],
        },
        "clean_val": best_val_clean,
        "robustness": robust_eval,
        "history": history,
    }
    return result, ckpt_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Transfer Learning Runner")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/STUDENT_PUBLIC/classification",
        help="Path to classification dataset directory containing train/ and validation/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="workspace_outputs/stage2",
        help="Directory to store Stage 2 metrics JSONs and intermediate checkpoints",
    )
    parser.add_argument(
        "--submission_dir",
        type=str,
        default="Team21_Stage2",
        help="Directory where final stage2_model.pth and config.json live",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    submission_dir = Path(args.submission_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples = collect_split_samples(data_dir / "train")
    val_samples = collect_split_samples(data_dir / "validation")
    print(f"[Stage 2] Device={device}, train={len(train_samples)}, val={len(val_samples)}, seed={args.seed}")

    experiments = [
        {"exp_id": "E1", "architecture": "densenet121", "strategy": "feature_extraction", "head_lr": 1e-3, "backbone_lr": 0.0},
        {"exp_id": "E2", "architecture": "densenet121", "strategy": "partial_finetuning", "head_lr": 1e-3, "backbone_lr": 3e-4},
        {"exp_id": "E3", "architecture": "densenet121", "strategy": "full_finetuning", "head_lr": 1e-3, "backbone_lr": 1.5e-4},
        {"exp_id": "E4", "architecture": "mobilenet_v3_large", "strategy": "feature_extraction", "head_lr": 1e-3, "backbone_lr": 0.0},
        {"exp_id": "E5", "architecture": "mobilenet_v3_large", "strategy": "partial_finetuning", "head_lr": 1e-3, "backbone_lr": 3e-4},
        {"exp_id": "E6", "architecture": "mobilenet_v3_large", "strategy": "full_finetuning", "head_lr": 1e-3, "backbone_lr": 1.5e-4},
    ]

    all_results = {}
    all_ckpts = {}

    for exp_cfg in experiments:
        eid = exp_cfg["exp_id"]
        print(f"[Stage 2] Running {eid}: {exp_cfg['architecture']} x {exp_cfg['strategy']}...")
        res, ckpt = train_single_experiment(
            exp_cfg=exp_cfg,
            train_samples=train_samples,
            val_samples=val_samples,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        ckpt_file = output_dir / "checkpoints" / f"{eid}.pth"
        torch.save(ckpt, ckpt_file)
        res["checkpoint_size_mb"] = float(ckpt_file.stat().st_size / (1024.0 * 1024.0))

        all_results[eid] = res
        all_ckpts[eid] = ckpt

        cv = res["clean_val"]
        rb = res["robustness"]
        print(
            f"  -> Best@{res['best_epoch']:2d}: Clean F1={cv['macro_f1']:.4f} (Acc={cv['accuracy']:.4f}), "
            f"Robust F1={rb['robust_macro_f1']:.4f} "
            f"[L={rb['lighting']['macro_f1']:.3f}, O={rb['optical']['macro_f1']:.3f}, S={rb['sensor_compression']['macro_f1']:.3f}], "
            f"Size={res['checkpoint_size_mb']:.1f}MB, CPU={res['latency']['cpu_ms']:.1f}ms"
        )

    # Select final model balancing Clean Macro F1 + Local Robust Macro F1 (composite score)
    best_id = max(
        all_results.keys(),
        key=lambda eid: (
            0.45 * all_results[eid]["clean_val"]["macro_f1"]
            + 0.55 * all_results[eid]["robustness"]["robust_macro_f1"],
            all_results[eid]["clean_val"]["macro_f1"],
        ),
    )

    final_ckpt_path = submission_dir / "stage2_model.pth"
    torch.save(all_ckpts[best_id], final_ckpt_path)
    final_size_mb = float(final_ckpt_path.stat().st_size / (1024.0 * 1024.0))

    winner = all_results[best_id]
    summary = {
        "selected_model_id": best_id,
        "selected_architecture": winner["architecture"],
        "selected_strategy": winner["strategy"],
        "selected_clean_val_macro_f1": winner["clean_val"]["macro_f1"],
        "selected_local_robust_macro_f1": winner["robustness"]["robust_macro_f1"],
        "checkpoint_size_mb": final_size_mb,
        "seed": args.seed,
        "experiments": all_results,
    }
    with open(output_dir / "stage2_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    config_payload = {
        "team_id": "Team21",
        "stage": 2,
        "architecture": winner["architecture"],
        "strategy": winner["strategy"],
        "pretrained_source": winner["pretrained_source"],
        "input_size": [3, INPUT_SIZE, INPUT_SIZE],
        "class_order": CLASS_ORDER,
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "seed": args.seed,
        "checkpoint_file": "stage2_model.pth",
    }
    with open(submission_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    print(
        f"[Stage 2] DONE! Selected {best_id} ({winner['architecture']} / {winner['strategy']}), "
        f"Clean F1={winner['clean_val']['macro_f1']:.4f}, "
        f"Local Robust F1={winner['robustness']['robust_macro_f1']:.4f}, "
        f"Saved to {final_ckpt_path} ({final_size_mb:.2f} MB)"
    )


if __name__ == "__main__":
    main()

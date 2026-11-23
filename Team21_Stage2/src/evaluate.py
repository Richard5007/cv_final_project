"""Evaluation and local robustness stress-testing for Stage 2 classification models."""

import io
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import transforms

from model import CLASS_ORDER, IMAGENET_MEAN, IMAGENET_STD


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_order: Optional[List[str]] = None,
) -> Dict:
    """Compute Accuracy, Macro Precision/Recall/F1, per-class metrics, and Confusion Matrix."""
    if class_order is None:
        class_order = CLASS_ORDER
    labels = list(range(len(class_order)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_order,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        cls_name: {
            "precision": float(report[cls_name]["precision"]),
            "recall": float(report[cls_name]["recall"]),
            "f1": float(report[cls_name]["f1-score"]),
            "support": int(report[cls_name]["support"]),
        }
        for cls_name in class_order
    }
    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def build_eval_transform(input_size: int = 224) -> transforms.Compose:
    """Fixed Stage 2 evaluation & inference preprocessing transform."""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def apply_lighting_corruption(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Simulate harsh factory illumination shifts: brightness/contrast/gamma + shading."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gamma = float(rng.uniform(0.45, 2.0))
    arr = np.clip(arr, 1e-4, 1.0) ** gamma

    alpha = float(rng.uniform(0.55, 1.45))  # contrast
    beta = float(rng.uniform(-0.20, 0.20))  # brightness
    arr = arr * alpha + beta

    # Linear uneven illumination gradient across X or Y
    h, w = arr.shape[:2]
    axis = int(rng.integers(0, 2))
    grad_strength = float(rng.uniform(-0.32, 0.32))
    if axis == 0:
        grad = np.linspace(1.0 - grad_strength, 1.0 + grad_strength, h, dtype=np.float32)[:, None, None]
    else:
        grad = np.linspace(1.0 - grad_strength, 1.0 + grad_strength, w, dtype=np.float32)[None, :, None]
    arr = np.clip(arr * grad, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")


def apply_optical_corruption(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Simulate optical defocus (Gaussian blur) and line-scan motion blur."""
    arr = np.asarray(img, dtype=np.uint8)
    sigma = float(rng.uniform(1.2, 2.2))
    ksize = int(rng.choice([5, 7]))
    arr = cv2.GaussianBlur(arr, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    # Add directional motion blur on ~65% of samples
    if rng.random() < 0.65:
        mk = 7
        kernel = np.zeros((mk, mk), dtype=np.float32)
        angle = float(rng.uniform(0.0, 180.0))
        if angle < 45 or angle >= 135:
            kernel[mk // 2, :] = 1.0 / mk
        else:
            kernel[:, mk // 2] = 1.0 / mk
        arr = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(arr, mode="RGB")


def apply_sensor_compression_corruption(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Simulate camera sensor read noise + aggressive JPEG compression artifacts."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    sigma = float(rng.uniform(0.04, 0.08))
    noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    noisy_img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")

    quality = int(rng.integers(15, 36))
    buf = io.BytesIO()
    noisy_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@torch.no_grad()
def evaluate_model_on_samples(
    model: nn.Module,
    samples: List[Tuple[Path, int]] if False else List[Tuple[object, int]],
    device: torch.device,
    eval_transform: transforms.Compose,
    criterion: Optional[nn.Module] = None,
    corruption_mode: Optional[str] = None,
    seed: int = 42,
    batch_size: int = 32,
    collect_errors: bool = False,
) -> Dict:
    """Evaluate model cleanly or under deterministic in-memory stress-test corruption."""
    model.eval()
    rng = np.random.default_rng(seed)

    tensors = []
    targets = []
    paths = []

    for path, label in samples:
        img = Image.open(path).convert("RGB")
        if corruption_mode == "lighting":
            img = apply_lighting_corruption(img, rng)
        elif corruption_mode == "optical":
            img = apply_optical_corruption(img, rng)
        elif corruption_mode == "sensor_compression":
            img = apply_sensor_compression_corruption(img, rng)
        elif corruption_mode is not None:
            raise ValueError(f"Unknown corruption mode: {corruption_mode}")

        tensors.append(eval_transform(img))
        targets.append(label)
        paths.append(str(path))

    total_loss = 0.0
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []

    for i in range(0, len(tensors), batch_size):
        xb = torch.stack(tensors[i : i + batch_size], dim=0).to(device)
        yb = torch.tensor(targets[i : i + batch_size], dtype=torch.long, device=device)
        logits = model(xb)
        if criterion is not None:
            total_loss += float(criterion(logits, yb).item()) * yb.size(0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_prob.append(probs)
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    metrics = compute_classification_metrics(targets, y_pred, CLASS_ORDER)
    if criterion is not None:
        metrics["loss"] = float(total_loss / max(len(targets), 1))

    if collect_errors:
        probs_all = np.concatenate(y_prob, axis=0)
        errors = []
        for idx, (t, p) in enumerate(zip(targets, y_pred)):
            if t != p:
                errors.append(
                    {
                        "path": paths[idx],
                        "true_label": CLASS_ORDER[t],
                        "pred_label": CLASS_ORDER[p],
                        "pred_confidence": float(probs_all[idx, p]),
                        "true_confidence": float(probs_all[idx, t]),
                    }
                )
        errors.sort(key=lambda x: x["pred_confidence"], reverse=True)
        metrics["errors"] = errors

    return metrics


@torch.no_grad()
def evaluate_clean_loader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    """Fast dataloader evaluation during training epochs."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for imgs, targets in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, targets)
        bs = targets.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs
        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, CLASS_ORDER)
    metrics["loss"] = float(total_loss / max(total_samples, 1))
    return metrics


@torch.no_grad()
def evaluate_local_robustness(
    model: nn.Module,
    val_samples: List[Tuple[object, int]],
    device: torch.device,
    eval_transform: transforms.Compose,
    seed: int = 42,
) -> Dict:
    """Run local robustness stress-test across lighting, optical, and sensor/compression."""
    sub_results = {}
    for mode, mode_seed_offset in [
        ("lighting", 101),
        ("optical", 202),
        ("sensor_compression", 303),
    ]:
        sub_results[mode] = evaluate_model_on_samples(
            model=model,
            samples=val_samples,
            device=device,
            eval_transform=eval_transform,
            criterion=None,
            corruption_mode=mode,
            seed=seed + mode_seed_offset,
            collect_errors=False,
        )

    robust_macro_f1 = float(
        (
            sub_results["lighting"]["macro_f1"]
            + sub_results["optical"]["macro_f1"]
            + sub_results["sensor_compression"]["macro_f1"]
        )
        / 3.0
    )
    robust_acc = float(
        (
            sub_results["lighting"]["accuracy"]
            + sub_results["optical"]["accuracy"]
            + sub_results["sensor_compression"]["accuracy"]
        )
        / 3.0
    )
    agg_cm = (
        np.array(sub_results["lighting"]["confusion_matrix"])
        + np.array(sub_results["optical"]["confusion_matrix"])
        + np.array(sub_results["sensor_compression"]["confusion_matrix"])
    ).tolist()

    return {
        "robust_macro_f1": robust_macro_f1,
        "robust_accuracy": robust_acc,
        "aggregated_confusion_matrix": agg_cm,
        "lighting": sub_results["lighting"],
        "optical": sub_results["optical"],
        "sensor_compression": sub_results["sensor_compression"],
    }


@torch.no_grad()
def measure_inference_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    warmup: int = 20,
    iters: int = 100,
) -> Dict[str, Optional[float]]:
    """Measure single-image inference latency (ms/image) on CPU and GPU."""
    results: Dict[str, Optional[float]] = {"cpu_ms": None, "gpu_ms": None}

    model_cpu = model.to("cpu").eval()
    x_cpu = torch.randn(*input_shape, device="cpu")
    for _ in range(min(10, warmup)):
        _ = model_cpu(x_cpu)
    t0 = time.perf_counter()
    for _ in range(min(40, iters)):
        _ = model_cpu(x_cpu)
    t1 = time.perf_counter()
    results["cpu_ms"] = float((t1 - t0) * 1000.0 / min(40, iters))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_gpu = model.to(device).eval()
        x_gpu = torch.randn(*input_shape, device=device)
        for _ in range(warmup):
            _ = model_gpu(x_gpu)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model_gpu(x_gpu)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        results["gpu_ms"] = float((t1 - t0) * 1000.0 / iters)

    return results

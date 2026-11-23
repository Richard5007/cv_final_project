"""Evaluation utilities for Stage 1 classification models."""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from model import CLASS_ORDER


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


@torch.no_grad()
def evaluate_cnn(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_order: Optional[List[str]] = None,
    collect_errors: bool = False,
) -> Dict:
    """Evaluate a PyTorch classification model on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    sample_paths: List[str] = []

    for batch in dataloader:
        if len(batch) == 3:
            imgs, targets, paths = batch
            sample_paths.extend(list(paths))
        else:
            imgs, targets = batch
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, targets)

        bs = targets.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.append(probs)

    metrics = compute_classification_metrics(y_true, y_pred, class_order=class_order)
    metrics["loss"] = float(total_loss / max(total_samples, 1))

    if collect_errors and sample_paths:
        probs_all = np.concatenate(y_prob, axis=0)
        cls_list = class_order or CLASS_ORDER
        errors = []
        for idx, (t, p) in enumerate(zip(y_true, y_pred)):
            if t != p:
                errors.append(
                    {
                        "path": sample_paths[idx],
                        "true_label": cls_list[t],
                        "pred_label": cls_list[p],
                        "pred_confidence": float(probs_all[idx, p]),
                        "true_confidence": float(probs_all[idx, t]),
                    }
                )
        errors.sort(key=lambda x: x["pred_confidence"], reverse=True)
        metrics["errors"] = errors

    return metrics


@torch.no_grad()
def measure_inference_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 1, 200, 200),
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
    for _ in range(iters):
        _ = model_cpu(x_cpu)
    t1 = time.perf_counter()
    results["cpu_ms"] = float((t1 - t0) * 1000.0 / iters)

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

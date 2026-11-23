"""Stage 3 Training & Controlled Comparison: YOLO11s, YOLO11m, Faster R-CNN ResNet50-FPN-v2."""

import argparse
import copy
import json
import os
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from ultralytics import YOLO

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import (
    CLASS_ORDER,
    categorize_failures,
    evaluate_detections_voc,
    parse_voc_annotations,
    validate_detections_schema,
)
from model import build_faster_rcnn, clamp_and_filter_boxes


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SteelDetectionVOCDataset(Dataset):
    """Detection dataset reading PNG images and Pascal VOC XML annotations."""

    def __init__(self, split_dir: Path, hflip_p: float = 0.0, vflip_p: float = 0.0) -> None:
        self.img_dir = split_dir / "images"
        self.voc_dir = split_dir / "annotations_voc"
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.cls_to_idx = {c: i + 1 for i, c in enumerate(CLASS_ORDER)}  # 0 reserved for background
        self.items: List[Tuple[Path, Path]] = []

        for img_path in sorted(self.img_dir.glob("*.png")):
            xml_path = self.voc_dir / f"{img_path.stem}.xml"
            if xml_path.exists():
                self.items.append((img_path, xml_path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall("object"):
            raw_name = obj.find("name").text.strip()
            cname = "rolled_in_scale" if raw_name == "rolled-in_scale" else raw_name
            bnd = obj.find("bndbox")
            x1 = max(0.0, float(bnd.find("xmin").text))
            y1 = max(0.0, float(bnd.find("ymin").text))
            x2 = min(float(w), float(bnd.find("xmax").text))
            y2 = min(float(h), float(bnd.find("ymax").text))
            if x2 > x1 + 1.0 and y2 > y1 + 1.0:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cls_to_idx[cname])

        tensor = TF.to_tensor(img)
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        if self.hflip_p > 0 and random.random() < self.hflip_p:
            tensor = TF.hflip(tensor)
            if len(boxes_t) > 0:
                x1_old = boxes_t[:, 0].clone()
                x2_old = boxes_t[:, 2].clone()
                boxes_t[:, 0] = float(w) - x2_old
                boxes_t[:, 2] = float(w) - x1_old

        if self.vflip_p > 0 and random.random() < self.vflip_p:
            tensor = TF.vflip(tensor)
            if len(boxes_t) > 0:
                y1_old = boxes_t[:, 1].clone()
                y2_old = boxes_t[:, 3].clone()
                boxes_t[:, 1] = float(h) - y2_old
                boxes_t[:, 3] = float(h) - y1_old

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "stem": img_path.stem,
            "size": (w, h),
        }
        return tensor, target


def collate_detection(batch):
    return tuple(zip(*batch))


def predict_yolo_on_dir(
    weights_path: Path,
    val_img_dir: Path,
    conf_thresh: float = 0.01,
    nms_iou: float = 0.50,
    imgsz: int = 640,
    device: str = "0",
) -> Tuple[List[Dict], Dict[str, float]]:
    """Run YOLO prediction on val_img_dir and measure GPU & CPU latency."""
    model = YOLO(str(weights_path))
    img_paths = sorted(val_img_dir.glob("*.png"))

    # Warmup on GPU if available
    if torch.cuda.is_available() and device != "cpu":
        _ = model.predict(source=str(img_paths[0]), imgsz=imgsz, conf=conf_thresh, iou=nms_iou, max_det=100, device=device, verbose=False)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    predictions: List[Dict] = []
    batch_size = 32
    for i in range(0, len(img_paths), batch_size):
        chunk = img_paths[i : i + batch_size]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=imgsz,
            conf=conf_thresh,
            iou=nms_iou,
            max_det=100,
            device=device,
            verbose=False,
            stream=False,
        )
        for p, res in zip(chunk, results):
            h, w = res.orig_shape
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy().tolist()
                confs = boxes.conf.cpu().numpy().tolist()
                clss = [int(x) for x in boxes.cls.cpu().numpy().tolist()]
            else:
                xyxy, confs, clss = [], [], []
            dets = clamp_and_filter_boxes(
                xyxy, confs, clss, width=w, height=h, conf_thresh=conf_thresh, max_det=100, label_offset=0
            )
            predictions.append({"image_id": p.stem, "detections": dets})

    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()
    gpu_total_s = time.perf_counter() - t0
    gpu_ms = (gpu_total_s * 1000.0) / len(img_paths)

    # CPU latency probe on 30 images
    cpu_sample = [str(p) for p in img_paths[:30]]
    _ = model.predict(source=cpu_sample[0], imgsz=imgsz, conf=conf_thresh, iou=nms_iou, max_det=100, device="cpu", verbose=False)
    tc0 = time.perf_counter()
    _ = model.predict(source=cpu_sample, imgsz=imgsz, conf=conf_thresh, iou=nms_iou, max_det=100, device="cpu", verbose=False)
    cpu_ms = ((time.perf_counter() - tc0) * 1000.0) / len(cpu_sample)

    return predictions, {"gpu_ms": float(gpu_ms), "cpu_ms": float(cpu_ms)}


@torch.no_grad()
def predict_faster_rcnn_on_loader(
    model: torch.nn.Module,
    val_img_dir: Path,
    device: torch.device,
    conf_thresh: float = 0.05,
    measure_cpu: bool = True,
) -> Tuple[List[Dict], Dict[str, float]]:
    """Run Faster R-CNN inference on validation images and measure latency."""
    model.eval().to(device)
    img_paths = sorted(val_img_dir.glob("*.png"))

    # Warmup
    if device.type == "cuda":
        w_img = TF.to_tensor(Image.open(img_paths[0]).convert("RGB")).to(device)
        _ = model([w_img])
        torch.cuda.synchronize()

    predictions: List[Dict] = []
    t0 = time.perf_counter()
    for p in img_paths:
        pil_img = Image.open(p).convert("RGB")
        w, h = pil_img.size
        xt = TF.to_tensor(pil_img).to(device)
        out = model([xt])[0]
        boxes = out["boxes"].cpu().numpy().tolist()
        scores = out["scores"].cpu().numpy().tolist()
        labels = out["labels"].cpu().numpy().tolist()
        dets = clamp_and_filter_boxes(
            boxes, scores, labels, width=w, height=h, conf_thresh=conf_thresh, max_det=100, label_offset=1
        )
        predictions.append({"image_id": p.stem, "detections": dets})

    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_ms = ((time.perf_counter() - t0) * 1000.0) / len(img_paths)

    cpu_ms = None
    if measure_cpu:
        model_cpu = copy.deepcopy(model).to("cpu").eval()
        sample_imgs = img_paths[:15]
        _ = model_cpu([TF.to_tensor(Image.open(sample_imgs[0]).convert("RGB"))])
        tc0 = time.perf_counter()
        for p in sample_imgs:
            _ = model_cpu([TF.to_tensor(Image.open(p).convert("RGB"))])
        cpu_ms = ((time.perf_counter() - tc0) * 1000.0) / len(sample_imgs)

    return predictions, {"gpu_ms": float(gpu_ms), "cpu_ms": float(cpu_ms) if cpu_ms is not None else None}


def train_yolo_variant(
    variant_name: str,
    pretrained_pt: Path,
    data_yaml: Path,
    val_img_dir: Path,
    gt_by_image: Dict,
    output_dir: Path,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    seed: int = 42,
    reuse_existing: bool = False,
) -> Dict:
    """Train one Ultralytics YOLO11 variant and evaluate via exact Slide-31 VOC mAP@0.5."""
    set_seed(seed)
    runs_dir = (output_dir / "yolo_runs").resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    save_dir = runs_dir / variant_name
    existing_best = save_dir / "weights" / "best.pt"

    if reuse_existing and existing_best.is_file():
        print(f"  [Reuse] Found existing {variant_name} weights at {existing_best}")
        best_pt = existing_best
        results_csv = save_dir / "results.csv"
        train_time = 0.0
        if results_csv.exists():
            import csv as _csv
            with open(results_csv, "r", encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
            if rows:
                last = {k.strip(): v.strip() for k, v in rows[-1].items()}
                train_time = float(last.get("time", 0.0))
    else:
        model = YOLO(str(pretrained_pt))
        t0 = time.perf_counter()
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            seed=seed,
            deterministic=True,
            fliplr=0.5,
            flipud=0.5,
            mosaic=1.0,
            close_mosaic=15,
            hsv_v=0.3,
            scale=0.4,
            patience=25,
            project=str(runs_dir),
            name=variant_name,
            exist_ok=True,
            verbose=False,
            plots=False,
        )
        train_time = time.perf_counter() - t0

        save_dir = Path(model.trainer.save_dir)
        best_pt = Path(model.trainer.best)
        if not best_pt.is_file():
            best_pt = save_dir / "weights" / "best.pt"
        results_csv = save_dir / "results.csv"

    # Extract parameter count
    loaded = YOLO(str(best_pt))
    total_params = sum(p.numel() for p in loaded.model.parameters())

    # Evaluate at low conf=0.01 (for full PR curve) and operating conf=0.25
    preds_raw, latency = predict_yolo_on_dir(
        best_pt, val_img_dir, conf_thresh=0.01, nms_iou=0.50, imgsz=imgsz, device="0" if torch.cuda.is_available() else "cpu"
    )
    schema_errs = validate_detections_schema(preds_raw, gt_by_image)
    if schema_errs:
        raise RuntimeError(f"Schema validation failed for {variant_name}: {schema_errs[:5]}")

    metrics_c001 = evaluate_detections_voc(preds_raw, gt_by_image, iou_threshold=0.5, eval_conf_threshold=0.01)
    metrics_c010 = evaluate_detections_voc(preds_raw, gt_by_image, iou_threshold=0.5, eval_conf_threshold=0.10)
    metrics_c025 = evaluate_detections_voc(preds_raw, gt_by_image, iou_threshold=0.5, eval_conf_threshold=0.25)

    # Read training curve CSV
    history = {"epoch": [], "box_loss": [], "cls_loss": [], "dfl_loss": [], "ultralytics_map50": []}
    if results_csv.exists():
        import csv as _csv
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                history["epoch"].append(int(float(row["epoch"])))
                history["box_loss"].append(float(row.get("train/box_loss", 0.0)))
                history["cls_loss"].append(float(row.get("train/cls_loss", 0.0)))
                history["dfl_loss"].append(float(row.get("train/dfl_loss", 0.0)))
                history["ultralytics_map50"].append(float(row.get("metrics/mAP50(B)", 0.0)))

    return {
        "model_id": variant_name,
        "detector": variant_name.upper(),
        "detector_family": "yolo11",
        "pretrained_source": f"Ultralytics COCO ({pretrained_pt.name})",
        "weights_path": str(best_pt),
        "file_size_mb": float(best_pt.stat().st_size / (1024.0 * 1024.0)),
        "total_params": int(total_params),
        "input_size": imgsz,
        "train_time_sec": float(train_time),
        "latency": latency,
        "eval_conf001": metrics_c001,
        "eval_conf010": metrics_c010,
        "eval_conf025": metrics_c025,
        "history": history,
    }


def train_faster_rcnn(
    det_root: Path,
    gt_by_image: Dict,
    output_dir: Path,
    epochs: int = 25,
    batch_size: int = 4,
    lr: float = 1e-4,
    seed: int = 42,
    reuse_existing: bool = False,
) -> Dict:
    """Train Torchvision Faster R-CNN ResNet50-FPN-v2 and evaluate via Slide-31 VOC mAP@0.5."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_pt = ckpt_dir / "faster_rcnn_r50_fpn_v2.pt"
    hist_path = ckpt_dir / "faster_rcnn_history.json"
    val_img_dir = det_root / "validation" / "images"

    if reuse_existing and best_pt.is_file():
        print(f"  [Reuse] Found existing Faster R-CNN checkpoint at {best_pt}")
        ckpt = torch.load(best_pt, map_location=device, weights_only=True)
        model = build_faster_rcnn(
            num_defect_classes=len(CLASS_ORDER),
            pretrained=False,
            min_size=int(ckpt.get("min_size", 600)),
            max_size=int(ckpt.get("max_size", 800)),
            box_score_thresh=0.005,
            box_nms_thresh=0.50,
            box_detections_per_img=100,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        total_params = sum(p.numel() for p in model.parameters())
        best_state = ckpt["state_dict"]
        best_epoch = int(ckpt.get("best_epoch", epochs))
        if hist_path.is_file():
            meta = json.loads(hist_path.read_text(encoding="utf-8"))
            history = meta["history"]
            train_time = float(meta["train_time_sec"])
        else:
            history = {"epoch": [], "train_loss": [], "val_map50": []}
            train_time = 0.0
    else:
        train_ds = SteelDetectionVOCDataset(det_root / "train", hflip_p=0.5, vflip_p=0.5)
        gen = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_detection,
            generator=gen,
        )

        model = build_faster_rcnn(
            num_defect_classes=len(CLASS_ORDER),
            pretrained=True,
            min_size=600,
            max_size=800,
            box_score_thresh=0.005,
            box_nms_thresh=0.50,
            box_detections_per_img=100,
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters())

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        best_map = -1.0
        best_epoch = 0
        best_state = None
        history = {"epoch": [], "train_loss": [], "val_map50": []}

        t0 = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0

            if epoch == 1:
                warmup_iters = max(1, len(train_loader))
                lr_scheduler_warm = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup_iters
                )
            else:
                lr_scheduler_warm = None

            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets_gpu = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    loss_dict = model(images, targets_gpu)
                    losses = sum(loss for loss in loss_dict.values())

                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

                if lr_scheduler_warm is not None:
                    lr_scheduler_warm.step()

                running_loss += float(losses.item())
                n_batches += 1

            if epoch > 1:
                scheduler.step()

            avg_loss = running_loss / max(n_batches, 1)
            preds_val, _ = predict_faster_rcnn_on_loader(
                model, val_img_dir, device, conf_thresh=0.05, measure_cpu=False
            )
            val_metrics = evaluate_detections_voc(preds_val, gt_by_image, iou_threshold=0.5, eval_conf_threshold=0.05)
            val_map = val_metrics["mAP50"]

            history["epoch"].append(epoch)
            history["train_loss"].append(float(avg_loss))
            history["val_map50"].append(float(val_map))

            print(f"  [FasterRCNN] Epoch {epoch:2d}/{epochs}: loss={avg_loss:.4f}, val_mAP@0.5={val_map:.4f}")
            if val_map > best_map:
                best_map = val_map
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        train_time = time.perf_counter() - t0
        torch.save(
            {
                "state_dict": best_state,
                "detector_family": "faster_rcnn",
                "min_size": 600,
                "max_size": 800,
                "num_classes": len(CLASS_ORDER),
                "class_order": CLASS_ORDER,
                "best_epoch": best_epoch,
                "seed": seed,
            },
            best_pt,
        )
        hist_path.write_text(json.dumps({"history": history, "train_time_sec": float(train_time), "best_epoch": best_epoch}, indent=2), encoding="utf-8")

    model.load_state_dict(best_state)
    preds_raw, latency = predict_faster_rcnn_on_loader(
        model, val_img_dir, device, conf_thresh=0.01, measure_cpu=True
    )
    metrics_c001 = evaluate_detections_voc(preds_raw, gt_by_image, iou_threshold=0.5, eval_conf_threshold=0.01)
    metrics_c010 = evaluate_detections_voc(preds_raw, gt_by_image, iou_threshold=0.5, eval_conf_threshold=0.10)
    metrics_c025 = evaluate_detections_voc(preds_raw, gt_by_image, iou_threshold=0.5, eval_conf_threshold=0.25)

    return {
        "model_id": "faster_rcnn_r50_fpn_v2",
        "detector": "Faster R-CNN ResNet50-FPN-v2",
        "detector_family": "faster_rcnn",
        "pretrained_source": "torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        "weights_path": str(best_pt),
        "file_size_mb": float(best_pt.stat().st_size / (1024.0 * 1024.0)),
        "total_params": int(total_params),
        "input_size": "600-800",
        "best_epoch": best_epoch,
        "train_time_sec": float(train_time),
        "latency": latency,
        "eval_conf001": metrics_c001,
        "eval_conf010": metrics_c010,
        "eval_conf025": metrics_c025,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 Object Detection Training & Comparison")
    parser.add_argument("--det_root", type=str, default="data/STUDENT_PUBLIC/detection")
    parser.add_argument("--yolo_yaml", type=str, default="data/yolo_dataset/data.yaml")
    parser.add_argument("--output_dir", type=str, default="workspace_outputs/stage3")
    parser.add_argument("--submission_dir", type=str, default="Team21_Stage3")
    parser.add_argument("--yolo_epochs", type=int, default=100)
    parser.add_argument("--frcnn_epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse_yolo", action="store_true")
    parser.add_argument("--reuse_frcnn", action="store_true")
    args = parser.parse_args()

    det_root = Path(args.det_root).resolve()
    yolo_yaml = Path(args.yolo_yaml).resolve()
    output_dir = Path(args.output_dir).resolve()
    submission_dir = Path(args.submission_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    pretrained_dir = output_dir / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    val_img_dir = det_root / "validation" / "images"
    gt_by_image = parse_voc_annotations(det_root / "validation" / "annotations_voc")
    print(f"[Stage 3] Val images={len(gt_by_image)}, seed={args.seed}")

    # Download base COCO weights cleanly into pretrained_dir if absent
    if not (pretrained_dir / "yolo11s.pt").is_file() or not (pretrained_dir / "yolo11m.pt").is_file():
        cwd_orig = Path.cwd()
        os.chdir(pretrained_dir)
        try:
            _ = YOLO("yolo11s.pt")
            _ = YOLO("yolo11m.pt")
        finally:
            os.chdir(cwd_orig)

    detectors = {}

    print("[Stage 3] Training Detector 1: YOLO11s...")
    detectors["yolo11s"] = train_yolo_variant(
        variant_name="yolo11s",
        pretrained_pt=pretrained_dir / "yolo11s.pt",
        data_yaml=yolo_yaml,
        val_img_dir=val_img_dir,
        gt_by_image=gt_by_image,
        output_dir=output_dir,
        epochs=args.yolo_epochs,
        batch=16,
        imgsz=640,
        seed=args.seed,
        reuse_existing=args.reuse_yolo,
    )
    print(
        f"  -> YOLO11s mAP@0.5(c=0.01)={detectors['yolo11s']['eval_conf001']['mAP50']:.4f}, "
        f"mAP@0.5(c=0.10)={detectors['yolo11s']['eval_conf010']['mAP50']:.4f}, "
        f"GPU={detectors['yolo11s']['latency']['gpu_ms']:.2f}ms"
    )

    print("[Stage 3] Training Detector 2: YOLO11m...")
    detectors["yolo11m"] = train_yolo_variant(
        variant_name="yolo11m",
        pretrained_pt=pretrained_dir / "yolo11m.pt",
        data_yaml=yolo_yaml,
        val_img_dir=val_img_dir,
        gt_by_image=gt_by_image,
        output_dir=output_dir,
        epochs=args.yolo_epochs,
        batch=8,
        imgsz=640,
        seed=args.seed,
        reuse_existing=args.reuse_yolo,
    )
    print(
        f"  -> YOLO11m mAP@0.5(c=0.01)={detectors['yolo11m']['eval_conf001']['mAP50']:.4f}, "
        f"mAP@0.5(c=0.10)={detectors['yolo11m']['eval_conf010']['mAP50']:.4f}, "
        f"GPU={detectors['yolo11m']['latency']['gpu_ms']:.2f}ms"
    )

    print("[Stage 3] Training Detector 3: Faster R-CNN ResNet50-FPN-v2...")
    detectors["faster_rcnn"] = train_faster_rcnn(
        det_root=det_root,
        gt_by_image=gt_by_image,
        output_dir=output_dir,
        epochs=args.frcnn_epochs,
        batch_size=4,
        lr=1e-4,
        seed=args.seed,
        reuse_existing=args.reuse_frcnn,
    )
    print(
        f"  -> FasterRCNN mAP@0.5(c=0.01)={detectors['faster_rcnn']['eval_conf001']['mAP50']:.4f}, "
        f"mAP@0.5(c=0.10)={detectors['faster_rcnn']['eval_conf010']['mAP50']:.4f}, "
        f"GPU={detectors['faster_rcnn']['latency']['gpu_ms']:.2f}ms"
    )

    # Select winning detector by highest Validation mAP@0.5 (full PR curve at c=0.01), tie-broken by c=0.10 and speed
    winner_id = max(
        detectors.keys(),
        key=lambda k: (
            round(detectors[k]["eval_conf001"]["mAP50"], 4),
            round(detectors[k]["eval_conf010"]["mAP50"], 4),
            -detectors[k]["latency"]["gpu_ms"],
        ),
    )
    winner = detectors[winner_id]
    print(f"[Stage 3] Winning detector: {winner['detector']} ({winner_id})")

    # Threshold & NMS IoU sensitivity sweep on the winning detector
    conf_sweep = []
    nms_sweep = []
    best_weights = Path(winner["weights_path"])
    conf_grid = [0.005, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    if winner["detector_family"] == "yolo11":
        # Full confidence sweep at NMS IoU=0.50
        raw_preds_50, _ = predict_yolo_on_dir(
            best_weights, val_img_dir, conf_thresh=0.005, nms_iou=0.50, imgsz=640, device="0" if torch.cuda.is_available() else "cpu"
        )
        for c_thr in conf_grid:
            ev = evaluate_detections_voc(raw_preds_50, gt_by_image, iou_threshold=0.5, eval_conf_threshold=c_thr)
            avg_det = sum(sum(1 for d in p["detections"] if d["confidence"] >= c_thr) for p in raw_preds_50) / len(raw_preds_50)
            conf_sweep.append(
                {
                    "conf_threshold": c_thr,
                    "mAP50": ev["mAP50"],
                    "precision": ev["precision"],
                    "recall": ev["recall"],
                    "f1": ev["f1"],
                    "avg_dets_per_img": float(avg_det),
                }
            )
        # NMS IoU sweep at conf=0.05
        for iou_thr in [0.35, 0.45, 0.50, 0.55, 0.65, 0.75]:
            p_iou, _ = predict_yolo_on_dir(
                best_weights, val_img_dir, conf_thresh=0.05, nms_iou=iou_thr, imgsz=640, device="0" if torch.cuda.is_available() else "cpu"
            )
            ev = evaluate_detections_voc(p_iou, gt_by_image, iou_threshold=0.5, eval_conf_threshold=0.05)
            nms_sweep.append(
                {
                    "nms_iou_threshold": iou_thr,
                    "mAP50": ev["mAP50"],
                    "precision": ev["precision"],
                    "recall": ev["recall"],
                    "f1": ev["f1"],
                }
            )
        best_nms_iou = max(nms_sweep, key=lambda x: (round(x["mAP50"], 4), x["f1"]))["nms_iou_threshold"]
        # Pick operating confidence threshold preserving >= 99.5% of peak mAP@0.5 (within 0.005) with highest F1
        peak_map = max(x["mAP50"] for x in conf_sweep)
        candidates = [x for x in conf_sweep if x["mAP50"] >= peak_map - 0.005]
        chosen_conf_entry = max(candidates, key=lambda x: (x["f1"], x["mAP50"]))
        best_conf = chosen_conf_entry["conf_threshold"]

        final_val_preds, final_latency = predict_yolo_on_dir(
            best_weights, val_img_dir, conf_thresh=best_conf, nms_iou=best_nms_iou, imgsz=640, device="0" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(best_weights, map_location=device, weights_only=True)
        frcnn_model = build_faster_rcnn(
            num_defect_classes=len(CLASS_ORDER),
            pretrained=False,
            min_size=600,
            max_size=800,
            box_score_thresh=0.005,
            box_nms_thresh=0.50,
            box_detections_per_img=100,
        ).to(device)
        frcnn_model.load_state_dict(ckpt["state_dict"])
        raw_preds_50, _ = predict_faster_rcnn_on_loader(frcnn_model, val_img_dir, device, conf_thresh=0.005, measure_cpu=False)
        for c_thr in conf_grid:
            ev = evaluate_detections_voc(raw_preds_50, gt_by_image, iou_threshold=0.5, eval_conf_threshold=c_thr)
            avg_det = sum(sum(1 for d in p["detections"] if d["confidence"] >= c_thr) for p in raw_preds_50) / len(raw_preds_50)
            conf_sweep.append(
                {
                    "conf_threshold": c_thr,
                    "mAP50": ev["mAP50"],
                    "precision": ev["precision"],
                    "recall": ev["recall"],
                    "f1": ev["f1"],
                    "avg_dets_per_img": float(avg_det),
                }
            )
        for iou_thr in [0.35, 0.45, 0.50, 0.55, 0.65]:
            frcnn_model.roi_heads.nms_thresh = iou_thr
            p_iou, _ = predict_faster_rcnn_on_loader(frcnn_model, val_img_dir, device, conf_thresh=0.01, measure_cpu=False)
            ev = evaluate_detections_voc(p_iou, gt_by_image, iou_threshold=0.5, eval_conf_threshold=0.01)
            nms_sweep.append(
                {
                    "nms_iou_threshold": iou_thr,
                    "mAP50": ev["mAP50"],
                    "precision": ev["precision"],
                    "recall": ev["recall"],
                    "f1": ev["f1"],
                }
            )
        best_nms_iou = max(nms_sweep, key=lambda x: (round(x["mAP50"], 4), x["f1"]))["nms_iou_threshold"]
        frcnn_model.roi_heads.nms_thresh = best_nms_iou
        peak_map = max(x["mAP50"] for x in conf_sweep)
        candidates = [x for x in conf_sweep if x["mAP50"] >= peak_map - 0.003]
        chosen_conf_entry = max(candidates, key=lambda x: (x["f1"], x["mAP50"]))
        best_conf = chosen_conf_entry["conf_threshold"]
        final_val_preds, final_latency = predict_faster_rcnn_on_loader(frcnn_model, val_img_dir, device, conf_thresh=best_conf, measure_cpu=True)

    final_val_eval = evaluate_detections_voc(final_val_preds, gt_by_image, iou_threshold=0.5)
    failure_conf_thr = max(0.25, best_conf)
    failures = categorize_failures(final_val_preds, gt_by_image, conf_threshold=failure_conf_thr, iou_threshold=0.5)

    # Save winning checkpoint as submission stage3_model.pt
    final_pt_path = submission_dir / "stage3_model.pt"
    shutil.copy2(best_weights, final_pt_path)

    with open(output_dir / "val_predictions.json", "w", encoding="utf-8") as f:
        json.dump(final_val_preds, f, indent=2)

    stage3_summary = {
        "selected_model_id": winner_id,
        "selected_detector": winner["detector"],
        "detector_family": winner["detector_family"],
        "confidence_threshold": best_conf,
        "nms_iou_threshold": best_nms_iou,
        "failure_analysis_conf_threshold": failure_conf_thr,
        "final_val_metrics": final_val_eval,
        "final_latency": final_latency,
        "checkpoint_size_mb": float(final_pt_path.stat().st_size / (1024.0 * 1024.0)),
        "conf_sweep": conf_sweep,
        "nms_sweep": nms_sweep,
        "failure_counts": {k: len(v) for k, v in failures.items()},
        "failures": {k: v[:15] for k, v in failures.items()},
        "detectors": detectors,
    }
    with open(output_dir / "stage3_results.json", "w", encoding="utf-8") as f:
        json.dump(stage3_summary, f, indent=2)

    config_payload = {
        "team_id": "Team21",
        "stage": 3,
        "detector": winner["detector"],
        "pretrained_source": winner["pretrained_source"],
        "input_size": winner["input_size"],
        "class_order": CLASS_ORDER,
        "confidence_threshold": best_conf,
        "nms_iou_threshold": best_nms_iou,
        "bbox_format": "xyxy_pixel",
        "seed": args.seed,
        "checkpoint_file": "stage3_model.pt",
    }
    with open(submission_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    print(
        f"[Stage 3] DONE! Selected={winner['detector']}, conf={best_conf}, nms_iou={best_nms_iou}, "
        f"Val mAP@0.5={final_val_eval['mAP50']:.4f}, P={final_val_eval['precision']:.4f}, R={final_val_eval['recall']:.4f}"
    )


if __name__ == "__main__":
    main()

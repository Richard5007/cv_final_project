"""Stage 3 Benchmark Object Detection Script (Offline, Single Checkpoint, Slide-30 JSON)."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

os.environ.setdefault("ULTRALYTICS_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_cfg"))

import torch
from PIL import Image
from torchvision.transforms import functional as TF

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import CLASS_ORDER, build_faster_rcnn, clamp_and_filter_boxes

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(input_dir: Path) -> List[Path]:
    """Collect image files inside input_dir deterministically."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        raise RuntimeError(f"No image files found in: {input_dir}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 Steel Defect Object Detection")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of test images")
    parser.add_argument("--output_json", type=str, required=True, help="Output detections JSON path")
    args = parser.parse_args()

    config_path = BASE_DIR / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    ckpt_path = BASE_DIR / cfg.get("checkpoint_file", "stage3_model.pt")
    conf_thresh = float(cfg.get("confidence_threshold", 0.10))
    nms_iou = cfg.get("nms_iou_threshold", 0.50)
    detector_name = str(cfg.get("detector", "YOLO11s")).lower()
    input_size = cfg.get("input_size", 640)

    image_paths = list_images(Path(args.input_dir))
    outputs = []

    if "yolo" in detector_name:
        from ultralytics import YOLO

        device_arg = "0" if torch.cuda.is_available() else "cpu"
        model = YOLO(str(ckpt_path))
        imgsz = int(input_size[0] if isinstance(input_size, list) else input_size)
        iou_val = float(nms_iou) if nms_iou is not None else 0.50

        batch_size = 32
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            results = model.predict(
                source=[str(p) for p in batch_paths],
                imgsz=imgsz,
                conf=conf_thresh,
                iou=iou_val,
                max_det=100,
                device=device_arg,
                verbose=False,
                stream=False,
            )
            for img_path, res in zip(batch_paths, results):
                h, w = res.orig_shape
                boxes = res.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy().tolist()
                    confs = boxes.conf.cpu().numpy().tolist()
                    clss = [int(x) for x in boxes.cls.cpu().numpy().tolist()]
                else:
                    xyxy, confs, clss = [], [], []
                dets = clamp_and_filter_boxes(
                    xyxy,
                    confs,
                    clss,
                    width=w,
                    height=h,
                    conf_thresh=conf_thresh,
                    max_det=100,
                    label_offset=0,
                )
                outputs.append({"image_id": img_path.stem, "detections": dets})
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = build_faster_rcnn(
            num_defect_classes=len(CLASS_ORDER),
            pretrained=False,
            min_size=int(ckpt.get("min_size", 600)),
            max_size=int(ckpt.get("max_size", 800)),
            box_score_thresh=conf_thresh,
            box_nms_thresh=float(nms_iou) if nms_iou is not None else 0.50,
            box_detections_per_img=100,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        with torch.no_grad():
            for img_path in image_paths:
                pil_img = Image.open(img_path).convert("RGB")
                w, h = pil_img.size
                xt = TF.to_tensor(pil_img).to(device)
                res = model([xt])[0]
                dets = clamp_and_filter_boxes(
                    res["boxes"].cpu().numpy().tolist(),
                    res["scores"].cpu().numpy().tolist(),
                    res["labels"].cpu().numpy().tolist(),
                    width=w,
                    height=h,
                    conf_thresh=conf_thresh,
                    max_det=100,
                    label_offset=1,
                )
                outputs.append({"image_id": img_path.stem, "detections": dets})

    out_path = Path(args.output_json)
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)


if __name__ == "__main__":
    main()

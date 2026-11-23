"""Stage 1 Benchmark Inference Script (Offline, Single Checkpoint, Deterministic)."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import CLASS_ORDER, SteelDefectCNN

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
    parser = argparse.ArgumentParser(description="Stage 1 Steel Defect Classification Inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of test images")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    config_path = BASE_DIR / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    ckpt_path = BASE_DIR / cfg.get("checkpoint_file", "stage1_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    model_kwargs = ckpt["model_kwargs"]
    class_order = ckpt.get("class_order", cfg.get("class_order", CLASS_ORDER))
    input_size = ckpt.get("input_size", [200, 200])
    norm_mean = ckpt.get("normalization_mean", cfg["normalization_mean"])
    norm_std = ckpt.get("normalization_std", cfg["normalization_std"])

    model = SteelDefectCNN(**model_kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((int(input_size[0]), int(input_size[1]))),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )

    image_paths = list_images(Path(args.input_dir))
    rows = []

    with torch.no_grad():
        batch_tensors = []
        batch_stems = []
        batch_size = 64

        for img_path in image_paths:
            img = Image.open(img_path).convert("L")
            tensor = tf(img)
            batch_tensors.append(tensor)
            batch_stems.append(img_path.stem)

            if len(batch_tensors) == batch_size:
                xb = torch.stack(batch_tensors, dim=0).to(device)
                preds = torch.argmax(model(xb), dim=1).cpu().tolist()
                for stem, pred_idx in zip(batch_stems, preds):
                    rows.append((stem, class_order[pred_idx]))
                batch_tensors.clear()
                batch_stems.clear()

        if batch_tensors:
            xb = torch.stack(batch_tensors, dim=0).to(device)
            preds = torch.argmax(model(xb), dim=1).cpu().tolist()
            for stem, pred_idx in zip(batch_stems, preds):
                rows.append((stem, class_order[pred_idx]))

    out_path = Path(args.output_csv)
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "prediction"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()

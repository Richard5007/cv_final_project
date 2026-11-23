"""CPU-only generalization diagnostic on already-trained Stage 1 E1-E5 checkpoints."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Team21_Stage1" / "src"))

from model import CLASS_ORDER, SteelDefectCNN


def main() -> None:
    cnn_path = ROOT / "workspace_outputs/stage1/cnn_ablation_results.json"
    cnn = json.loads(cnn_path.read_text(encoding="utf-8"))
    exps = cnn["experiments"]

    val_dir = ROOT / "data/STUDENT_PUBLIC/classification/validation"
    samples = []
    for idx, cname in enumerate(CLASS_ORDER):
        for p in sorted((val_dir / cname).glob("*.png")):
            samples.append((p, idx))

    # Pre-generate deterministic mildly shifted validation tensors (seed=42)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    shift_tf = transforms.Compose(
        [
            transforms.Resize((200, 200)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=cnn["normalization_mean"], std=cnn["normalization_std"]),
        ]
    )

    shifted_imgs = []
    y_true = []
    for p, lbl in samples:
        with Image.open(p) as im:
            g = im.convert("L")
            shifted_imgs.append(shift_tf(g))
            y_true.append(lbl)
    xb = torch.stack(shifted_imgs, dim=0)

    diag = {}
    device = torch.device("cpu")
    for eid in ["E1", "E2", "E3", "E4", "E5"]:
        e = exps[eid]
        h = e["history"]
        ckpt_path = ROOT / f"workspace_outputs/stage1/checkpoints/{eid}.pth"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = SteelDefectCNN(**ckpt["model_kwargs"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        preds = []
        with torch.no_grad():
            for i in range(0, len(xb), 64):
                logits = model(xb[i : i + 64])
                preds.extend(torch.argmax(logits, dim=1).tolist())

        pert_acc = float(accuracy_score(y_true, preds))
        pert_f1 = float(f1_score(y_true, preds, labels=list(range(6)), average="macro", zero_division=0))
        ep98 = next((ep for ep, f1 in zip(h["epoch"], h["val_macro_f1"]) if f1 >= 0.98), None)
        last10_f1 = float(sum(h["val_macro_f1"][-10:]) / min(10, len(h["val_macro_f1"])))
        last10_loss = float(sum(h["val_loss"][-10:]) / min(10, len(h["val_loss"])))

        diag[eid] = {
            "best_val_loss": float(e["best_val"]["loss"]),
            "ep_ge_098": ep98,
            "last10_val_f1": last10_f1,
            "last10_val_loss": last10_loss,
            "perturbed_val_acc": pert_acc,
            "perturbed_val_f1": pert_f1,
        }
        print(
            f"[Diag] {eid}: best_val_loss={diag[eid]['best_val_loss']:.5f}, "
            f"ep>=0.98={ep98}, last10_f1={last10_f1:.4f}, last10_loss={last10_loss:.4f}, "
            f"perturbed_f1={pert_f1:.4f}"
        )

    out_path = ROOT / "workspace_outputs/stage1/generalization_diag.json"
    out_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"[Diag] Saved {out_path}")


if __name__ == "__main__":
    main()

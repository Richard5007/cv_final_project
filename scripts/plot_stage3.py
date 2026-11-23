"""Generate publication-quality charts & detection visualizations for Stage 3."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_ORDER = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]
CLS_COLORS = {
    "crazing": "#ef4444",
    "inclusion": "#3b82f6",
    "patches": "#10b981",
    "pitted_surface": "#f59e0b",
    "rolled_in_scale": "#8b5cf6",
    "scratches": "#ec4899",
}


def draw_boxes_on_ax(ax, img_pil, boxes, is_gt: bool, title: str):
    ax.imshow(img_pil)
    for b in boxes:
        cls_name = b["class"]
        x1, y1, x2, y2 = b["bbox"]
        col = CLS_COLORS.get(cls_name, "#ffffff")
        ls = "-" if is_gt else "--"
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8, edgecolor=col, facecolor="none", linestyle=ls)
        ax.add_patch(rect)
        lbl = cls_name if is_gt else f"{cls_name} {b['confidence']:.2f}"
        ax.text(
            x1,
            max(y1 - 3, 8),
            lbl,
            fontsize=6.5,
            color="white",
            bbox=dict(boxstyle="square,pad=0.15", fc=col, ec="none", alpha=0.82),
        )
    ax.set_title(title, fontsize=8.5, fontweight="bold")
    ax.axis("off")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "workspace_outputs" / "stage3"
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root / "Team21_Stage3" / "src"))
    from evaluate import parse_voc_annotations

    data = json.loads((out_dir / "stage3_results.json").read_text(encoding="utf-8"))
    preds = json.loads((out_dir / "val_predictions.json").read_text(encoding="utf-8"))
    pred_map = {p["image_id"]: p["detections"] for p in preds}
    val_img_dir = root / "data/STUDENT_PUBLIC/detection/validation/images"
    gt_by_image = parse_voc_annotations(root / "data/STUDENT_PUBLIC/detection/validation/annotations_voc")

    # 1. Per-class AP@0.5 across detectors (eval_conf001 for full Pascal VOC 2010+ PR curve)
    det_ids = ["yolo11s", "yolo11m", "faster_rcnn"]
    det_labels = {"yolo11s": "YOLO11s", "yolo11m": "YOLO11m", "faster_rcnn": "Faster R-CNN R50-FPN-v2"}
    det_cols = {"yolo11s": "#0284c7", "yolo11m": "#7c3aed", "faster_rcnn": "#059669"}

    fig, ax = plt.subplots(figsize=(11.8, 5.2), dpi=200)
    xc = np.arange(len(CLASS_ORDER) + 1)
    w = 0.25
    for idx, did in enumerate(det_ids):
        ev = data["detectors"][did]["eval_conf001"]
        vals = [ev["per_class"][c]["ap50"] for c in CLASS_ORDER] + [ev["mAP50"]]
        bars = ax.bar(xc + (idx - 1) * w, vals, width=w, label=f"{det_labels[did]} (mAP={ev['mAP50']:.4f})", color=det_cols[did], zorder=3)
        for b in bars:
            v = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.012,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                rotation=90,
                zorder=4,
                bbox=dict(boxstyle="square,pad=0.04", fc="white", ec="none", alpha=0.75),
            )
    ax.axhline(0.90, color="#dc2626", linestyle="--", linewidth=1.2, zorder=1, label="Stage 3 Full-Mark Target (0.90)")
    ax.set_xticks(xc)
    ax.set_xticklabels(CLASS_ORDER + ["Overall mAP@0.5"], fontsize=9.5)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("AP@0.5 / mAP@0.5 (Slide-31 VOC 2010+)", fontsize=10)
    ax.set_title("Stage 3: Detector Per-Class AP@0.5 & Overall mAP@0.5 Comparison (Validation, N=270, conf=0.01)", fontsize=11.8, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_stage3_detector_comparison.png", dpi=200)
    plt.close(fig)

    # 2. Confidence Threshold Trade-off & Per-class PR Curves (1x2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=200)
    sweep = data["conf_sweep"]
    cs = [s["conf_threshold"] for s in sweep]
    axes[0].plot(cs, [s["mAP50"] for s in sweep], "o-", label="mAP@0.5", color="#1d4ed8", linewidth=2)
    axes[0].plot(cs, [s["precision"] for s in sweep], "s--", label="Precision", color="#059669", linewidth=1.8)
    axes[0].plot(cs, [s["recall"] for s in sweep], "^--", label="Recall", color="#d97706", linewidth=1.8)
    axes[0].plot(cs, [s["f1"] for s in sweep], "d-", label="F1 Score", color="#7c3aed", linewidth=1.8)
    axes[0].axvline(data["confidence_threshold"], color="#dc2626", linestyle=":", linewidth=1.6, label=f"Selected Conf={data['confidence_threshold']}")
    axes[0].set_title(f"(a) Confidence Threshold Sensitivity ({data['selected_detector']})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Confidence Threshold", fontsize=10)
    axes[0].set_ylabel("Score", fontsize=10)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8.5)

    final_pc = data["final_val_metrics"]["per_class"]
    for c in CLASS_ORDER:
        pr = final_pc[c]["pr_curve"]
        if pr["recall"]:
            axes[1].plot(pr["recall"], pr["precision"], label=f"{c} (AP={final_pc[c]['ap50']:.3f})", color=CLS_COLORS[c], linewidth=1.8)
    axes[1].set_title(f"(b) Per-Class Precision-Recall Curves (mAP@0.5={data['final_val_metrics']['mAP50']:.4f})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].set_xlim(0.0, 1.02)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8.0, loc="lower left")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_stage3_pr_and_threshold.png", dpi=200)
    plt.close(fig)

    # 3. Training Curves (YOLO11s/m + Faster R-CNN)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=200)
    for did in ["yolo11s", "yolo11m"]:
        h = data["detectors"][did]["history"]
        if h["epoch"]:
            axes[0].plot(h["epoch"], h["ultralytics_map50"], label=det_labels[did], color=det_cols[did], linewidth=2.0)
    axes[0].set_title("YOLO11s vs. YOLO11m Training Val mAP@0.5", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=9.5)
    axes[0].set_ylabel("mAP@0.5", fontsize=9.5)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=9)

    fh = data["detectors"]["faster_rcnn"]["history"]
    ax_l = axes[1]
    ax_r = ax_l.twinx()
    l1 = ax_l.plot(fh["epoch"], fh["train_loss"], "o-", color="#dc2626", label="Train Loss", linewidth=1.8)
    l2 = ax_r.plot(fh["epoch"], fh["val_map50"], "s-", color="#0284c7", label="Val mAP@0.5", linewidth=1.8)
    ax_l.set_title("Faster R-CNN ResNet50-FPN-v2 Training Progress", fontsize=11, fontweight="bold")
    ax_l.set_xlabel("Epoch", fontsize=9.5)
    ax_l.set_ylabel("Total Loss", color="#dc2626", fontsize=9.5)
    ax_r.set_ylabel("Val mAP@0.5", color="#0284c7", fontsize=9.5)
    ax_l.grid(True, linestyle="--", alpha=0.35)
    lines = l1 + l2
    ax_l.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_stage3_training_curves.png", dpi=200)
    plt.close(fig)

    # 4. 12 Validation Detection Visualizations (display filter conf >= 0.25 for crisp readability)
    display_conf = max(0.25, float(data.get("confidence_threshold", 0.01)))
    selected_ids = []
    for c in CLASS_ORDER:
        c_imgs = [img_id for img_id, info in sorted(gt_by_image.items()) if any(b["class"] == c for b in info["boxes"])]
        multi = [img_id for img_id in c_imgs if len({b["class"] for b in gt_by_image[img_id]["boxes"]}) > 1]
        pick1 = multi[0] if multi else c_imgs[0]
        pick2 = next((x for x in c_imgs[len(c_imgs) // 2 :] if x not in selected_ids and x != pick1), c_imgs[-1])
        for p in (pick1, pick2):
            if p not in selected_ids:
                selected_ids.append(p)
    # Fill to exactly 12 unique validation images if multi-class overlap caused fewer than 12
    for img_id in sorted(gt_by_image.keys()):
        if len(selected_ids) >= 12:
            break
        if img_id not in selected_ids:
            selected_ids.append(img_id)
    selected_ids = selected_ids[:12]

    for part_idx, chunk in enumerate([selected_ids[:6], selected_ids[6:12]], start=1):
        fig, axes = plt.subplots(3, 4, figsize=(13.5, 9.8), dpi=200)
        for row_i in range(3):
            for col_pair in range(2):
                sample_pos = row_i * 2 + col_pair
                ax_gt = axes[row_i, col_pair * 2]
                ax_pr = axes[row_i, col_pair * 2 + 1]
                if sample_pos >= len(chunk):
                    ax_gt.axis("off")
                    ax_pr.axis("off")
                    continue
                img_id = chunk[sample_pos]
                im = Image.open(val_img_dir / f"{img_id}.png").convert("RGB")
                gts = gt_by_image[img_id]["boxes"]
                prs = [d for d in pred_map.get(img_id, []) if d["confidence"] >= display_conf]
                draw_boxes_on_ax(ax_gt, im, gts, is_gt=True, title=f"{img_id} — Ground Truth ({len(gts)})")
                draw_boxes_on_ax(ax_pr, im, prs, is_gt=False, title=f"{img_id} — Pred conf>={display_conf:.2f} ({len(prs)})")
        fig.suptitle(
            f"Stage 3 Validation Visualization Part {part_idx} (Solid=GT, Dashed=Pred @ Display>={display_conf:.2f}; JSON Output Uses >={data['confidence_threshold']})",
            fontsize=11.5,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fig_dir / f"fig4_stage3_vis_part{part_idx}.png", dpi=200)
        plt.close(fig)

    # 5. 6 Failure Analysis Cases in crisp 3x4 grid (Row 1=FP, Row 2=FN, Row 3=Poor Localization)
    fails = data["failures"]
    picked_fails = []
    for k in ["false_positives", "false_negatives", "poor_localization"]:
        seen_imgs = set()
        cnt = 0
        for case in fails[k]:
            if case["image_id"] not in seen_imgs:
                picked_fails.append(case)
                seen_imgs.add(case["image_id"])
                cnt += 1
            if cnt == 2:
                break

    if picked_fails:
        n_pairs = len(picked_fails)
        rows = (n_pairs + 1) // 2
        fig, axes = plt.subplots(rows, 4, figsize=(13.5, 3.3 * rows), dpi=200)
        if rows == 1:
            axes = np.array([axes])
        for p_idx, case in enumerate(picked_fails):
            r_i = p_idx // 2
            c_pair = p_idx % 2
            ax_gt = axes[r_i, c_pair * 2]
            ax_pr = axes[r_i, c_pair * 2 + 1]
            img_id = case["image_id"]
            im = Image.open(val_img_dir / f"{img_id}.png").convert("RGB")
            gts = gt_by_image[img_id]["boxes"]
            prs = [d for d in pred_map.get(img_id, []) if d["confidence"] >= display_conf]
            ctype = case["type"].upper()
            if case["type"] == "false_positive":
                detail = f"FP {case['pred_class']} ({case['confidence']:.2f}), maxIoU={case['max_iou_any_gt']:.2f}"
            elif case["type"] == "false_negative":
                detail = f"FN {case['gt_class']} ({case['gt_area']:.0f}px2)"
            else:
                detail = f"PoorLoc {case['class']} ({case['confidence']:.2f}), IoU={case['iou']:.3f}"
            draw_boxes_on_ax(ax_gt, im, gts, is_gt=True, title=f"[{ctype}] {img_id} GT")
            draw_boxes_on_ax(ax_pr, im, prs, is_gt=False, title=f"{img_id} | {detail}")
        if n_pairs % 2 == 1:
            axes[rows - 1, 2].axis("off")
            axes[rows - 1, 3].axis("off")
        fig.suptitle(f"Stage 3 Failure Analysis Cases ({len(picked_fails)} Representative Cases @ Operating Conf>={display_conf:.2f})", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fig_dir / "fig5_stage3_failure_cases.png", dpi=200)
        plt.close(fig)

    print(f"[Plot Stage 3] Saved figures to {fig_dir}, picked {len(selected_ids)} vis & {len(picked_fails)} failures")


if __name__ == "__main__":
    main()

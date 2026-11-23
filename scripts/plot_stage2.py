"""Generate publication-quality charts for Stage 2 from real experiment JSONs."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
SHORT_CLASSES = ["crazing", "inclusion", "patches", "pitted", "rolled", "scratches"]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "workspace_outputs" / "stage2"
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads((out_dir / "stage2_results.json").read_text(encoding="utf-8"))
    exps = data["experiments"]
    exp_ids = ["E1", "E2", "E3", "E4", "E5", "E6"]

    strat_colors = {
        "feature_extraction": "#64748b",
        "partial_finetuning": "#0284c7",
        "full_finetuning": "#059669",
    }
    strat_names = {
        "feature_extraction": "Feature Extraction",
        "partial_finetuning": "Partial Fine-tuning",
        "full_finetuning": "Full Fine-tuning",
    }

    # 1. Training / Validation Curves per Architecture (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), dpi=200)
    for eid in ["E1", "E2", "E3"]:
        e = exps[eid]
        h = e["history"]
        c = strat_colors[e["strategy"]]
        lbl = strat_names[e["strategy"]]
        axes[0, 0].plot(h["epoch"], h["train_loss"], linestyle="--", color=c, alpha=0.65, linewidth=1.3)
        axes[0, 0].plot(h["epoch"], h["val_loss"], color=c, linewidth=1.9, label=f"{lbl} (Val)")
        axes[0, 1].plot(h["epoch"], h["train_macro_f1"], linestyle="--", color=c, alpha=0.65, linewidth=1.3)
        axes[0, 1].plot(h["epoch"], h["val_macro_f1"], color=c, linewidth=1.9, label=f"{lbl} (Val)")

    for eid in ["E4", "E5", "E6"]:
        e = exps[eid]
        h = e["history"]
        c = strat_colors[e["strategy"]]
        lbl = strat_names[e["strategy"]]
        axes[1, 0].plot(h["epoch"], h["train_loss"], linestyle="--", color=c, alpha=0.65, linewidth=1.3)
        axes[1, 0].plot(h["epoch"], h["val_loss"], color=c, linewidth=1.9, label=f"{lbl} (Val)")
        axes[1, 1].plot(h["epoch"], h["train_macro_f1"], linestyle="--", color=c, alpha=0.65, linewidth=1.3)
        axes[1, 1].plot(h["epoch"], h["val_macro_f1"], color=c, linewidth=1.9, label=f"{lbl} (Val)")

    panel_titles = [
        "DenseNet121 — Loss (Dashed=Train, Solid=Val)",
        "DenseNet121 — Macro F1 (Dashed=Train, Solid=Val)",
        "MobileNetV3-Large — Loss (Dashed=Train, Solid=Val)",
        "MobileNetV3-Large — Macro F1 (Dashed=Train, Solid=Val)",
    ]
    for idx, (ax, t) in enumerate(zip(axes.flat, panel_titles)):
        ax.set_title(t, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9.5)
        ax.set_ylabel("Loss" if idx % 2 == 0 else "Macro F1", fontsize=9.5)
        if idx % 2 == 1:
            ax.set_ylim(0.4, 1.03)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8.2, loc="upper right" if idx % 2 == 0 else "lower right")
    fig.suptitle("Stage 2: 2x3 Transfer Learning Convergence Curves (Seed=42)", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / "fig1_stage2_curves.png", dpi=200)
    plt.close(fig)

    # 2. Clean vs Local Robustness Grouped Bar Chart
    labels = [
        f"{exps[eid]['architecture'].replace('mobilenet_v3_large', 'MobileNetV3-L').replace('densenet121', 'DenseNet121')}\n{strat_names[exps[eid]['strategy']]}"
        for eid in exp_ids
    ]
    clean_vals = [exps[eid]["clean_val"]["macro_f1"] for eid in exp_ids]
    light_vals = [exps[eid]["robustness"]["lighting"]["macro_f1"] for eid in exp_ids]
    opt_vals = [exps[eid]["robustness"]["optical"]["macro_f1"] for eid in exp_ids]
    sens_vals = [exps[eid]["robustness"]["sensor_compression"]["macro_f1"] for eid in exp_ids]
    rob_avg = [exps[eid]["robustness"]["robust_macro_f1"] for eid in exp_ids]

    x = np.arange(len(exp_ids))
    w = 0.16
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=200)
    bars_cfg = [
        ("Clean Val", clean_vals, "#1d4ed8", -2),
        ("Lighting", light_vals, "#f59e0b", -1),
        ("Optical", opt_vals, "#06b6d4", 0),
        ("Sensor/JPEG", sens_vals, "#8b5cf6", 1),
        ("Robust Avg", rob_avg, "#059669", 2),
    ]
    ax.axhline(0.95, color="#1d4ed8", linestyle=":", linewidth=1.2, zorder=1, label="Standard Full-Mark Target (0.95)")
    ax.axhline(0.85, color="#059669", linestyle="--", linewidth=1.2, zorder=1, label="Robust Full-Mark Target (0.85)")
    for lbl, vals, col, off in bars_cfg:
        bs = ax.bar(x + off * w, vals, width=w, label=lbl, color=col, zorder=3)
        for b in bs:
            v = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.012,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=90,
                zorder=4,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.75),
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.16)
    ax.set_ylabel("Macro F1 Score", fontsize=10.5)
    ax.set_title("Stage 2: Standard vs. Local Robustness Stress-Test Macro F1 (2x3 Controlled Comparison)", fontsize=12.5, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=4, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_stage2_robustness_bar.png", dpi=200)
    plt.close(fig)

    # 3. Deployment Trade-off Scatter (Total Params M vs Robust Macro F1, bubble ~ checkpoint MB)
    fig, ax = plt.subplots(figsize=(10.2, 5.4), dpi=200)
    offsets = {
        "E1": (-150, 8),
        "E2": (-160, -28),
        "E3": (-160, 10),
        "E4": (14, -8),
        "E5": (14, -14),
        "E6": (14, -8),
    }
    for eid in exp_ids:
        e = exps[eid]
        params_m = e["layer_summary"]["total_params"] / 1e6
        rf1 = e["robustness"]["robust_macro_f1"]
        cf1 = e["clean_val"]["macro_f1"]
        mb = e["checkpoint_size_mb"]
        marker = "o" if e["architecture"] == "densenet121" else "^"
        col = strat_colors[e["strategy"]]
        ax.scatter(params_m, rf1, s=mb * 18 + 80, color=col, marker=marker, alpha=0.88, edgecolors="#0f172a", linewidths=1.2, zorder=4)
        ox, oy = offsets[eid]
        ax.annotate(
            f"{eid} ({e['architecture'].split('_')[0]}, {e['strategy'].split('_')[0]})\nClean={cf1:.3f}, Robust={rf1:.3f}\nParams={params_m:.2f}M, Size={mb:.1f}MB",
            (params_m, rf1),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=8.0,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.9, alpha=0.8),
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=col, alpha=0.92),
            zorder=5,
        )
    ax.set_xlim(3.5, 7.6)
    ax.set_ylim(0.875, 1.006)
    ax.set_xlabel("Total Parameters (Millions)", fontsize=10.5)
    ax.set_ylabel("Local Robust Macro F1", fontsize=10.5)
    ax.set_title("Stage 2: Model Complexity vs. Robustness Trade-off (Bubble Size = Checkpoint MB)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_stage2_deployment_tradeoff.png", dpi=200)
    plt.close(fig)

    # 4. Final Model Confusion Matrices (Clean Val + Aggregated Robustness)
    sel_id = data["selected_model_id"]
    sel = exps[sel_id]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), dpi=200)
    specs = [
        (f"Final Model ({sel_id}: {sel['architecture']} / {sel['strategy']})\nClean Validation (N=270)", np.array(sel["clean_val"]["confusion_matrix"]), 45),
        (f"Final Model ({sel_id})\nAggregated Local Robustness (3x270=810)", np.array(sel["robustness"]["aggregated_confusion_matrix"]), 135),
    ]
    for ax, (title, cm, vmax) in zip(axes, specs):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(SHORT_CLASSES, rotation=28, ha="right", fontsize=8.5)
        ax.set_yticklabels(SHORT_CLASSES, fontsize=8.5)
        ax.set_xlabel("Predicted", fontsize=9.5)
        ax.set_ylabel("True", fontsize=9.5)
        for i in range(6):
            for j in range(6):
                v = int(cm[i, j])
                ax.text(j, i, str(v), ha="center", va="center", color="white" if v > vmax * 0.5 else "#0f172a", fontsize=9, fontweight="bold" if i == j else "normal")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_stage2_final_cm.png", dpi=200)
    plt.close(fig)

    # 5. Visual Strip of Clean vs 3 Synthetic Robustness Corruptions
    sys.path.insert(0, str(root / "Team21_Stage2" / "src"))
    from evaluate import apply_lighting_corruption, apply_optical_corruption, apply_sensor_compression_corruption
    sample_img_path = root / "data/STUDENT_PUBLIC/classification/validation/crazing/VAL_0001.png"
    sample_pil = Image.open(sample_img_path).convert("RGB")
    rng = np.random.default_rng(42)
    imgs_strip = [
        ("Clean Original\n(VAL_0001: crazing)", sample_pil),
        ("Lighting Shift\n(Gamma+Contrast+Shading)", apply_lighting_corruption(sample_pil, rng)),
        ("Optical Blur\n(Defocus + Motion Blur)", apply_optical_corruption(sample_pil, rng)),
        ("Sensor + JPEG\n(Noise + Quality 15-35)", apply_sensor_compression_corruption(sample_pil, rng)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), dpi=200)
    for ax, (t, im) in zip(axes, imgs_strip):
        ax.imshow(im)
        ax.set_title(t, fontsize=9.5, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Stage 2 Local Robustness Stress-Test Examples (Applied to Validation Only)", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(fig_dir / "fig5_stage2_corruption_strip.png", dpi=200)
    plt.close(fig)

    print(f"[Plot Stage 2] Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()

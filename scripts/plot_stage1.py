"""Generate publication-quality charts for Stage 1 from real experiment JSONs."""

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
    out_dir = root / "workspace_outputs" / "stage1"
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    trad = json.loads((out_dir / "traditional_results.json").read_text(encoding="utf-8"))
    cnn = json.loads((out_dir / "cnn_ablation_results.json").read_text(encoding="utf-8"))
    exps = cnn["experiments"]
    exp_ids = ["E1", "E2", "E3", "E4", "E5"]
    colors = {
        "E1": "#64748b",
        "E2": "#0284c7",
        "E3": "#d97706",
        "E4": "#059669",
        "E5": "#7c3aed",
    }
    short_labels = {
        "E1": "E1: Baseline",
        "E2": "E2: +BN (A)",
        "E3": "E3: +Aug (B)",
        "E4": "E4: +BN+Aug (A+B)",
        "E5": "E5: +A+B+Reg (C)",
    }

    # 1. Ablation Curves (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), dpi=200)
    for eid in exp_ids:
        h = exps[eid]["history"]
        ep = h["epoch"]
        c = colors[eid]
        lbl = short_labels[eid]
        axes[0, 0].plot(ep, h["train_loss"], label=lbl, color=c, linewidth=1.8)
        axes[0, 1].plot(ep, h["val_loss"], label=lbl, color=c, linewidth=1.8)
        axes[1, 0].plot(ep, h["train_macro_f1"], label=lbl, color=c, linewidth=1.8)
        axes[1, 1].plot(ep, h["val_macro_f1"], label=lbl, color=c, linewidth=1.8)

    titles = [
        "(a) Training Loss",
        "(b) Validation Loss",
        "(c) Training Macro F1",
        "(d) Validation Macro F1",
    ]
    ylabels = ["Cross-Entropy Loss", "Cross-Entropy Loss", "Macro F1", "Macro F1"]
    for ax, title, yl in zip(axes.flat, titles, ylabels):
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1, 0].set_ylim(0.2, 1.03)
    axes[1, 1].set_ylim(0.2, 1.03)
    axes[1, 1].axhline(0.90, color="#dc2626", linestyle=":", linewidth=1.3, label="Benchmark Full Score (0.90)")
    axes[0, 0].legend(fontsize=8.5, loc="upper right")
    axes[1, 1].legend(fontsize=8.0, loc="lower right")
    fig.suptitle("Stage 1: Controlled CNN Ablation Curves (E1–E5, Seed=42)", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / "fig1_stage1_ablation_curves.png", dpi=200)
    plt.close(fig)

    # 2. Overall Comparison Bar Chart (Traditional + CNN)
    names = ["PCA+KNN", "PCA+SVM"] + [short_labels[eid] for eid in exp_ids]
    f1s = [
        trad["models"]["PCA_KNN"]["macro_f1"],
        trad["models"]["PCA_SVM"]["macro_f1"],
    ] + [exps[eid]["best_val"]["macro_f1"] for eid in exp_ids]
    accs = [
        trad["models"]["PCA_KNN"]["accuracy"],
        trad["models"]["PCA_SVM"]["accuracy"],
    ] + [exps[eid]["best_val"]["accuracy"] for eid in exp_ids]

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=200)
    b1 = ax.bar(x - w / 2, f1s, width=w, label="Validation Macro F1", color="#1d4ed8", edgecolor="white")
    b2 = ax.bar(x + w / 2, accs, width=w, label="Validation Accuracy", color="#60a5fa", edgecolor="white")
    ax.axhline(0.90, color="#dc2626", linestyle="--", linewidth=1.4, label="Stage 1 Benchmark Full-Mark Target (0.90)")
    for rect in b1:
        val = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e3a8a")
    for rect in b2:
        val = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=8.0, color="#1d4ed8")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9.5)
    ax.set_ylim(0.0, 1.09)
    ax.set_ylabel("Score", fontsize=10.5)
    ax.set_title("Stage 1: Traditional Baselines vs. Self-Designed CNN Ablation (Validation Set, N=270)", fontsize=12.5, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_stage1_comparison_bar.png", dpi=200)
    plt.close(fig)

    # 3. Confusion Matrices (PCA+SVM, E1 Baseline, Selected Final CNN)
    sel_id = cnn["selected_model_id"]
    cm_specs = [
        ("Traditional Baseline: PCA + SVM", np.array(trad["models"]["PCA_SVM"]["confusion_matrix"])),
        ("E1: Baseline CNN (From Scratch)", np.array(exps["E1"]["best_val"]["confusion_matrix"])),
        (f"Final Selected: {sel_id}", np.array(exps[sel_id]["best_val"]["confusion_matrix"])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), dpi=200)
    for ax, (title, cm) in zip(axes, cm_specs):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=45)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(SHORT_CLASSES, rotation=30, ha="right", fontsize=8.5)
        ax.set_yticklabels(SHORT_CLASSES, fontsize=8.5)
        ax.set_xlabel("Predicted", fontsize=9.5)
        ax.set_ylabel("True", fontsize=9.5)
        for i in range(6):
            for j in range(6):
                val = int(cm[i, j])
                ax.text(j, i, str(val), ha="center", va="center", color="white" if val > 22 else "#0f172a", fontsize=9, fontweight="bold" if i == j else "normal")
    fig.suptitle("Stage 1 Validation Confusion Matrices Comparison (45 samples / class)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_dir / "fig3_stage1_confusion_matrices.png", dpi=200)
    plt.close(fig)

    # 4. Per-class F1 breakdown
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=200)
    xc = np.arange(6)
    w = 0.25
    series = [
        ("PCA + SVM", [trad["models"]["PCA_SVM"]["per_class"][c]["f1"] for c in CLASS_ORDER], "#94a3b8"),
        ("E1: Baseline CNN", [exps["E1"]["best_val"]["per_class"][c]["f1"] for c in CLASS_ORDER], "#3b82f6"),
        (f"Final ({sel_id})", [exps[sel_id]["best_val"]["per_class"][c]["f1"] for c in CLASS_ORDER], "#059669"),
    ]
    for k, (lbl, vals, col) in enumerate(series):
        bars = ax.bar(xc + (k - 1) * w, vals, width=w, label=lbl, color=col)
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xc)
    ax.set_xticklabels(CLASS_ORDER, fontsize=9.5)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("Per-Class F1 Score", fontsize=10)
    ax.set_title("Stage 1: Per-Class F1 Comparison Across Defect Categories", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_stage1_per_class_f1.png", dpi=200)
    plt.close(fig)

    # 5. Calibration Loss & Mild-Shift Generalization Diagnostic (2-panel)
    diag_path = out_dir / "generalization_diag.json"
    if diag_path.is_file():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=200)
        xe = np.arange(len(exp_ids))
        bw = 0.34

        best_losses = [diag[eid]["best_val_loss"] for eid in exp_ids]
        last10_losses = [diag[eid]["last10_val_loss"] for eid in exp_ids]
        b1 = axes[0].bar(xe - bw / 2, best_losses, width=bw, label="Best Checkpoint Val Loss", color="#0284c7")
        b2 = axes[0].bar(xe + bw / 2, last10_losses, width=bw, label="Last-10-Epoch Mean Val Loss", color="#94a3b8")
        for b in list(b1) + list(b2):
            v = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.0008, f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)
        axes[0].set_xticks(xe)
        axes[0].set_xticklabels([short_labels[eid] for eid in exp_ids], rotation=15, ha="right", fontsize=8.8)
        axes[0].set_ylabel("Cross-Entropy Loss", fontsize=10)
        axes[0].set_title("(a) Validation Calibration & Late-Epoch Loss Stability", fontsize=11, fontweight="bold")
        axes[0].set_ylim(0, max(last10_losses) * 1.25)
        axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)
        axes[0].legend(fontsize=8.5)

        clean_f1s = [exps[eid]["best_val"]["macro_f1"] for eid in exp_ids]
        pert_f1s = [diag[eid]["perturbed_val_f1"] for eid in exp_ids]
        b3 = axes[1].bar(xe - bw / 2, clean_f1s, width=bw, label="Clean Val Macro F1", color="#1d4ed8")
        b4 = axes[1].bar(xe + bw / 2, pert_f1s, width=bw, label="Mild-Shift Probe F1 (±15° / ±20% BC)", color="#059669")
        for b in list(b3) + list(b4):
            v = b.get_height()
            axes[1].text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        axes[1].set_xticks(xe)
        axes[1].set_xticklabels([short_labels[eid] for eid in exp_ids], rotation=15, ha="right", fontsize=8.8)
        axes[1].set_ylabel("Macro F1 Score", fontsize=10)
        axes[1].set_title("(b) Clean vs. Mild-Shift Generalization Probe (Seed=42)", fontsize=11, fontweight="bold")
        axes[1].set_ylim(0, 1.12)
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
        axes[1].legend(fontsize=8.5, loc="lower right")

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle("Stage 1: Overfitting, Calibration Loss & Generalization Diagnostic (E1–E5)", fontsize=12.5, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(fig_dir / "fig5_stage1_generalization.png", dpi=200)
        plt.close(fig)

    # 6. Misclassified Error Gallery fallback if needed
    errors = exps[sel_id]["best_val"].get("errors", [])
    fallback_errors = exps["E1"]["best_val"].get("errors", [])
    display_errs = errors if len(errors) > 0 else fallback_errors
    title_prefix = f"Stage 1 ({sel_id}) Misclassified Samples" if len(errors) > 0 else f"Stage 1 E1 Baseline Errors (Resolved by {sel_id}, 0 Final Val Errors)"
    n_show = min(8, len(display_errs))
    if n_show > 0:
        cols = min(4, n_show)
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.3 * rows), dpi=200)
        axes_arr = np.atleast_1d(axes).ravel()
        for ax in axes_arr:
            ax.axis("off")
        for idx in range(n_show):
            err = display_errs[idx]
            img_path = Path(err["path"])
            if not img_path.is_absolute():
                img_path = root / img_path
            im = Image.open(img_path).convert("L")
            ax = axes_arr[idx]
            ax.imshow(im, cmap="gray")
            ax.set_title(
                f"{img_path.stem}\nTrue: {err['true_label']}\nPred: {err['pred_label']} ({err['pred_confidence']:.2f})",
                fontsize=8.5,
                color="#b91c1c",
            )
        fig.suptitle(f"{title_prefix} (Showing {n_show}/{len(display_errs)})", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(fig_dir / "fig6_stage1_error_gallery.png", dpi=200)
        plt.close(fig)

    print(f"[Plot Stage 1] Saved figures to {fig_dir}, total errors={len(errors)}")


if __name__ == "__main__":
    main()

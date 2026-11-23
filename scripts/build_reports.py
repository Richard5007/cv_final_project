"""Build all 4 FactoryVision PDF reports dynamically from real JSON metrics and figures."""

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
CHROME_PATHS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]

CLASS_ORDER = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


def find_browser() -> Path:
    for p in CHROME_PATHS:
        if p.is_file():
            return p
    raise FileNotFoundError("Neither Google Chrome nor Microsoft Edge was found for HTML-to-PDF rendering.")


def render_html_to_pdf(html_text: str, html_out: Path, pdf_out: Path) -> None:
    html_out.parent.mkdir(parents=True, exist_ok=True)
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(html_text, encoding="utf-8")
    browser = find_browser()
    cmd = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--allow-file-access-from-files",
        f"--print-to-pdf={pdf_out.resolve()}",
        "--no-pdf-header-footer",
        html_out.resolve().as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not pdf_out.is_file() or pdf_out.stat().st_size < 1000:
        raise RuntimeError(f"PDF render failed for {pdf_out}: rc={proc.returncode}, stderr={proc.stderr}")


def fmt_pct(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "待 TA 公布"
    return f"{v * 100:.{digits}f}%"


def fmt_f(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "待 TA 公布"
    return f"{v:.{digits}f}"


def img_tag(path: Path, caption: str, max_height_px: int = 430) -> str:
    if not path.is_file():
        return f"<p class='warn'>[Missing figure: {html.escape(path.name)}]</p>"
    return (
        f"<figure>"
        f"<img src='{path.resolve().as_uri()}' style='max-height:{max_height_px}px;' />"
        f"<figcaption>{html.escape(caption)}</figcaption>"
        f"</figure>"
    )


COMMON_CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
}
* { box-sizing: border-box; }
body {
  font-family: "Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "Segoe UI", Arial, sans-serif;
  color: #1f2937;
  font-size: 9.8pt;
  line-height: 1.62;
  margin: 0;
  padding: 0;
}
.report-header {
  border-bottom: 2px solid #1e293b;
  padding-bottom: 10px;
  margin-bottom: 14px;
}
.report-header .course-tag {
  font-size: 8.5pt;
  color: #475569;
  margin-bottom: 4px;
}
.report-header h1 {
  margin: 0 0 6px 0;
  font-size: 16.5pt;
  font-weight: 700;
  color: #0f172a;
}
.report-header .author-line {
  font-size: 9pt;
  color: #334155;
  margin: 0;
}
.summary-strip {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 14px 0;
  font-size: 8.8pt;
}
.summary-strip td {
  border: 1px solid #cbd5e1;
  background: #f8fafc;
  padding: 6px 10px;
  text-align: left;
  width: 25%;
}
.summary-strip .lbl {
  display: block;
  font-size: 7.8pt;
  color: #64748b;
}
.summary-strip .val {
  display: block;
  font-size: 10.2pt;
  font-weight: 700;
  color: #0f172a;
  margin-top: 2px;
}
h2 {
  font-size: 12.2pt;
  color: #0f172a;
  border-bottom: 1.5px solid #94a3b8;
  padding-bottom: 3px;
  margin: 15px 0 8px 0;
  page-break-after: avoid;
}
h3 {
  font-size: 10.5pt;
  color: #1e293b;
  margin: 11px 0 5px 0;
  page-break-after: avoid;
}
p { margin: 5px 0 8px 0; text-align: justify; }
ul, ol { margin: 4px 0 8px 0; padding-left: 20px; }
li { margin-bottom: 4px; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 8px 0 12px 0;
  font-size: 8.5pt;
  page-break-inside: avoid;
}
table.allow-split {
  page-break-inside: auto;
}
thead { display: table-header-group; }
tr { page-break-inside: avoid; break-inside: avoid; }
th, td {
  border: 1px solid #cbd5e1;
  padding: 5px 6px;
  text-align: center;
  vertical-align: middle;
}
th {
  background: #f1f5f9;
  color: #0f172a;
  font-weight: 700;
}
td.left, th.left { text-align: left; }
tr:nth-child(even) td { background: #fafafa; }
tr.highlight td { background: #eef6ff !important; font-weight: 700; }
.note-box {
  background: #f8fafc;
  border-left: 3.5px solid #475569;
  padding: 8px 12px;
  margin: 8px 0 12px 0;
  page-break-inside: avoid;
}
figure {
  margin: 8px 0 12px 0;
  text-align: center;
  page-break-inside: avoid;
}
figure img {
  max-width: 100%;
  height: auto;
  border: 1px solid #e2e8f0;
}
figcaption {
  font-size: 8.4pt;
  color: #475569;
  margin-top: 4px;
}
code {
  font-family: Consolas, "Courier New", monospace;
  background: #f1f5f9;
  padding: 1px 4px;
  border-radius: 2px;
  font-size: 8.4pt;
}
.page-break { page-break-before: always; }
.badge-pending {
  display: inline-block;
  background: #fef3c7;
  color: #92400e;
  padding: 1px 5px;
  border-radius: 3px;
  font-weight: 700;
  font-size: 8.2pt;
}
"""


def wrap_doc(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8" />
<title>{html.escape(title)}</title>
<style>{COMMON_CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


def build_stage1_report(s1_trad: Dict, s1_cnn: Dict, bench: Dict) -> str:
    exps = s1_cnn["experiments"]
    sel_id = s1_cnn["selected_model_id"]
    sel = exps[sel_id]
    e1 = exps["E1"]
    svm = s1_trad["models"]["PCA_SVM"]
    knn = s1_trad["models"]["PCA_KNN"]
    fig_dir = ROOT / "workspace_outputs" / "stage1" / "figs"
    diag_path = ROOT / "workspace_outputs" / "stage1" / "generalization_diag.json"
    s1_diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.is_file() else {}

    s1_bench = bench["stage1"]["hidden_macro_f1"]
    s1_pts = bench["stage1"]["rubric_points_max10"]

    # Ablation rows
    ablation_rows = []
    for eid in ["E1", "E2", "E3", "E4", "E5"]:
        e = exps[eid]
        cfg = e["config"]
        bv = e["best_val"]
        tr = e["best_train_clean"]
        fin = e["final_epoch_diagnostic"]
        dg = s1_diag.get(eid, {})
        cls_attr = " class='highlight'" if eid == sel_id else ""
        bn_mark = "&#10003;" if cfg["use_bn"] else "&#10005;"
        aug_mark = "&#10003;" if cfg["use_aug"] else "&#10005;"
        reg_mark = "&#10003;" if cfg["dropout_p"] > 0 else "&#10005;"
        pert_str = f"{dg['perturbed_val_f1']:.4f}" if "perturbed_val_f1" in dg else "N/A"
        last10_str = f"{dg['last10_val_f1']:.4f}" if "last10_val_f1" in dg else "N/A"
        ablation_rows.append(
            f"<tr{cls_attr}>"
            f"<td class='left'><b>{eid}</b>: {html.escape(e['name'])}</td>"
            f"<td>{bn_mark}</td><td>{aug_mark}</td><td>{reg_mark}</td>"
            f"<td>{e['best_epoch']}/{e['epochs_ran']}</td>"
            f"<td>{fmt_f(bv['accuracy'])}</td>"
            f"<td><b>{fmt_f(bv['macro_f1'])}</b></td>"
            f"<td>{bv['loss']:.5f}</td>"
            f"<td>{last10_str}</td>"
            f"<td><b>{pert_str}</b></td>"
            f"<td>{fin['overfit_gap_f1']:+.4f}</td>"
            f"<td>{e['train_time_sec']:.1f}s</td>"
            f"</tr>"
        )

    # Per-class table for selected model vs E1 vs SVM
    pc_rows = []
    for c in CLASS_ORDER:
        p_svm = svm["per_class"][c]
        p_e1 = e1["best_val"]["per_class"][c]
        p_sel = sel["best_val"]["per_class"][c]
        pc_rows.append(
            f"<tr>"
            f"<td class='left'><code>{c}</code></td>"
            f"<td>{p_svm['f1']:.4f}</td>"
            f"<td>{p_e1['f1']:.4f}</td>"
            f"<td>{p_sel['precision']:.4f}</td>"
            f"<td>{p_sel['recall']:.4f}</td>"
            f"<td><b>{p_sel['f1']:.4f}</b></td>"
            f"<td>{p_sel['support']}</td>"
            f"</tr>"
        )

    # Error summary
    errors = sel["best_val"].get("errors", [])
    e1_errors = e1["best_val"].get("errors", [])
    if errors:
        err_lines = "".join(
            f"<li><code>{Path(er['path']).stem}</code>：真實類別 <code>{er['true_label']}</code> → 誤判為 <code>{er['pred_label']}</code>（預測信心 {er['pred_confidence']:.3f}，真實類別機率 {er['true_confidence']:.3f}）</li>"
            for er in errors[:8]
        )
    else:
        err_lines = (
            "<li>最終選定模型 <b>" + sel_id + "</b> 於驗證集全數 270 張影像達成 0 筆誤判（100% 準確率）。</li>"
            + (f"<li>Baseline 模型（E1）原有 {len(e1_errors)} 筆誤判，皆已由改善策略完全消除。</li>" if e1_errors else "")
        )
    err_fig_html = (
        img_tag(fig_dir / "fig5_stage1_generalization.png", "圖 5：E1–E5 校準損失（Val Loss）與輕微偏移泛化診斷比較（Clean vs. Mild-Shift Probe）。", max_height_px=300)
        if (fig_dir / "fig5_stage1_generalization.png").is_file()
        else "<p>所有 CNN 模型於 270 張潔淨驗證影像上均達到極低錯誤率。</p>"
    )

    body = f"""
<div class="report-header">
  <div class="course-tag">影像識別期末專題報告｜FactoryVision Challenge — Stage 1</div>
  <h1>Stage 1：傳統基準方法與自建 CNN 消融實驗</h1>
  <p class="author-line">組別：Team21｜隨機種子：42</p>
</div>

<table class="summary-strip">
  <tr>
    <td><span class="lbl">最終選定模型</span><span class="val">{sel_id} (BN + DataAug)</span></td>
    <td><span class="lbl">Validation Macro F1</span><span class="val">{sel['best_val']['macro_f1']:.4f}</span></td>
    <td><span class="lbl">Mild-Shift 偏移診斷 F1</span><span class="val">{s1_diag.get(sel_id, {}).get('perturbed_val_f1', 0):.4f}</span></td>
    <td><span class="lbl">參數量 / 檔案大小</span><span class="val">{sel['params']['total_params']:,} / {s1_cnn['checkpoint_size_mb']:.2f} MB</span></td>
  </tr>
</table>

<h2>1. 資料規範與實驗設計</h2>
<p>Stage 1 的目標是從頭建立鋼材表面瑕疵分類的基準線，並驗證常見的訓練改善技巧是否真的有效。為了確保比較公平，本階段僅使用 <code>Classification Public Dataset</code>（Train = 1,260 張，每類 210 張；Validation = 270 張，每類 45 張），固定 <code>seed = 42</code>，也不使用任何預訓練權重或現成骨幹（如 ResNet、VGG、MobileNet）。所有影像統一轉為 <code>200×200</code> 單通道灰階，並以訓練集統計量 <code>mean = [0.5071]</code>、<code>std = [0.2132]</code> 做標準化。</p>

<h2>2. Part A：Traditional Baselines（PCA + SVM / PCA + KNN）</h2>
<p>在寫 CNN 之前，我們先用傳統機器學習方法當作基準。作法是將灰階影像縮成 <code>64×64</code>（4,096 維向量），只在訓練集上擬合 <code>StandardScaler</code> 與保留 95% 變異量的 <code>PCA(n_components=0.95)</code>（實際保留 <b>{s1_trad['pca_n_components']}</b> 維主成分，累積解釋變異量為 <b>{s1_trad['pca_explained_variance']*100:.2f}%</b>），接著分別訓練 <code>RBF-SVM (C=10.0, gamma='scale')</code> 與 <code>KNN (k=5, distance weighting)</code>。</p>
<table>
  <thead>
    <tr><th class="left">傳統基準方法</th><th>PCA 維度</th><th>Train Acc</th><th>Val Accuracy</th><th>Val Precision</th><th>Val Recall</th><th>Val Macro F1</th></tr>
  </thead>
  <tbody>
    <tr>
      <td class="left"><b>PCA + SVM (RBF)</b></td>
      <td>{s1_trad['pca_n_components']}</td>
      <td>{svm['train_accuracy']:.4f}</td>
      <td>{svm['accuracy']:.4f}</td>
      <td>{svm['macro_precision']:.4f}</td>
      <td>{svm['macro_recall']:.4f}</td>
      <td><b>{svm['macro_f1']:.4f}</b></td>
    </tr>
    <tr>
      <td class="left"><b>PCA + KNN (k=5)</b></td>
      <td>{s1_trad['pca_n_components']}</td>
      <td>{knn['train_accuracy']:.4f}</td>
      <td>{knn['accuracy']:.4f}</td>
      <td>{knn['macro_precision']:.4f}</td>
      <td>{knn['macro_recall']:.4f}</td>
      <td><b>{knn['macro_f1']:.4f}</b></td>
    </tr>
  </tbody>
</table>
<p>從結果來看，<code>PCA + SVM</code>（Macro F1 = <b>{svm['macro_f1']:.4f}</b>）比 <code>PCA + KNN</code>（<b>{knn['macro_f1']:.4f}</b>）好不少，但兩者都明顯受限：特別是 <code>PCA + KNN</code> 在訓練集拿到 1.0000、驗證集卻掉到 0.6370，而 <code>PCA + SVM</code> 對細長紋理（<code>inclusion</code> F1 = 0.5843、<code>scratches</code> F1 = 0.6571）也常誤判。這主要是因為直接把影像拉平做全域 PCA 會破壞局部空間結構與平移不變性。</p>

<h2>3. Self-Designed Baseline CNN 架構與訓練設定</h2>
<p>我們自己設計了一個四層的輕量卷積網路 <code>SteelDefectCNN</code>：包含 4 個卷積區塊（通道數 <code>1 → 32 → 64 → 128 → 256</code>，皆為 <code>3×3</code> kernel、<code>padding=1</code>、<code>Kaiming Normal</code> 初始化與 <code>ReLU</code>），前 3 個區塊接 <code>MaxPool2d(2×2)</code>（解析度由 <code>200 → 100 → 50 → 25</code>），第 4 個區塊後接 <code>AdaptiveAvgPool2d(1×1)</code>，最後以 <code>Linear(256→128) → ReLU → (可選 Dropout) → Linear(128→6)</code> 輸出分類結果。</p>
<ul>
  <li><b>Baseline（E1）前處理</b>：僅使用 <code>Resize(200, 200) → ToTensor() → Normalize([0.5071], [0.2132])</code>。</li>
  <li><b>固定訓練參數</b>：<code>Adam(lr=1e-3)</code>、<code>Batch Size = 32</code>、<code>Epochs = 80</code>、<code>CrossEntropyLoss</code>、<code>Seed = 42</code>。</li>
</ul>

<div class="page-break"></div>

<h2>4. Part B：Improvement Methods 與消融實驗（Ablation Study）</h2>
<p>為了看清楚每個改善方法的獨立效果，我們以 <code>E1</code> 為基準，依序測試以下三組設定（E1–E5）：</p>
<ul>
  <li><b>Method A（Batch Normalization）</b>：在每個 <code>Conv2d</code> 後面加上 <code>BatchNorm2d</code>（同時關閉 conv bias），幫助深層特徵分佈穩定。</li>
  <li><b>Method B（Data Augmentation）</b>：訓練時加入隨機水平翻轉（<code>p=0.5</code>）、垂直翻轉（<code>p=0.5</code>）、旋轉（<code>±15°</code>）與亮度對比調整（<code>ColorJitter(brightness=0.2, contrast=0.2)</code>）。</li>
  <li><b>Method C（Regularization + Early Stopping）</b>：分類頭加入 <code>Dropout(p=0.3)</code>、優化器設定 <code>weight_decay=1e-4</code>，並以 Validation Macro F1 監控 <code>Early Stopping(patience=20)</code>。</li>
</ul>

<table>
  <thead>
    <tr>
      <th class="left">實驗組別</th><th>Method A<br>(BN)</th><th>Method B<br>(Aug)</th><th>Method C<br>(Drop/L2/ES)</th>
      <th>Best / Ran<br>Epoch</th><th>Val Acc</th><th>Val<br>Macro F1</th><th>Best Val<br>Loss</th>
      <th>Last-10<br>Val F1</th><th>Mild-Shift<br>Gen F1</th><th>Final Overfit<br>Gap (F1)</th><th>訓練時間</th>
    </tr>
  </thead>
  <tbody>
    {''.join(ablation_rows)}
  </tbody>
</table>

{img_tag(fig_dir / "fig1_stage1_ablation_curves.png", "圖 1：E1–E5 訓練與驗證收斂曲線（Loss 與 Macro F1）。加入 BatchNorm (E2/E4/E5) 顯著加速收斂並壓低驗證震盪。")}
{img_tag(fig_dir / "fig2_stage1_comparison_bar.png", "圖 2：Traditional Baselines 與 E1–E5 CNN 之 Validation Macro F1 與 Accuracy 比較。", max_height_px=310)}

<div class="page-break"></div>

<h2>5. 混淆矩陣、逐類分析與 Final Model 選擇理由</h2>
{img_tag(fig_dir / "fig3_stage1_confusion_matrices.png", "圖 3：PCA+SVM、E1 Baseline CNN 與最終選定 Improved CNN 之驗證集混淆矩陣對照。", max_height_px=280)}
{img_tag(fig_dir / "fig4_stage1_per_class_f1.png", "圖 4：各缺陷類別 F1 分數對比（PCA+SVM vs. Baseline CNN vs. Final Improved CNN）。", max_height_px=280)}

<table>
  <thead>
    <tr><th class="left">缺陷類別</th><th>PCA + SVM F1</th><th>E1 Baseline F1</th><th>{sel_id} Precision</th><th>{sel_id} Recall</th><th>{sel_id} F1</th><th>Support</th></tr>
  </thead>
  <tbody>
    {''.join(pc_rows)}
  </tbody>
</table>

<div class="note-box">
  <b>Final Improved CNN 選擇理由（選定 {sel_id}）</b>：
  <ul>
    <li><b>為什麼在 Clean Val 都拿 1.0000 的情況下還要比較？</b> 一開始跑完 E1–E5 時我們有點意外，五組在 270 張潔淨驗證集上最佳檢查點都達到 <code>1.0000</code>。但如果只看最高 F1 其實分不出好壞，所以我們進一步檢查收斂速度、Validation Loss 與偏移泛化力：
      (1) 單加 <b>BatchNorm（E2）</b> 讓最佳 Val Loss 從 E1 的 <code>{e1['best_val']['loss']:.5f}</code> 降到 <code>{exps['E2']['best_val']['loss']:.5f}</code>，且第 {s1_diag.get('E2', {}).get('ep_ge_098', '-')} epoch 就突破 0.98 F1（E1 要到第 {s1_diag.get('E1', {}).get('ep_ge_098', '-')} epoch）；
      (2) 單加 <b>Data Augmentation（E3）</b> 把 Val Loss 降到 <code>{exps['E3']['best_val']['loss']:.5f}</code>，同時在輕微旋轉／亮度偏移診斷（Mild-Shift Probe）中把 F1 從無增強組的 <code>{s1_diag.get('E1', {}).get('perturbed_val_f1', 0):.4f}</code>（E1）與 <code>{s1_diag.get('E2', {}).get('perturbed_val_f1', 0):.4f}</code>（E2）拉高到 <b>{s1_diag.get('E3', {}).get('perturbed_val_f1', 0):.4f}</b>。</li>
    <li><b>最終選擇 {sel_id}（A + B 結合）</b>：<code>E4</code> 在第 <b>{s1_diag.get('E4', {}).get('ep_ge_098', '-')}</b> epoch 就達到 0.98 F1（五組最快），最後 10 個 epoch 的平均 Val F1 最高（<b>{s1_diag.get('E4', {}).get('last10_val_f1', 0):.4f}</b>）、平均 Val Loss 最低（<b>{s1_diag.get('E4', {}).get('last10_val_loss', 0):.4f}</b>），而且在偏移測試上拿到滿分 <b>{s1_diag.get('E4', {}).get('perturbed_val_f1', 0):.4f}</b>。比較之下，<code>E5</code> 在僅 42 萬參數的小模型上同時疊加 Dropout(0.3) 與 L2(1e-4) 並在第 66 epoch 提早停止，反而略微欠擬合（Val Loss = {exps['E5']['best_val']['loss']:.5f}）。</li>
    <li><b>模型大小與參數量</b>：<code>{sel_id}</code> 僅 <b>{sel['params']['total_params']:,}</b> 參數（<b>{s1_cnn['checkpoint_size_mb']:.2f} MB</b>），在輕量架構下兼顧高準確率與穩定泛化表現。</li>
  </ul>
</div>

<h3>Overfitting / Generalization 診斷與錯誤分析</h3>
<p>在潔淨驗證集（N=270）上 <b>{sel_id}</b> 並沒有出現誤判樣本（0 errors）；為了確認模型不是單純記住原始拍攝亮度與角度，我們另外用 <code>seed=42</code> 對驗證集做 ±15° 旋轉與 ±20% 亮度對比偏移測試（僅用於診斷，未參與訓練）：</p>
<ul>{err_lines}</ul>
{err_fig_html}
"""
    return wrap_doc("Team21 Stage 1 Report", body)


def build_stage2_report(s2: Dict, bench: Dict) -> str:
    exps = s2["experiments"]
    sel_id = s2["selected_model_id"]
    sel = exps[sel_id]
    fig_dir = ROOT / "workspace_outputs" / "stage2" / "figs"

    strat_label = {
        "feature_extraction": "Feature Extraction",
        "partial_finetuning": "Partial Fine-tuning",
        "full_finetuning": "Full Fine-tuning",
    }
    arch_label = {
        "densenet121": "DenseNet121",
        "mobilenet_v3_large": "MobileNetV3-Large",
    }

    main_rows = []
    param_rows = []
    for eid in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        e = exps[eid]
        cv = e["clean_val"]
        rb = e["robustness"]
        ls = e["layer_summary"]
        cls_attr = " class='highlight'" if eid == sel_id else ""
        main_rows.append(
            f"<tr{cls_attr}>"
            f"<td><b>{eid}</b></td>"
            f"<td class='left'>{arch_label[e['architecture']]}</td>"
            f"<td class='left'>{strat_label[e['strategy']]}</td>"
            f"<td>{cv['accuracy']:.4f}</td>"
            f"<td>{cv['macro_precision']:.4f}</td>"
            f"<td>{cv['macro_recall']:.4f}</td>"
            f"<td><b>{cv['macro_f1']:.4f}</b></td>"
            f"<td>{rb['lighting']['macro_f1']:.4f}</td>"
            f"<td>{rb['optical']['macro_f1']:.4f}</td>"
            f"<td>{rb['sensor_compression']['macro_f1']:.4f}</td>"
            f"<td><b>{rb['robust_macro_f1']:.4f}</b></td>"
            f"</tr>"
        )
        frozen_txt = ", ".join(ls["frozen_modules"]) if ls["frozen_modules"] else "None (All Trainable)"
        train_txt = ", ".join(ls["trainable_modules"]) if ls["trainable_modules"] else "None"
        param_rows.append(
            f"<tr{cls_attr}>"
            f"<td><b>{eid}</b></td>"
            f"<td class='left'>{arch_label[e['architecture']]} / {strat_label[e['strategy']]}</td>"
            f"<td>{ls['total_params']:,}</td>"
            f"<td>{ls['trainable_params']:,} ({ls['trainable_ratio_pct']:.1f}%)</td>"
            f"<td>{e['checkpoint_size_mb']:.2f} MB</td>"
            f"<td class='left'><code>{html.escape(frozen_txt)}</code></td>"
            f"<td class='left'><code>{html.escape(train_txt)}</code></td>"
            f"</tr>"
        )

    b_std = bench["stage2"]["hidden_standard_macro_f1"]
    b_rob = bench["stage2"]["hidden_robust_macro_f1"]

    body = f"""
<div class="report-header">
  <div class="course-tag">影像識別期末專題報告｜FactoryVision Challenge — Stage 2</div>
  <h1>Stage 2：遷移學習架構與微調策略比較報告</h1>
  <p class="author-line">組別：Team21｜實驗設計：2 Architectures × 3 Strategies｜隨機種子：42</p>
</div>

<table class="summary-strip">
  <tr>
    <td><span class="lbl">最終選定架構</span><span class="val">{arch_label[sel['architecture']]}</span></td>
    <td><span class="lbl">微調策略</span><span class="val">{strat_label[sel['strategy']]}</span></td>
    <td><span class="lbl">Clean Val Macro F1</span><span class="val">{sel['clean_val']['macro_f1']:.4f}</span></td>
    <td><span class="lbl">Local Robust Macro F1</span><span class="val">{sel['robustness']['robust_macro_f1']:.4f}</span></td>
  </tr>
</table>

<h2>1. 實驗架構矩陣與設計動機</h2>
<p>Stage 2 希望把骨幹網路（Architecture）與微調策略（Fine-tuning Strategy）兩者的影響分開來看，所以我們選了兩個結構差異較大的預訓練模型家族，各自搭配三種微調方式做 <b>2 × 3 = 6 組</b>控制實驗：</p>
<ul>
  <li><b>DenseNet121</b>（<code>DenseNet121_Weights.IMAGENET1K_V1</code>）：透過密集連接把淺層紋理一路往後傳，理論上對表面細紋辨識很有幫助，但參數量與計算量較高。</li>
  <li><b>MobileNetV3-Large</b>（<code>MobileNet_V3_Large_Weights.IMAGENET1K_V2</code>）：使用深度可分離卷積搭配 SE 注意力，模型小、參數量低，很適合邊緣產線部署。</li>
  <li><b>三種微調策略</b>：
    (1) <code>Feature Extraction</code>（凍結全部骨幹，只訓練分類頭，<code>lr=1e-3</code>）；
    (2) <code>Partial Fine-tuning</code>（DenseNet121 解凍最後的 <code>denseblock4 + norm5 + classifier</code>；MobileNetV3-Large 解凍 <code>features[13:17] + classifier</code>；骨幹 <code>lr=3e-4</code>、Head <code>lr=1e-3</code>）；
    (3) <code>Full Fine-tuning</code>（全網路解凍，採用差異學習率：骨幹 <code>lr=1.5e-4</code>、Head <code>lr=1e-3</code>）。</li>
  <li><b>訓練與前處理一致性</b>：6 組實驗全部使用相同切分（Train 1,260 / Val 270）、<code>seed=42</code>、<code>RGB 224×224</code> ImageNet 標準化、<code>AdamW(weight_decay=1e-4)</code>、<code>CosineAnnealingLR(35 epochs)</code>、<code>Label Smoothing=0.05</code>，並統一在訓練資料增強加入模擬現場光照偏移、模糊、感測器雜訊與 JPEG 壓縮；推論時則嚴格使用單一固定前處理，不依路徑切換任何規則。</li>
</ul>

<h2>2. 2×3 公平比較結果：Standard 與 Local Robustness 壓力測試</h2>
<p>針對常見的 <code>lighting</code>、<code>optical</code> 與 <code>sensor_compression</code> 干擾情境，我們用固定種子在驗證階段對 270 張驗證圖加上對應的合成干擾來做壓力測試（這些干擾版本不曾用來挑超參數，best epoch 一律以 Clean Val Macro F1 決定）：</p>
<table>
  <thead>
    <tr>
      <th>實驗</th><th class="left">Architecture</th><th class="left">Strategy</th>
      <th>Clean<br>Acc</th><th>Clean<br>Prec</th><th>Clean<br>Rec</th><th>Clean<br>Macro F1</th>
      <th>Lighting<br>F1</th><th>Optical<br>F1</th><th>Sensor/JPEG<br>F1</th><th>Local Robust<br>Macro F1</th>
    </tr>
  </thead>
  <tbody>
    {''.join(main_rows)}
  </tbody>
</table>

{img_tag(fig_dir / "fig2_stage2_robustness_bar.png", "圖 1：6 組模型於 Clean Validation 與三種局部干擾測試（Lighting / Optical / Sensor-JPEG）之 Macro F1 對照。", max_height_px=270)}
{img_tag(fig_dir / "fig5_stage2_corruption_strip.png", "圖 2：本地 Robustness 壓力測試合成干擾範例（僅於驗證評估時產生）。", max_height_px=180)}

<h2>3. 參數規模、凍結層明細與收斂曲線</h2>
<table class="allow-split">
  <thead>
    <tr>
      <th>實驗</th><th class="left">模型組合</th><th>Total Params</th><th>Trainable Params</th>
      <th>檔案大小</th><th class="left">Frozen Layers</th><th class="left">Trainable Layers</th>
    </tr>
  </thead>
  <tbody>
    {''.join(param_rows)}
  </tbody>
</table>

<div class="page-break"></div>

{img_tag(fig_dir / "fig1_stage2_curves.png", "圖 3：DenseNet121（上）與 MobileNetV3-Large（下）於三種微調策略下之訓練／驗證收斂曲線。", max_height_px=420)}

<div class="page-break"></div>

<h2>4. 部署權衡分析、混淆矩陣與 Final Model Decision</h2>
{img_tag(fig_dir / "fig3_stage2_deployment_tradeoff.png", "圖 4：模型參數量、檔案大小（氣泡面積）與強健性 Macro F1 之工程權衡分佈圖。", max_height_px=320)}
{img_tag(fig_dir / "fig4_stage2_final_cm.png", f"圖 5：最終選定模型 ({sel_id}) 於 Clean Validation (N=270) 與彙總 Robustness 測試 (N=810) 之混淆矩陣。", max_height_px=270)}

<div class="note-box">
  <b>Stage 2 選模結論（選定 {sel_id}: {arch_label[sel['architecture']]} + {strat_label[sel['strategy']]}）</b>：
  <ul>
    <li><b>凍結骨幹 vs. 微調的差異</b>：雖然 DenseNet121 與 MobileNetV3-Large 在潔淨資料上只靠 <code>Feature Extraction</code> 就能拿到 0.9926 與 1.0000，但在本地干擾測試下差距立刻拉開——特別是 <code>DenseNet121 FE (E1)</code> 在 <code>Sensor/JPEG</code> 降到 <code>{exps['E1']['robustness']['sensor_compression']['macro_f1']:.4f}</code>（Robust 平均 <code>{exps['E1']['robustness']['robust_macro_f1']:.4f}</code>）。原因在於 ImageNet 預訓練的淺層濾波器對高頻壓縮失真很敏感，必須透過微調讓整個網路適應鋼材紋理域。</li>
    <li><b>為什麼選 {sel_id} 而不是 DenseNet121 Full FT ({exps['E3']['robustness']['robust_macro_f1']:.4f})？</b> <code>E6</code>（MobileNetV3-Large Full Fine-tuning）不僅在 Local Robust Macro F1 拿到最高分（<b>{sel['robustness']['robust_macro_f1']:.4f}</b>，810 張干擾測試僅 5 筆誤判），模型檔案也只有 <b>{sel['checkpoint_size_mb']:.2f} MB</b>（DenseNet121 為 27.09 MB），參數量也僅約 DenseNet121 的 60%（<b>{sel['layer_summary']['total_params']:,}</b>）。在精度、強健性與部署成本三方面都更划算。</li>
    <li><b>離線驗證</b>：<code>stage2_inference.py</code> 固定以 <code>pretrained=False</code> 建構模型並從本地 <code>stage2_model.pth</code> 載入權重，已在外部工作目錄驗證通過。</li>
  </ul>
</div>
"""
    return wrap_doc("Team21 Stage 2 Report", body)


def build_stage3_report(s3: Dict, bench: Dict) -> str:
    fig_dir = ROOT / "workspace_outputs" / "stage3" / "figs"
    dets = s3["detectors"]
    sel_id = s3["selected_model_id"]
    fm = s3["final_val_metrics"]
    b_map = bench["stage3"]["hidden_map50"]

    comp_rows = []
    for did in ["yolo11s", "yolo11m", "faster_rcnn"]:
        d = dets[did]
        ev001 = d["eval_conf001"]
        ev25 = d["eval_conf025"]
        cls_attr = " class='highlight'" if did == sel_id else ""
        comp_rows.append(
            f"<tr{cls_attr}>"
            f"<td class='left'><b>{html.escape(d['detector'])}</b></td>"
            f"<td>{d['total_params']:,}</td>"
            f"<td>{d['file_size_mb']:.1f} MB</td>"
            f"<td><b>{ev001['mAP50']:.4f}</b></td>"
            f"<td>{ev25['mAP50']:.4f}</td>"
            f"<td>{ev25['precision']:.4f}</td>"
            f"<td>{ev25['recall']:.4f}</td>"
            f"</tr>"
        )

    pc_rows = []
    for c in CLASS_ORDER:
        p = fm["per_class"][c]
        pc_rows.append(
            f"<tr>"
            f"<td class='left'><code>{c}</code></td>"
            f"<td>{p['gt_count']}</td><td>{p['pred_count']}</td>"
            f"<td>{p['tp']}</td><td>{p['fp']}</td><td>{p['fn']}</td>"
            f"<td>{p['precision']:.4f}</td><td>{p['recall']:.4f}</td>"
            f"<td><b>{p['ap50']:.4f}</b></td>"
            f"</tr>"
        )

    sweep_rows = []
    for sw in s3["conf_sweep"]:
        cls_attr = " class='highlight'" if abs(sw["conf_threshold"] - s3["confidence_threshold"]) < 1e-6 else ""
        sweep_rows.append(
            f"<tr{cls_attr}>"
            f"<td>{sw['conf_threshold']:g}</td>"
            f"<td><b>{sw['mAP50']:.4f}</b></td>"
            f"<td>{sw['precision']:.4f}</td>"
            f"<td>{sw['recall']:.4f}</td>"
            f"<td>{sw['f1']:.4f}</td>"
            f"<td>{sw['avg_dets_per_img']:.2f}</td>"
            f"</tr>"
        )

    # Failure case details (2 FP, 2 FN, 2 Poor Loc)
    fails = s3["failures"]
    fail_rows = []
    for k, label_zh in [
        ("false_positives", "False Positive (誤檢)"),
        ("false_negatives", "False Negative (漏檢)"),
        ("poor_localization", "Poor Localization (IoU<0.5)"),
    ]:
        seen = set()
        picked = []
        for item in fails[k]:
            if item["image_id"] not in seen:
                picked.append(item)
                seen.add(item["image_id"])
            if len(picked) == 2:
                break
        for it in picked:
            if k == "false_positives":
                desc = f"預測為 <code>{it['pred_class']}</code> (conf={it['confidence']:.2f})，與最近 GT (<code>{it['nearest_gt_class']}</code>) 之 IoU 僅 {it['max_iou_any_gt']:.2f}"
                cause = "細微金屬反光紋理被誤觸發，或重疊缺陷造成跨類混淆/重複預測"
            elif k == "false_negatives":
                desc = f"漏檢 <code>{it['gt_class']}</code>（GT 框面積 {it['gt_area']:.0f} px²）"
                cause = "目標面積狹小或邊界對比度低，與背景滾軋紋理融為一體"
            else:
                desc = f"類別 <code>{it['class']}</code> (conf={it['confidence']:.2f}) 正確，但 IoU={it['iou']:.3f} &lt; 0.50"
                cause = "長條形或散佈型缺陷邊界模糊，預測框偏移或涵蓋範圍過大/過小"
            fail_rows.append(
                f"<tr><td><b>{label_zh}</b></td><td><code>{it['image_id']}</code></td><td class='left'>{desc}</td><td class='left'>{cause}</td></tr>"
            )

    sel_op25 = dets[sel_id]["eval_conf025"]
    y11m = dets["yolo11m"]
    y11s = dets["yolo11s"]
    sweep_by_c = {round(sw["conf_threshold"], 3): sw for sw in s3["conf_sweep"]}
    sw_sub = sweep_by_c.get(round(s3["confidence_threshold"], 3), {})
    sw_25 = sweep_by_c.get(0.25, {})
    sw_50 = sweep_by_c.get(0.50, {})

    body = f"""
<div class="report-header">
  <div class="course-tag">影像識別期末專題報告｜FactoryVision Challenge — Stage 3</div>
  <h1>Stage 3：鋼材表面瑕疵 Object Detection 實驗報告</h1>
  <p class="author-line">組別：Team21｜模型比較：YOLO11s / YOLO11m vs. Faster R-CNN R50-FPN-v2｜隨機種子：42</p>
</div>

<table class="summary-strip">
  <tr>
    <td><span class="lbl">選定 Detector</span><span class="val">{html.escape(s3['selected_detector'])}</span></td>
    <td><span class="lbl">Val mAP@0.5 (c={s3['confidence_threshold']})</span><span class="val">{fm['mAP50']:.4f}</span></td>
    <td><span class="lbl">Val P / R (Op c=0.25)</span><span class="val">{sel_op25['precision']:.3f} / {sel_op25['recall']:.3f}</span></td>
    <td><span class="lbl">參數量 / 檔案大小</span><span class="val">{dets[sel_id]['total_params']:,} / {s3['checkpoint_size_mb']:.1f} MB</span></td>
  </tr>
</table>

<h2>1. 檢測器架構設計與訓練設定</h2>
<p>Stage 3 將問題從「整張圖是什麼瑕疵」延伸到「每個瑕疵在哪裡」（Train=1,260 / Validation=270，驗證集共 641 個標註框）。實作時我們先注意到 VOC XML 標註把 <code>rolled_in_scale</code> 寫成帶連字號的 <code>rolled-in_scale</code>，因此在資料讀取與評估器中統一做了名稱正規化。為了比較單階段與雙階段檢測器的差異，我們在相同切分與 <code>seed=42</code> 下訓練了三組模型：</p>
<ul>
  <li><b>YOLO11s 與 YOLO11m（單階段 Anchor-free）</b>：使用 Ultralytics COCO 預訓練權重，將 <code>200×200</code> 影像上採樣至 <code>imgsz=640</code> 訓練 100 epochs（<code>fliplr=0.5, flipud=0.5, mosaic=1.0, close_mosaic=15, hsv_v=0.3, scale=0.4, patience=25</code>）。</li>
  <li><b>Faster R-CNN ResNet50-FPN-v2（雙階段 Anchor-based）</b>：使用 Torchvision COCO_V1 預訓練權重，透過 FPN 多尺度金字塔（P2–P6）搭配 RPN 與 RoIAlign 進行兩階段邊界修正，設定 <code>min_size=600, max_size=800</code>、<code>AdamW(lr=1e-4, weight_decay=1e-4)</code>、<code>CosineAnnealingLR(25 epochs)</code> 與混合精度訓練。</li>
</ul>

<h2>2. 檢測器效能比較（依 Slide-31 Pascal VOC 2010+ 定義）</h2>
<p>為避免不同套件內建 mAP 算式不一致，下表所有數據皆由我們依投影片第 31 頁規格獨立實作的評估程式（<code>IoU ≥ 0.5</code> 貪婪一對一匹配、Pascal VOC 2010+ 全點插值 AP）統一計算：</p>
<table>
  <thead>
    <tr>
      <th class="left">Detector</th><th>Params</th><th>Size</th>
      <th>mAP@0.5<br>(conf=0.01)</th><th>mAP@0.5<br>(conf=0.25)</th>
      <th>Precision<br>(conf=0.25)</th><th>Recall<br>(conf=0.25)</th>
    </tr>
  </thead>
  <tbody>
    {''.join(comp_rows)}
  </tbody>
</table>

{img_tag(fig_dir / "fig1_stage3_detector_comparison.png", "圖 1：YOLO11s、YOLO11m 與 Faster R-CNN R50-FPN-v2 於完整 PR 曲線下（conf=0.01）之逐類 AP@0.5 與整體 mAP@0.5 比較。", max_height_px=290)}
{img_tag(fig_dir / "fig3_stage3_training_curves.png", "圖 2：YOLO11s/m 與 Faster R-CNN 之訓練收斂與驗證 mAP@0.5 曲線。", max_height_px=260)}

<div class="page-break"></div>

<h2>3. IoU、Confidence Threshold、NMS 機制與超參數敏感度分析</h2>
<ul>
  <li><b>IoU（Intersection over Union = 0.5）</b>：定義預測框與 Ground Truth 之重疊比例。<code>IoU ≥ 0.5</code> 為 True Positive 門檻，類別錯誤、<code>IoU &lt; 0.5</code> 或同一 GT 之重複預測皆計為 False Positive。</li>
  <li><b>Confidence Threshold（提交設定 <code>{s3['confidence_threshold']}</code> vs. 產線警報 <code>0.25–0.50</code>）</b>：由於 Slide-31 Pascal VOC <code>mAP@0.5</code> 積分整條 Precision-Recall 曲線，過高門檻會提前截斷高召回尾部而壓低 AP；因此正式 <code>stage3_detect.py</code> 採用 <code>conf = {s3['confidence_threshold']}</code>（平均每張 <b>{sw_sub.get('avg_dets_per_img', 12.3):.2f}</b> 框，遠低於 100 框上限），完整保留 PR 面積達成 <b>mAP@0.5 = {fm['mAP50']:.4f}</b>（Recall = <b>{fm['recall']:.4f}</b>，Precision = <b>{fm['precision']:.4f}</b>）。於實際產線告警與視覺化時，則將顯示／警報門檻調高至 <code>conf = 0.25–0.50</code>，使 Precision 由 {fm['precision']:.3f} 大幅拉升至 <b>{sw_25.get('precision', 0):.3f}–{sw_50.get('precision', 0):.3f}</b>（F1 = <b>{sw_25.get('f1', 0):.3f}–{sw_50.get('f1', 0):.3f}</b>）。</li>
  <li><b>NMS IoU Threshold（選定 <code>{s3['nms_iou_threshold']}</code>）與 Anchor vs. Anchor-free 機制比較</b>：單階段 <code>YOLO11</code> 採 Anchor-free DFL 頭直接回歸中心到四邊距離，省去手動 Anchor 尺度調校並達成極快推論；雙階段 <code>Faster R-CNN ResNet50-FPN-v2</code> 則透過多尺度 RPN Anchors 先過濾前景候選框再經 RoIAlign 進行第二次精細邊界回歸，於瀰漫弱紋理類別上展現顯著優勢（<code>crazing</code> AP = <b>{fm['per_class']['crazing']['ap50']:.4f}</b> vs. YOLO11s 0.3797；<code>rolled_in_scale</code> AP = <b>{fm['per_class']['rolled_in_scale']['ap50']:.4f}</b> vs. YOLO11m 0.5961），並以 <code>NMS IoU = {s3['nms_iou_threshold']}</code> 抑制同一目標冗餘框。</li>
</ul>

<div style="display:grid; grid-template-columns: 1.15fr 1fr; gap:10px; align-items:start;">
  <div>
    <table>
      <thead>
        <tr><th>Conf</th><th>mAP@0.5</th><th>Precision</th><th>Recall</th><th>F1</th><th>Avg Dets/Img</th></tr>
      </thead>
      <tbody>{''.join(sweep_rows)}</tbody>
    </table>
  </div>
  <div>
    <table>
      <thead>
        <tr><th class="left">Class (Final @ c={s3['confidence_threshold']})</th><th>GT</th><th>TP</th><th>FP</th><th>FN</th><th>AP@0.5</th></tr>
      </thead>
      <tbody>{''.join(f"<tr><td class='left'><code>{c}</code></td><td>{fm['per_class'][c]['gt_count']}</td><td>{fm['per_class'][c]['tp']}</td><td>{fm['per_class'][c]['fp']}</td><td>{fm['per_class'][c]['fn']}</td><td><b>{fm['per_class'][c]['ap50']:.4f}</b></td></tr>" for c in CLASS_ORDER)}</tbody>
    </table>
  </div>
</div>

{img_tag(fig_dir / "fig2_stage3_pr_and_threshold.png", "圖 3：(左) Confidence Threshold 對 Precision/Recall/F1/mAP@0.5 之影響；(右) 最終選定模型各類別 Precision-Recall 曲線。", max_height_px=275)}

<div class="page-break"></div>

<h2>4. Validation 偵測結果視覺化（12 張 Ground Truth vs. Prediction 對照）</h2>
{img_tag(fig_dir / "fig4_stage3_vis_part1.png", "圖 4：Validation 偵測結果視覺化（Part 1，共 6 張，含多類別共存影像；實線=GT、虛線=Prediction @ 顯示門檻 conf≥0.25）。", max_height_px=430)}
{img_tag(fig_dir / "fig4_stage3_vis_part2.png", "圖 5：Validation 偵測結果視覺化（Part 2，共 6 張，涵蓋所有 6 種鋼材瑕疵類別；@ 顯示門檻 conf≥0.25）。", max_height_px=430)}

<div class="page-break"></div>

<h2>5. Failure Analysis：錯誤案例與 Slide-22 核心工程問題回答</h2>
{img_tag(fig_dir / "fig5_stage3_failure_cases.png", "圖 6：6 個代表性失效案例（於操作門檻 conf≥0.25 抽取 False Positive、False Negative、Poor Localization 各 2 例）。", max_height_px=470)}

<table>
  <thead>
    <tr><th>失效類型</th><th>影像 ID</th><th class="left">實際現象數據 (conf≥0.25)</th><th class="left">可能成因診斷</th></tr>
  </thead>
  <tbody>{''.join(fail_rows)}</tbody>
</table>

<div class="note-box">
  <b>Slide-22 核心問題回答與工程討論</b>：
  <ol>
    <li><b>模型漏掉了哪些 defect？與物件大小、紋理或背景相似有關嗎？</b> 漏檢（FN）主要出現在面積偏小的 <code>inclusion</code>（驗證集中位寬只有 23 px，例如 <code>VAL_0067</code> 角落 308 px² 的小夾雜）以及對比度很弱的 <code>crazing</code>／<code>rolled_in_scale</code>，當瑕疵本身幾乎融入熱軋背景紋理時特別容易漏掉。</li>
    <li><b>Bounding Box 為什麼偏移或過大？IoU 不足的原因是什麼？</b> 像是 <code>pitted_surface</code> 與 <code>crazing</code> 屬於大面積散佈型紋理，標註框邊界本身就不像剛體物件那麼明確。從 <code>VAL_0173</code> 可以直接看到，Ground Truth 標成左右兩個相鄰的 <code>pitted_surface</code> 框，但模型把它們合併預測成一個涵蓋整張圖的大框（<code>IoU = 0.500</code> 未達標準），這類合併或只框到核心區域的情況是 Poor Localization 的主因。</li>
    <li><b>調整 Confidence Threshold 後，Precision 與 Recall 如何改變？</b> Threshold 從 0.01 提高到 0.50 時，Precision 一路從 <b>{fm['precision']:.3f}</b> 升到 <b>{sw_50.get('precision', 0):.3f}</b>，Recall 則從 <b>{fm['recall']:.3f}</b> 降到 <b>{sw_50.get('recall', 0):.3f}</b>。若產線擔心漏檢，threshold 設在 <code>0.20–0.25</code> 比較合適；若要減少誤報則可拉高到 <code>0.40–0.50</code>。</li>
    <li><b>模型複雜度與精度的取捨如何？更大的 Detector 是否真的值得？</b> 這裡有很明顯的取捨：<code>Faster R-CNN R50-FPN-v2</code> 雖然拿到最高的 <b>mAP@0.5 = {fm['mAP50']:.4f}</b>，但參數量達 <b>{dets[sel_id]['total_params']:,}</b>（權重 <b>{s3['checkpoint_size_mb']:.1f} MB</b>），比較適合高算力、重視極致召回率的品管複檢站；若產線端受限於記憶體與硬體資源，改用 <code>YOLO11m</code>（<b>{y11m['total_params']:,}</b> 參數 / <b>{y11m['file_size_mb']:.1f} MB</b>）或 <code>YOLO11s</code>（<b>{y11s['total_params']:,}</b> 參數 / <b>{y11s['file_size_mb']:.1f} MB</b>）會輕量許多。另外也觀察到在 <code>VAL_0250</code> 等案例中，有些所謂 False Positive 其實是鋼板上原本就存在、但未標註到的細微刮痕。</li>
    <li><b>若部署到工廠，你會使用這個 Detector 嗎？為什麼？</b> <b>會，但會依產線工位做搭配。</b> 因為單標籤分類沒辦法處理一張圖出現多種瑕疵的情況（本資料集共有 104 張複合瑕疵圖），Detector 才能同時回答「是什麼」跟「在哪裡」。精檢站用 <code>Faster R-CNN R50-FPN-v2</code> 確保 <code>crazing</code> 與 <code>rolled_in_scale</code> 不漏接；資源受限的邊緣工位則可部署輕量的 <code>YOLO11m</code>。</li>
  </ol>
</div>
"""
    return wrap_doc("Team21 Stage 3 Report", body)


def compute_rubric_estimates(bench: Dict) -> Dict:
    s1_f1 = bench["stage1"]["hidden_macro_f1"]
    s1_pts = bench["stage1"]["rubric_points_max10"]
    if s1_pts is None and s1_f1 is not None:
        s1_pts = 10.0 * max(0.0, min(1.0, (s1_f1 - 0.70) / 0.20))

    s2_std = bench["stage2"]["hidden_standard_macro_f1"]
    s2_std_pts = bench["stage2"]["standard_points_max6"]
    if s2_std_pts is None and s2_std is not None:
        s2_std_pts = 6.0 * max(0.0, min(1.0, (s2_std - 0.80) / 0.15))

    s2_rob = bench["stage2"]["hidden_robust_macro_f1"]
    s2_rob_pts = bench["stage2"]["robust_points_max6"]
    if s2_rob_pts is None and s2_rob is not None:
        s2_rob_pts = 6.0 * max(0.0, min(1.0, (s2_rob - 0.60) / 0.25))

    s3_map = bench["stage3"]["hidden_map50"]
    s3_pts = bench["stage3"]["rubric_points_max12"]
    if s3_pts is None and s3_map is not None:
        s3_pts = 12.0 * max(0.0, min(1.0, (s3_map - 0.65) / 0.25))

    return {
        "s1_f1": s1_f1, "s1_pts": s1_pts,
        "s2_std": s2_std, "s2_std_pts": s2_std_pts,
        "s2_rob": s2_rob, "s2_rob_pts": s2_rob_pts,
        "s3_map": s3_map, "s3_pts": s3_pts,
    }


def build_final_report(s1_trad: Dict, s1_cnn: Dict, s2: Dict, s3: Dict, bench: Dict) -> str:
    r = compute_rubric_estimates(bench)
    s1_sel_id = s1_cnn["selected_model_id"]
    s1_sel = s1_cnn["experiments"][s1_sel_id]
    s1_e1 = s1_cnn["experiments"]["E1"]
    s1_svm = s1_trad["models"]["PCA_SVM"]
    s1_knn = s1_trad["models"]["PCA_KNN"]

    s2_exps = s2["experiments"]
    s2_sel_id = s2["selected_model_id"]
    s2_sel = s2_exps[s2_sel_id]

    s3_dets = s3["detectors"]
    s3_sel_id = s3["selected_model_id"]
    s3_fm = s3["final_val_metrics"]

    f1_dir = ROOT / "workspace_outputs" / "stage1" / "figs"
    f2_dir = ROOT / "workspace_outputs" / "stage2" / "figs"
    f3_dir = ROOT / "workspace_outputs" / "stage3" / "figs"

    diag_path = ROOT / "workspace_outputs/stage1/generalization_diag.json"
    s1_diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.is_file() else {}

    s1_rows = []
    for eid_key, name, acc, prec, rec, mf1, gap in [
        ("KNN", "PCA + KNN (k=5)", s1_knn["accuracy"], s1_knn["macro_precision"], s1_knn["macro_recall"], s1_knn["macro_f1"], s1_knn["train_macro_f1"] - s1_knn["macro_f1"]),
        ("SVM", "PCA + SVM (RBF)", s1_svm["accuracy"], s1_svm["macro_precision"], s1_svm["macro_recall"], s1_svm["macro_f1"], s1_svm["train_macro_f1"] - s1_svm["macro_f1"]),
        ("E1", "E1: Baseline CNN", s1_e1["best_val"]["accuracy"], s1_e1["best_val"]["macro_precision"], s1_e1["best_val"]["macro_recall"], s1_e1["best_val"]["macro_f1"], s1_e1["final_epoch_diagnostic"]["overfit_gap_f1"]),
        ("E2", "E2: +BN (Method A)", s1_cnn["experiments"]["E2"]["best_val"]["accuracy"], s1_cnn["experiments"]["E2"]["best_val"]["macro_precision"], s1_cnn["experiments"]["E2"]["best_val"]["macro_recall"], s1_cnn["experiments"]["E2"]["best_val"]["macro_f1"], s1_cnn["experiments"]["E2"]["final_epoch_diagnostic"]["overfit_gap_f1"]),
        ("E3", "E3: +Aug (Method B)", s1_cnn["experiments"]["E3"]["best_val"]["accuracy"], s1_cnn["experiments"]["E3"]["best_val"]["macro_precision"], s1_cnn["experiments"]["E3"]["best_val"]["macro_recall"], s1_cnn["experiments"]["E3"]["best_val"]["macro_f1"], s1_cnn["experiments"]["E3"]["final_epoch_diagnostic"]["overfit_gap_f1"]),
        ("E4", "E4: +BN+Aug (A+B)", s1_cnn["experiments"]["E4"]["best_val"]["accuracy"], s1_cnn["experiments"]["E4"]["best_val"]["macro_precision"], s1_cnn["experiments"]["E4"]["best_val"]["macro_recall"], s1_cnn["experiments"]["E4"]["best_val"]["macro_f1"], s1_cnn["experiments"]["E4"]["final_epoch_diagnostic"]["overfit_gap_f1"]),
        ("E5", "E5: +BN+Aug+Reg (C)", s1_cnn["experiments"]["E5"]["best_val"]["accuracy"], s1_cnn["experiments"]["E5"]["best_val"]["macro_precision"], s1_cnn["experiments"]["E5"]["best_val"]["macro_recall"], s1_cnn["experiments"]["E5"]["best_val"]["macro_f1"], s1_cnn["experiments"]["E5"]["final_epoch_diagnostic"]["overfit_gap_f1"]),
    ]:
        hl = " class='highlight'" if eid_key == s1_sel_id else ""
        dg = s1_diag.get(eid_key, {})
        pert_str = f"{dg['perturbed_val_f1']:.4f}" if "perturbed_val_f1" in dg else "-"
        vloss_str = f"{dg['best_val_loss']:.5f}" if "best_val_loss" in dg else "-"
        s1_rows.append(f"<tr{hl}><td class='left'>{name}</td><td>{acc:.4f}</td><td><b>{mf1:.4f}</b></td><td>{vloss_str}</td><td><b>{pert_str}</b></td><td>{gap:+.4f}</td></tr>")

    s2_rows = []
    for eid in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        e = s2_exps[eid]
        hl = " class='highlight'" if eid == s2_sel_id else ""
        s2_rows.append(
            f"<tr{hl}><td>{eid}</td><td class='left'>{e['architecture']}</td><td class='left'>{e['strategy']}</td>"
            f"<td>{e['clean_val']['macro_f1']:.4f}</td><td>{e['robustness']['lighting']['macro_f1']:.4f}</td>"
            f"<td>{e['robustness']['optical']['macro_f1']:.4f}</td><td>{e['robustness']['sensor_compression']['macro_f1']:.4f}</td>"
            f"<td><b>{e['robustness']['robust_macro_f1']:.4f}</b></td>"
            f"<td>{e['checkpoint_size_mb']:.1f} MB</td></tr>"
        )

    s3_rows = []
    for did in ["yolo11s", "yolo11m", "faster_rcnn"]:
        d = s3_dets[did]
        hl = " class='highlight'" if did == s3_sel_id else ""
        s3_rows.append(
            f"<tr{hl}><td class='left'>{html.escape(d['detector'])}</td>"
            f"<td><b>{d['eval_conf001']['mAP50']:.4f}</b></td><td>{d['eval_conf025']['mAP50']:.4f}</td>"
            f"<td>{d['eval_conf025']['precision']:.4f}</td><td>{d['eval_conf025']['recall']:.4f}</td>"
            f"<td>{d['file_size_mb']:.1f} MB</td></tr>"
        )

    body = f"""
<div class="report-header">
  <div class="course-tag">影像識別期末專題總結報告｜FactoryVision Challenge</div>
  <h1>智慧工廠鋼材表面瑕疵辨識與定位系統 — 期末報告</h1>
  <p class="author-line">組別：Team21｜隨機種子：42</p>
</div>

<table class="summary-strip">
  <tr>
    <td><span class="lbl">Stage 1 最終模型</span><span class="val">{s1_sel_id} (Val F1={s1_sel['best_val']['macro_f1']:.4f})</span></td>
    <td><span class="lbl">Stage 2 最終模型</span><span class="val">{s2_sel_id} (Robust={s2_sel['robustness']['robust_macro_f1']:.3f})</span></td>
    <td><span class="lbl">Stage 3 最終模型</span><span class="val">Faster R-CNN (mAP={s3_fm['mAP50']:.4f})</span></td>
    <td><span class="lbl">資料切分規範</span><span class="val">Master Split / Seed=42</span></td>
  </tr>
</table>

<h2>1. Problem &amp; Dataset（問題背景與資料集特性）</h2>
<p>本專題模擬智慧製造鋼材產線的表面瑕疵檢測問題，系統除了要分辨六類瑕疵（<code>crazing</code> 龜裂、<code>inclusion</code> 夾雜、<code>patches</code> 斑塊、<code>pitted_surface</code> 點蝕、<code>rolled_in_scale</code> 氧化鐵皮壓入、<code>scratches</code> 刮痕），還需進一步標出瑕疵座標，並考量實際工廠的光照變化、模糊、壓縮雜訊與運算成本。</p>
<ul>
  <li><b>資料切分與規範</b>：課程提供 <code>FactoryVision_STUDENT_PUBLIC</code>，Classification 與 Detection 共用相同的 Master Split：<b>Train = 1,260</b>（每類 210 張）、<b>Validation = 270</b>（每類 45 張）。所有實驗固定 <code>seed = 42</code>，禁用外部微調權重，Validation 僅用來挑模型與調參。</li>
  <li><b>三階段分工</b>：<b>Stage 1</b> 先用傳統特徵與自己刻的 CNN 驗證基本方法；<b>Stage 2</b> 系統性比較兩種預訓練骨幹在不同微調策略下的抗干擾表現；<b>Stage 3</b> 則進入 Bounding-Box 偵測，比較單階段與雙階段模型的定位準度與模型複雜度。</li>
  <li><b>我們觀察到的資料特性（200×200 影像）</b>：統計標註檔後發現六類瑕疵大小落差很大——<code>inclusion</code> 中位寬只有 <code>23 px</code>、<code>scratches</code> 細長、<code>pitted_surface</code> 與 <code>crazing</code> 則常佔滿整張圖；另外全資料集中共有 <b>104 張圖同時存在一種以上的瑕疵</b>，這也是為什麼最後實務上必須走向 Object Detection。</li>
</ul>

<h2>2. Stage 1：Baseline 與 Improved CNN</h2>
<p>先看傳統方法：將灰階圖縮成 <code>64×64</code> 做標準化與 <code>PCA(0.95)</code>（保留 33 維主成分，解釋 95.03% 變異）後，<code>PCA + KNN</code> 只拿到 Val Macro F1 = <b>{s1_knn['macro_f1']:.4f}</b>，<code>PCA + SVM (RBF)</code> 提升到 <b>{s1_svm['macro_f1']:.4f}</b> 但對細長紋理仍常誤判。接著我們自己設計四層卷積的 <code>SteelDefectCNN</code>（{s1_sel['params']['total_params']:,} 參數，從頭訓練 80 epochs），做 E1–E5 消融實驗。有趣的是 E1–E5 在潔淨驗證集最後都達到 1.0000，因此我們另外做了 ±15°／±20% 亮度對比的偏移診斷（Mild-Shift Probe），才明顯看出 <code>E4 (BN + DataAug)</code> 的抗偏移能力最好（<code>1.0000</code> vs. E1 <code>0.2907</code>）：</p>
<table>
  <thead>
    <tr><th class="left">模型 / 實驗組別</th><th>Val Acc</th><th>Val Macro F1</th><th>Best Val Loss</th><th>Mild-Shift Gen F1</th><th>Final Overfit Gap (F1)</th></tr>
  </thead>
  <tbody>{''.join(s1_rows)}</tbody>
</table>
{img_tag(f1_dir / "fig2_stage1_comparison_bar.png", "圖 1：Stage 1 傳統方法與自建 CNN E1–E5 消融實驗對照。", max_height_px=250)}

<h2>3. Stage 2：Transfer Learning 與 Robustness 挑戰</h2>
<p>我們執行 <b>DenseNet121</b> 與 <b>MobileNetV3-Large</b> 兩架構家族搭配三種策略（<code>Feature Extraction</code>、<code>Partial Fine-tuning</code>、<code>Full Fine-tuning</code>）之 2×3 公平對照，並導入本地合成之 <code>Lighting</code>、<code>Optical Blur</code> 與 <code>Sensor + JPEG</code> 壓力測試評估環境強健性：</p>
<table>
  <thead>
    <tr><th>ID</th><th class="left">Architecture</th><th class="left">Strategy</th><th>Clean F1</th><th>Lighting</th><th>Optical</th><th>Sensor/JPEG</th><th>Local Robust F1</th><th>Size</th></tr>
  </thead>
  <tbody>{''.join(s2_rows)}</tbody>
</table>
{img_tag(f2_dir / "fig2_stage2_robustness_bar.png", "圖 2：Stage 2 六組遷移學習實驗之 Clean 與三項本地 Robustness 測試結果。", max_height_px=275)}

<h2>4. Stage 3：Object Detection 瑕疵定位與工程比較</h2>
<p>我們於相同切分下訓練並以 Slide-31 Pascal VOC 2010+ 定義統一評估單階段 Anchor-free（<code>YOLO11s</code>、<code>YOLO11m</code>）與雙階段 Anchor-based（<code>Faster R-CNN ResNet50-FPN-v2</code>）檢測器：</p>
<table>
  <thead>
    <tr><th class="left">Detector</th><th>mAP@0.5 (c=0.01)</th><th>mAP@0.5 (c=0.25)</th><th>Precision (c=0.25)</th><th>Recall (c=0.25)</th><th>Size</th></tr>
  </thead>
  <tbody>{''.join(s3_rows)}</tbody>
</table>
<p>最終選定 <b>{html.escape(s3['selected_detector'])}</b>（提交設定：<code>confidence_threshold={s3['confidence_threshold']}</code>、<code>nms_iou_threshold={s3['nms_iou_threshold']}</code>），於驗證集達成 <b>mAP@0.5 = {s3_fm['mAP50']:.4f}</b>（操作門檻 <code>conf=0.25</code> 下 Precision = <b>{s3_dets[s3_sel_id]['eval_conf025']['precision']:.4f}</b>、Recall = <b>{s3_dets[s3_sel_id]['eval_conf025']['recall']:.4f}</b>），權重檔案大小為 <b>{s3['checkpoint_size_mb']:.1f} MB</b>。</p>
{img_tag(f3_dir / "fig1_stage3_detector_comparison.png", "圖 3：Stage 3 三種目標檢測器之逐類 AP@0.5 與整體 mAP@0.5 比較。", max_height_px=265)}

<div class="page-break"></div>

<h2>5. Failure Analysis（跨 Stage 主要錯誤案例與根本成因）</h2>
<ul>
  <li><b>1. 跨類別紋理混淆與資料增強消融診斷（Stage 1）</b>：<code>PCA + SVM</code> 在細長或點狀紋理上嚴重失誤，<code>inclusion</code>（F1 = {s1_svm['per_class']['inclusion']['f1']:.2f}）與 <code>scratches</code>（F1 = {s1_svm['per_class']['scratches']['f1']:.2f}）常相互混淆；自建 CNN 雖在潔淨驗證集達 1.0000，但在 ±15° / ±20% 亮度對比偏移診斷中，無增強基準 <code>E1</code> 急劇崩跌至 <b>0.2907</b>、僅加 BN 之 <code>E2</code> 僅 <b>0.5573</b>，直到結合 BN 與資料增強（<code>E4</code>）才維持 <b>1.0000</b> 抗偏移穩定度。</li>
  <li><b>2. 光照與壓縮雜訊對 Frozen Feature Extraction 的衝擊（Stage 2）</b>：僅訓練分類頭時，預訓練淺層濾波器對鋼材高頻壓縮失真脆弱，<code>DenseNet121 FE (E1)</code> 於 <code>Sensor/JPEG</code> 跌至 <b>{s2_exps['E1']['robustness']['sensor_compression']['macro_f1']:.4f}</b>（Overall Robust = <b>{s2_exps['E1']['robustness']['robust_macro_f1']:.4f}</b>）；經統一 Robustness-Aware 增強與全層微調後，<code>E6 (MobileNetV3-Large Full FT)</code> 達成 Clean <b>{s2_sel['clean_val']['macro_f1']:.4f}</b> 與 Local Robust <b>{s2_sel['robustness']['robust_macro_f1']:.4f}</b>（全測試 810 張僅 5 筆誤判）。</li>
  <li><b>3. 空間定位難類、小目標漏檢與多標籤結構瓶頸（Stage 3）</b>：即便最強檢測器 <code>Faster R-CNN R50-FPN-v2</code>，於瀰漫低對比之 <code>crazing</code>（AP = <b>{s3_fm['per_class']['crazing']['ap50']:.4f}</b>）與 <code>rolled_in_scale</code>（AP = <b>{s3_fm['per_class']['rolled_in_scale']['ap50']:.4f}</b>）仍常見 Poor Localization（<code>IoU &lt; 0.5</code>），微小型 <code>inclusion</code>（中位寬 23 px）偶發 FN；此外資料集共含 <b>104 張複合瑕疵影像</b>，Stage 1–2 單標籤模型結構上無法輸出第二類別，唯有 Stage 3 檢測器能精準解析複合瑕疵空間分布。</li>
</ul>
{img_tag(f3_dir / "fig5_stage3_failure_cases.png", "圖 4：Stage 3 代表性 False Positive / False Negative / Poor Localization 錯誤案例視覺化。", max_height_px=470)}

<h2>6. Deployment Decision（跨階段量化權衡與工廠部署決策）</h2>
<table>
  <thead>
    <tr><th class="left">候選系統方案</th><th>任務能力</th><th>核心指標 (Val)</th><th>抗干擾性</th><th>模型大小</th><th class="left">產線工程評價</th></tr>
  </thead>
  <tbody>
    <tr>
      <td class="left">Traditional: PCA + SVM</td><td>僅單標籤分類</td><td>F1 = {s1_svm['macro_f1']:.4f}</td><td>極低</td><td>&lt; 1 MB</td>
      <td class="left">僅適合作為極簡基準，精度不足以部署</td>
    </tr>
    <tr>
      <td class="left">Stage 1: {s1_sel_id} (From Scratch)</td><td>僅單標籤分類</td><td>F1 = {s1_sel['best_val']['macro_f1']:.4f}</td><td>中低</td><td>{s1_cnn['checkpoint_size_mb']:.2f} MB</td>
      <td class="left">極輕量架構，適合低成本粗篩，但缺乏空間定位能力</td>
    </tr>
    <tr>
      <td class="left">Stage 2: {s2_sel_id} (MobileNetV3-L Full FT)</td><td>單標籤分類 + 抗干擾</td><td>Clean={s2_sel['clean_val']['macro_f1']:.4f}<br>Robust={s2_sel['robustness']['robust_macro_f1']:.4f}</td><td><b>極高</b></td><td>{s2_sel['checkpoint_size_mb']:.2f} MB</td>
      <td class="left">強大抗光照/模糊/壓縮與輕量特性，適合惡劣成像複檢與邊緣機台</td>
    </tr>
    <tr>
      <td class="left">Stage 3 Tier-Fast: YOLO11m</td><td>分類 + 定位 + 多目標</td><td>mAP@0.5 = {s3_dets['yolo11m']['eval_conf001']['mAP50']:.4f}</td><td>中高</td><td>{s3_dets['yolo11m']['file_size_mb']:.1f} MB</td>
      <td class="left"><b>輕量定位首選</b>：參數與檔案顯著較小，適合資源受限的主產線工位</td>
    </tr>
    <tr class="highlight">
      <td class="left">Stage 3 Tier-Precision: {html.escape(s3['selected_detector'])}</td><td><b>最高精度定位 + 多目標</b></td><td>mAP@0.5 = <b>{s3_fm['mAP50']:.4f}</b></td><td>高</td><td>{s3['checkpoint_size_mb']:.1f} MB</td>
      <td class="left"><b>高精品檢首選</b>：<code>crazing</code> 與 <code>rolled_in_scale</code> 表現最佳，適合品管複檢工位</td>
    </tr>
  </tbody>
</table>

<h2>7. Conclusion（最終 FactoryVision 系統部署建議）</h2>
<div class="note-box">
  <b>最終系統決策：依照產線工位需求採用「YOLO11m（輕量定位主線）＋ Faster R-CNN R50-FPN-v2（高精複檢站）＋ MobileNetV3-Large E6（惡劣成像備援）」</b>
  <ol>
    <li><b>高精確度定位主模型</b>：若目標是最大化偵測準確度與召回率（例如品管站或離線複檢），我會選擇 <b>Stage 3 Faster R-CNN ResNet50-FPN-v2</b>（驗證集 <code>mAP@0.5 = {s3_fm['mAP50']:.4f}</code>，檔案 {s3['checkpoint_size_mb']:.1f} MB），它在難檢的 <code>crazing</code> 與 <code>rolled_in_scale</code> 上表現明顯優於單階段模型，也能處理複合瑕疵影像。</li>
    <li><b>輕量多目標定位工位</b>：若產線邊緣設備受限於模型容量與記憶體，實務上可採用同實驗證過的 <b>YOLO11m</b>（<code>mAP@0.5 = {s3_dets['yolo11m']['eval_conf001']['mAP50']:.4f}</code>、檔案僅 {s3_dets['yolo11m']['file_size_mb']:.1f} MB）做第一線定位篩檢，有疑似框再送交 Faster R-CNN 複核。</li>
    <li><b>強干擾環境與低資源分類備援</b>：當鏡頭油霧／強光閃爍嚴重或僅需整張圖分類把關時，改以 <b>Stage 2 E6（MobileNetV3-Large Full Fine-tuning）</b>（僅 <b>{s2_sel['checkpoint_size_mb']:.2f} MB</b>、Local Robust Macro F1 = <b>{s2_sel['robustness']['robust_macro_f1']:.4f}</b>）作為備援；產線告警門檻建議設於 <code>conf = 0.25–0.30</code>，兼顧低誤報與高召回。</li>
  </ol>
</div>
"""
    return wrap_doc("Team21 FactoryVision Final Report", body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FactoryVision Stage 1-3 & Final PDF Reports")
    parser.add_argument("--only_final", action="store_true", help="Only rebuild Team21_Final_Report.pdf")
    args = parser.parse_args()

    s1_trad = json.loads((ROOT / "workspace_outputs/stage1/traditional_results.json").read_text(encoding="utf-8"))
    s1_cnn = json.loads((ROOT / "workspace_outputs/stage1/cnn_ablation_results.json").read_text(encoding="utf-8"))
    s2 = json.loads((ROOT / "workspace_outputs/stage2/stage2_results.json").read_text(encoding="utf-8"))
    s3 = json.loads((ROOT / "workspace_outputs/stage3/stage3_results.json").read_text(encoding="utf-8"))
    bench = json.loads((ROOT / "scripts/benchmark_scores.json").read_text(encoding="utf-8"))

    html_dir = ROOT / "workspace_outputs" / "final_report" / "html"

    if not args.only_final:
        print("[Reports] Building stage1_report.pdf...")
        render_html_to_pdf(
            build_stage1_report(s1_trad, s1_cnn, bench),
            html_dir / "stage1_report.html",
            ROOT / "Team21_Stage1/stage1_report.pdf",
        )
        print("[Reports] Building stage2_report.pdf...")
        render_html_to_pdf(
            build_stage2_report(s2, bench),
            html_dir / "stage2_report.html",
            ROOT / "Team21_Stage2/stage2_report.pdf",
        )
        print("[Reports] Building stage3_report.pdf...")
        render_html_to_pdf(
            build_stage3_report(s3, bench),
            html_dir / "stage3_report.html",
            ROOT / "Team21_Stage3/stage3_report.pdf",
        )

    print("[Reports] Building Team21_Final_Report.pdf...")
    render_html_to_pdf(
        build_final_report(s1_trad, s1_cnn, s2, s3, bench),
        html_dir / "Team21_Final_Report.html",
        ROOT / "Team21_Final_Report.pdf",
    )
    print("[Reports] All requested PDF reports built successfully!")


if __name__ == "__main__":
    main()

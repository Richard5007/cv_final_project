# FactoryVision Challenge — 智慧工廠鋼材表面瑕疵辨識與定位系統 (Team21)

「影像識別」期末專題（`seed = 42`）。  
本專案依 [FactoryVision_Final_Project.pdf](./FactoryVision_Final_Project.pdf) 規格完成 Stage 1–3 模型訓練、消融與強健性實驗、離線推論腳本與報告。

## 實驗結果總覽（Validation Split, N=270）

| Stage | 任務 | 最終選定模型 | 核心指標 | 參數量 / 模型大小 |
| :--- | :--- | :--- | :--- | :---: |
| **Stage 1** | From-Scratch CNN 分類 | `SteelDefectCNN_E4` (BN + DataAug) | Clean Macro F1 = `1.0000` / Mild-Shift = `1.0000` | 421,574 / 1.62 MB |
| **Stage 2** | Transfer Learning + Robustness | `MobileNetV3-Large` (Full Fine-tuning, `E6`) | Clean F1 = `1.0000` / Local Robust F1 = `0.9938` | 4,209,718 / 16.26 MB |
| **Stage 3** | Object Detection 定位 | `Faster R-CNN ResNet50-FPN-v2` | Slide-31 VOC mAP@0.5 = `0.7866` (`c=0.01`) | 43,275,906 / 165.47 MB |

> 註：Stage 3 另與 `YOLO11s`（mAP@0.5 = `0.7413`, 9.4M / 18.3 MB）及 `YOLO11m`（mAP@0.5 = `0.7603`, 20.1M / 38.7 MB）進行同切分比較；依產線資源需求，輕量工位可選用 `YOLO11m`，高精度精檢則選用 `Faster R-CNN R50-FPN-v2`。

## 目錄結構

- `Team21_Stage1/`：Stage 1 程式碼（`src/`）、推論腳本 `stage1_inference.py`、設定與 [stage1_report.pdf](./Team21_Stage1/stage1_report.pdf)
- `Team21_Stage2/`：Stage 2 程式碼（`src/`）、推論腳本 `stage2_inference.py`、設定與 [stage2_report.pdf](./Team21_Stage2/stage2_report.pdf)
- `Team21_Stage3/`：Stage 3 程式碼（`src/`）、偵測腳本 `stage3_detect.py`、設定與 [stage3_report.pdf](./Team21_Stage3/stage3_report.pdf)
- `scripts/`：繪圖、報告產生（`build_reports.py` / `rebuild_final_report.py`）、打包與提交格式驗證器（`validate_submission.py`）
- `workspace_outputs/`：各階段實驗 JSON 數據與分析圖表
- [Team21_Final_Report.pdf](./Team21_Final_Report.pdf)：整合三階段實驗與部署建議之期末總結報告

> 因 GitHub 單檔 100 MB 上限，原始資料集（`data/`）、大型權重檔（`*.pt` / `*.pth`）與繳交用 `*.zip` 已由 `.gitignore` 排除。

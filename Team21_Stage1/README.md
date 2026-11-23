# FactoryVision Stage 1 — Baseline + Improved CNN (Team21)

## Environment & Versions
- **Python**: 3.13.2
- **PyTorch**: 2.7.1+cu118
- **Torchvision**: 0.22.1+cu118
- **scikit-learn**: 1.9.1
- **Pillow**: 11.0.0
- **NumPy**: 2.2.6

## Final Model Summary
- **Team ID**: Team21
- **Final Model Name**: `SteelDefectCNN_E4` (Baseline CNN + Batch Normalization + Data Augmentation)
- **Parameters**: 421,990 total / 421,990 trainable
- **Checkpoint File**: `stage1_model.pth` (1.62 MB)
- **Improvement Methods**:
  - `Batch Normalization` (`BatchNorm2d` inside each of the 4 convolutional blocks)
  - `Data Augmentation` (`RandomHorizontalFlip(0.5)`, `RandomVerticalFlip(0.5)`, `RandomRotation(15)`, `ColorJitter(brightness=0.2, contrast=0.2)`)
- **Pretrained Source**: None (Strictly trained from scratch on FactoryVision Classification Public Train set, `seed=42`)
- **Input Size**: `1 x 200 x 200` (1-channel Grayscale)
- **Preprocessing**:
  1. Convert image to grayscale (`PIL.Image.convert("L")`)
  2. Resize to `(200, 200)`
  3. Convert to Tensor (`[0.0, 1.0]`)
  4. Normalize with training-set grayscale statistics: `mean=[0.5071]`, `std=[0.2132]`

## Benchmark Inference Command (TA Standard)
```bash
python stage1_inference.py --input_dir <HIDDEN_DIR> --output_csv predictions.csv
```
- Runs 100% offline using local `config.json` and `stage1_model.pth` resolved relative to `stage1_inference.py`.
- Uses deterministic `model.eval()` + `torch.no_grad()` without TTA or ensemble.
- Outputs 2-column CSV: `image_id,prediction`.

## Training & Evaluation Reproduction
```bash
python src/train.py --data_dir ../data/STUDENT_PUBLIC/classification --output_dir ../workspace_outputs/stage1 --submission_dir .
```

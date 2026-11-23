# FactoryVision Stage 2 — Transfer Learning & Robustness Challenge (Team21)

## Environment & Versions
- **Python**: 3.13.2
- **PyTorch**: 2.7.1+cu118
- **Torchvision**: 0.22.1+cu118
- **scikit-learn**: 1.9.1
- **OpenCV**: 4.13.0
- **Pillow**: 11.0.0
- **NumPy**: 2.2.6

## Final Selected Model Summary
- **Team ID**: Team21
- **Selected Experiment**: `E6` (`mobilenet_v3_large` × `full_finetuning`)
- **Pretrained Source**: `torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2`
- **Checkpoint File**: `stage2_model.pth` (16.26 MB)
- **Parameters**: 4,209,718 total / 4,209,718 trainable
- **Validation Performance**: Clean Macro F1 = `1.0000`, Local Robust Macro F1 = `0.9938`
- **Unified Preprocessing**:
  1. Convert image to RGB (`PIL.Image.convert("RGB")`)
  2. Resize to `(224, 224)`
  3. Convert to Tensor (`[0.0, 1.0]`)
  4. ImageNet Normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

## Experimental Matrix (2 Architectures x 3 Strategies)
- **Architectures**: `densenet121` (`IMAGENET1K_V1`), `mobilenet_v3_large` (`IMAGENET1K_V2`)
- **Strategies**: `feature_extraction`, `partial_finetuning`, `full_finetuning`

## Benchmark Inference Command (TA Standard)
```bash
python stage2_inference.py --input_dir <DIR> --output_csv predictions.csv
```
- Uses the exact same `stage2_inference.py`, `stage2_model.pth`, and preprocessing across Standard and all 3 Robustness benchmarks (`lighting`, `optical`, `sensor_compression`).
- Runs 100% offline (`pretrained=False`), with zero path-dependent branching.

## Training & Evaluation Reproduction
```bash
python src/train.py --data_dir ../data/STUDENT_PUBLIC/classification --output_dir ../workspace_outputs/stage2 --submission_dir .
```

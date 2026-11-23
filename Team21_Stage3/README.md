# FactoryVision Stage 3 — Object Detection (Team21)

## Environment & Versions
- **Python**: 3.13.2
- **PyTorch**: 2.7.1+cu118
- **Torchvision**: 0.22.1+cu118
- **Ultralytics**: 8.4.216
- **OpenCV**: 4.13.0
- **Pillow**: 11.0.0
- **NumPy**: 2.2.6

## Final Selected Model Summary
- **Team ID**: Team21
- **Selected Detector**: `Faster R-CNN ResNet50-FPN-v2` (Two-stage Anchor-based RPN + RoIAlign)
- **Pretrained Source**: `torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`
- **Checkpoint File**: `stage3_model.pt` (165.47 MB)
- **Input Size**: Dynamic aspect-preserving rescale (`min_size=600, max_size=800`)
- **Thresholds**:
  - `confidence_threshold`: `0.01` (preserves full Pascal VOC PR-curve tail; average 12.26 detections/image, max <= 100; recommended factory visual alert threshold = `0.25–0.30`)
  - `nms_iou_threshold`: `0.50`
  - `bbox_format`: `xyxy_pixel`

## Experimental Comparison (Slide-31 Pascal VOC 2010+ mAP@0.5)
- **Detectors Evaluated**: `YOLO11s`, `YOLO11m`, and `Faster R-CNN ResNet50-FPN-v2`
- **Class Order**: `["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]`

## Benchmark Inference Command (TA Standard)
```bash
python stage3_detect.py --input_dir <HIDDEN_DIR> --output_json detections.json
```
- Runs 100% offline (`pretrained=False`) using local `config.json` and `stage3_model.pt` resolved relative to `stage3_detect.py`.
- `confidence_threshold` and `nms_iou_threshold` are read directly from `config.json`.

## Training & Evaluation Reproduction
```bash
python src/train.py --det_root ../data/STUDENT_PUBLIC/detection --yolo_yaml ../data/yolo_dataset/data.yaml --output_dir ../workspace_outputs/stage3 --submission_dir .
```

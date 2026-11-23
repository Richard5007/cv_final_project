"""Stage 3 Object Detector definitions & offline checkpoint loaders (YOLO11 & Faster R-CNN)."""

from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

CLASS_ORDER = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


def build_faster_rcnn(
    num_defect_classes: int = 6,
    pretrained: bool = False,
    min_size: int = 600,
    max_size: int = 800,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.50,
    box_detections_per_img: int = 100,
):
    """Build Faster R-CNN ResNet50-FPN-v2 detector.

    Call with pretrained=False during offline inference so weights are loaded purely
    from the local checkpoint without attempting any network download.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    weights_backbone = None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=weights_backbone,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_defect_classes + 1)
    return model


def clamp_and_filter_boxes(
    boxes_xyxy: List[List[float]],
    scores: List[float],
    labels: List[int],
    width: int,
    height: int,
    conf_thresh: float,
    max_det: int = 100,
    label_offset: int = 0,
) -> List[Dict]:
    """Convert raw predictions to Slide-30 compliant pixel xyxy detection dicts."""
    dets: List[Dict] = []
    for box, score, cls_idx in zip(boxes_xyxy, scores, labels):
        conf = float(score)
        if conf < conf_thresh or conf != conf:  # filters NaN
            continue
        conf = max(0.0, min(1.0, conf))
        c_idx = int(cls_idx) - label_offset
        if c_idx < 0 or c_idx >= len(CLASS_ORDER):
            continue
        x1 = max(0.0, min(float(width), float(box[0])))
        y1 = max(0.0, min(float(height), float(box[1])))
        x2 = max(0.0, min(float(width), float(box[2])))
        y2 = max(0.0, min(float(height), float(box[3])))
        if x2 <= x1 + 1e-3 or y2 <= y1 + 1e-3:
            continue
        dets.append(
            {
                "class": CLASS_ORDER[c_idx],
                "confidence": round(conf, 6),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )

    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets[:max_det]

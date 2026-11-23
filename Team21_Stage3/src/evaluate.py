"""Exact Slide-31 Pascal VOC 2010+ mAP@0.5 evaluator and schema validator for Stage 3."""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CLASS_ORDER = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


def parse_voc_annotations(voc_dir: Path) -> Dict[str, Dict]:
    """Parse all VOC XML ground truth files in voc_dir keyed by image_id (filename stem)."""
    gt_by_image: Dict[str, Dict] = {}
    for xml_path in sorted(voc_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_el = root.find("size")
        width = int(float(size_el.find("width").text))
        height = int(float(size_el.find("height").text))
        boxes = []
        for obj in root.findall("object"):
            raw_name = obj.find("name").text.strip()
            cls_name = "rolled_in_scale" if raw_name == "rolled-in_scale" else raw_name
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
            boxes.append({"class": cls_name, "bbox": [xmin, ymin, xmax, ymax]})
        gt_by_image[xml_path.stem] = {
            "width": width,
            "height": height,
            "boxes": boxes,
        }
    return gt_by_image


def validate_detections_schema(
    predictions: List[Dict],
    gt_by_image: Dict[str, Dict],
) -> List[str]:
    """Verify predictions strictly conform to Slide 30 JSON specification."""
    errors: List[str] = []
    if not isinstance(predictions, list):
        return ["Top-level JSON must be a list."]

    expected_ids = set(gt_by_image.keys())
    seen_ids = set()
    valid_classes = set(CLASS_ORDER)

    for idx, item in enumerate(predictions):
        if not isinstance(item, dict):
            errors.append(f"Item {idx} is not an object.")
            continue
        img_id = item.get("image_id")
        if img_id not in expected_ids:
            errors.append(f"Unexpected or missing image_id: {img_id}")
            continue
        if img_id in seen_ids:
            errors.append(f"Duplicate image_id: {img_id}")
            continue
        seen_ids.add(img_id)

        dets = item.get("detections")
        if not isinstance(dets, list):
            errors.append(f"{img_id}: detections must be a list.")
            continue
        if len(dets) > 100:
            errors.append(f"{img_id}: detections count {len(dets)} exceeds limit 100.")

        w = gt_by_image[img_id]["width"]
        h = gt_by_image[img_id]["height"]
        for d_i, det in enumerate(dets):
            cls_name = det.get("class")
            if cls_name not in valid_classes:
                errors.append(f"{img_id}[{d_i}]: invalid class {cls_name}")
            conf = det.get("confidence")
            if not isinstance(conf, (int, float)) or math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
                errors.append(f"{img_id}[{d_i}]: invalid confidence {conf}")
            bbox = det.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"{img_id}[{d_i}]: invalid bbox {bbox}")
                continue
            xmin, ymin, xmax, ymax = bbox
            if not all(isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v) for v in bbox):
                errors.append(f"{img_id}[{d_i}]: non-finite bbox {bbox}")
                continue
            if not (0.0 <= xmin < xmax <= w + 1e-4 and 0.0 <= ymin < ymax <= h + 1e-4):
                errors.append(f"{img_id}[{d_i}]: bbox out of bounds {bbox} for ({w}, {h})")

    missing = sorted(expected_ids - seen_ids)
    if missing:
        errors.append(f"Missing {len(missing)} image_ids, e.g. {missing[:5]}")
    return errors


def compute_iou(box_a: List[float], boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between one [xmin, ymin, xmax, ymax] box and Nx4 array."""
    if len(boxes_b) == 0:
        return np.zeros((0,), dtype=np.float64)
    ixmin = np.maximum(box_a[0], boxes_b[:, 0])
    iymin = np.maximum(box_a[1], boxes_b[:, 1])
    ixmax = np.minimum(box_a[2], boxes_b[:, 2])
    iymax = np.minimum(box_a[3], boxes_b[:, 3])
    iw = np.maximum(ixmax - ixmin, 0.0)
    ih = np.maximum(iymax - iymin, 0.0)
    inter = iw * ih
    area_a = max(box_a[2] - box_a[0], 0.0) * max(box_a[3] - box_a[1], 0.0)
    area_b = np.maximum(boxes_b[:, 2] - boxes_b[:, 0], 0.0) * np.maximum(boxes_b[:, 3] - boxes_b[:, 1], 0.0)
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / union, 0.0)


def voc_ap_2010_plus(rec: np.ndarray, prec: np.ndarray) -> float:
    """Pascal VOC 2010+ all-points interpolated Average Precision."""
    if len(rec) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def evaluate_detections_voc(
    predictions: List[Dict],
    gt_by_image: Dict[str, Dict],
    iou_threshold: float = 0.5,
    eval_conf_threshold: Optional[float] = None,
) -> Dict:
    """Evaluate detection JSON using Slide-31 exact Pascal VOC 2010+ mAP@0.5 definition."""
    per_class_results: Dict[str, Dict] = {}
    aps: List[float] = []
    total_tp = 0
    total_fp = 0
    total_gt_all = 0

    for cls_name in CLASS_ORDER:
        # Collect all GTs for this class keyed by image_id
        cls_gts: Dict[str, Dict] = {}
        n_gt = 0
        for img_id, info in gt_by_image.items():
            boxes = [b["bbox"] for b in info["boxes"] if b["class"] == cls_name]
            if boxes:
                cls_gts[img_id] = {
                    "bboxes": np.array(boxes, dtype=np.float64),
                    "matched": np.zeros(len(boxes), dtype=bool),
                }
                n_gt += len(boxes)

        # Collect all predictions for this class across all images
        cls_preds: List[Tuple[float, str, List[float]]] = []
        for item in predictions:
            img_id = item["image_id"]
            for det in item["detections"]:
                if det["class"] != cls_name:
                    continue
                conf = float(det["confidence"])
                if eval_conf_threshold is not None and conf < eval_conf_threshold:
                    continue
                cls_preds.append((conf, img_id, det["bbox"]))

        cls_preds.sort(key=lambda x: x[0], reverse=True)
        n_preds = len(cls_preds)
        tp = np.zeros(n_preds, dtype=np.float64)
        fp = np.zeros(n_preds, dtype=np.float64)

        for i, (conf, img_id, pred_box) in enumerate(cls_preds):
            gt_entry = cls_gts.get(img_id)
            if gt_entry is None:
                fp[i] = 1.0
                continue
            ious = compute_iou(pred_box, gt_entry["bboxes"])
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])
            if best_iou >= iou_threshold and not gt_entry["matched"][best_idx]:
                tp[i] = 1.0
                gt_entry["matched"][best_idx] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / float(n_gt) if n_gt > 0 else np.zeros_like(tp_cum)
        denom = np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
        prec = tp_cum / denom

        ap = voc_ap_2010_plus(rec, prec) if n_gt > 0 else 0.0
        aps.append(ap)

        cls_tp = int(np.sum(tp))
        cls_fp = int(np.sum(fp))
        cls_fn = int(n_gt - cls_tp)
        cls_prec = float(cls_tp / (cls_tp + cls_fp)) if (cls_tp + cls_fp) > 0 else 0.0
        cls_rec = float(cls_tp / n_gt) if n_gt > 0 else 0.0
        cls_f1 = float(2 * cls_prec * cls_rec / (cls_prec + cls_rec)) if (cls_prec + cls_rec) > 0 else 0.0

        total_tp += cls_tp
        total_fp += cls_fp
        total_gt_all += n_gt

        # Sample PR curve points (up to 50 points for compact JSON storage)
        if len(rec) > 0:
            idxs = np.unique(np.linspace(0, len(rec) - 1, min(50, len(rec)), dtype=int))
            pr_curve = {
                "recall": [float(rec[k]) for k in idxs],
                "precision": [float(prec[k]) for k in idxs],
            }
        else:
            pr_curve = {"recall": [], "precision": []}

        per_class_results[cls_name] = {
            "ap50": float(ap),
            "gt_count": int(n_gt),
            "pred_count": int(n_preds),
            "tp": cls_tp,
            "fp": cls_fp,
            "fn": cls_fn,
            "precision": cls_prec,
            "recall": cls_rec,
            "f1": cls_f1,
            "pr_curve": pr_curve,
        }

    overall_prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = float(total_tp / total_gt_all) if total_gt_all > 0 else 0.0
    overall_f1 = (
        float(2 * overall_prec * overall_rec / (overall_prec + overall_rec))
        if (overall_prec + overall_rec) > 0
        else 0.0
    )

    return {
        "mAP50": float(np.mean(aps)),
        "precision": overall_prec,
        "recall": overall_rec,
        "f1": overall_f1,
        "total_gt": int(total_gt_all),
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_gt_all - total_tp),
        "per_class": per_class_results,
    }


def categorize_failures(
    predictions: List[Dict],
    gt_by_image: Dict[str, Dict],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> Dict[str, List[Dict]]:
    """Extract representative False Positives, False Negatives, and Poor Localization cases."""
    pred_map = {p["image_id"]: p["detections"] for p in predictions}
    fp_cases: List[Dict] = []
    fn_cases: List[Dict] = []
    poor_loc_cases: List[Dict] = []

    for img_id, info in sorted(gt_by_image.items()):
        gts = info["boxes"]
        preds = [d for d in pred_map.get(img_id, []) if float(d["confidence"]) >= conf_threshold]
        preds.sort(key=lambda d: d["confidence"], reverse=True)

        matched_gt = set()
        for det in preds:
            det_cls = det["class"]
            det_box = det["bbox"]
            det_conf = float(det["confidence"])

            same_cls_indices = [i for i, g in enumerate(gts) if g["class"] == det_cls]
            if same_cls_indices:
                gt_arr = np.array([gts[i]["bbox"] for i in same_cls_indices], dtype=np.float64)
                ious = compute_iou(det_box, gt_arr)
                best_local = int(np.argmax(ious))
                best_gt_idx = same_cls_indices[best_local]
                best_iou = float(ious[best_local])
                if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                    matched_gt.add(best_gt_idx)
                    continue
                if 0.10 <= best_iou < iou_threshold:
                    poor_loc_cases.append(
                        {
                            "image_id": img_id,
                            "type": "poor_localization",
                            "class": det_cls,
                            "confidence": det_conf,
                            "iou": best_iou,
                            "pred_bbox": det_box,
                            "gt_bbox": gts[best_gt_idx]["bbox"],
                        }
                    )
                    continue

            # Check overlap with any GT of a different class
            all_gt_arr = np.array([g["bbox"] for g in gts], dtype=np.float64)
            any_ious = compute_iou(det_box, all_gt_arr)
            max_any_iou = float(np.max(any_ious)) if len(any_ious) > 0 else 0.0
            max_any_cls = gts[int(np.argmax(any_ious))]["class"] if len(any_ious) > 0 else None
            fp_cases.append(
                {
                    "image_id": img_id,
                    "type": "false_positive",
                    "pred_class": det_cls,
                    "confidence": det_conf,
                    "pred_bbox": det_box,
                    "nearest_gt_class": max_any_cls,
                    "max_iou_any_gt": max_any_iou,
                }
            )

        for i, g in enumerate(gts):
            if i not in matched_gt:
                gx1, gy1, gx2, gy2 = g["bbox"]
                fn_cases.append(
                    {
                        "image_id": img_id,
                        "type": "false_negative",
                        "gt_class": g["class"],
                        "gt_bbox": g["bbox"],
                        "gt_area": float((gx2 - gx1) * (gy2 - gy1)),
                    }
                )

    fp_cases.sort(key=lambda x: x["confidence"], reverse=True)
    poor_loc_cases.sort(key=lambda x: x["confidence"], reverse=True)
    fn_cases.sort(key=lambda x: x["gt_area"])

    return {
        "false_positives": fp_cases,
        "false_negatives": fn_cases,
        "poor_localization": poor_loc_cases,
    }

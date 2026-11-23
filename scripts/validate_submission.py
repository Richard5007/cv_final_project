"""Strict submission validator implementing FactoryVision Slides 24-34 checks."""

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score

CLASS_ORDER = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

REQUIRED_CONFIG_FIELDS = {
    1: [
        "team_id",
        "stage",
        "model_name",
        "input_size",
        "class_order",
        "normalization_mean",
        "normalization_std",
        "seed",
        "improvement_methods",
        "checkpoint_file",
    ],
    2: [
        "team_id",
        "stage",
        "architecture",
        "strategy",
        "pretrained_source",
        "input_size",
        "class_order",
        "normalization_mean",
        "normalization_std",
        "seed",
        "checkpoint_file",
    ],
    3: [
        "team_id",
        "stage",
        "detector",
        "pretrained_source",
        "input_size",
        "class_order",
        "confidence_threshold",
        "nms_iou_threshold",
        "bbox_format",
        "seed",
        "checkpoint_file",
    ],
}

FORBIDDEN_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:\\\\"),
    re.compile(r"[A-Za-z]:/"),
    re.compile(r"/home/"),
    re.compile(r"/Users/"),
]


def check_stage_submission(stage: int, submission_arg: Path, run_inference: bool = True) -> bool:
    errors: List[str] = []
    warnings: List[str] = []
    temp_dir = None

    try:
        if submission_arg.is_file() and submission_arg.suffix.lower() == ".zip":
            size_bytes = submission_arg.stat().st_size
            print(f"[Validator] Checking ZIP: {submission_arg.name} ({size_bytes / (1024*1024):.2f} MB)")
            if size_bytes > 1024**3:
                errors.append(f"ZIP exceeds 1 GB limit: {size_bytes} bytes")
            temp_dir = Path(tempfile.mkdtemp(prefix="fv_val_"))
            with zipfile.ZipFile(submission_arg, "r") as zf:
                zf.extractall(temp_dir)
            top_entries = [p for p in temp_dir.iterdir()]
            if len(top_entries) != 1 or not top_entries[0].is_dir():
                errors.append(f"ZIP must extract to exactly 1 root directory, found: {[p.name for p in top_entries]}")
                sub_root = temp_dir
            else:
                sub_root = top_entries[0]
            expected_stem = submission_arg.stem
            if sub_root.name != expected_stem:
                errors.append(f"ZIP root dir '{sub_root.name}' must match ZIP stem '{expected_stem}'")
        elif submission_arg.is_dir():
            sub_root = submission_arg.resolve()
            print(f"[Validator] Checking directory: {sub_root}")
        else:
            print(f"[FAIL] Submission path does not exist: {submission_arg}")
            return False

        if not re.match(rf"^Team\d{{2}}_Stage{stage}$", sub_root.name):
            errors.append(f"Root directory '{sub_root.name}' does not match TeamXX_Stage{stage}")

        script_name = f"stage{stage}_detect.py" if stage == 3 else f"stage{stage}_inference.py"
        ckpt_name = f"stage{stage}_model.pt" if stage == 3 else f"stage{stage}_model.pth"
        report_name = f"stage{stage}_report.pdf"

        required_files = [
            report_name,
            script_name,
            ckpt_name,
            "config.json",
            "requirements.txt",
            "README.md",
            "src/model.py",
            "src/train.py",
            "src/evaluate.py",
        ]
        for rel in required_files:
            p = sub_root / rel
            if not p.is_file():
                errors.append(f"Missing required file: {rel}")

        # Check PDF validity
        pdf_path = sub_root / report_name
        if pdf_path.is_file():
            header = pdf_path.read_bytes()[:8]
            if not header.startswith(b"%PDF-"):
                errors.append(f"{report_name} is not a valid PDF file.")
            elif pdf_path.stat().st_size < 5000:
                warnings.append(f"{report_name} is suspiciously small ({pdf_path.stat().st_size} bytes).")

        # Check single checkpoint rule
        ckpts = [
            p for p in sub_root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors", ".onnx"}
        ]
        if len(ckpts) != 1 or ckpts[0].relative_to(sub_root).as_posix() != ckpt_name:
            errors.append(f"Must have exactly 1 checkpoint '{ckpt_name}', found: {[c.relative_to(sub_root).as_posix() for c in ckpts]}")

        # Check config.json
        cfg_path = sub_root / "config.json"
        cfg = {}
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                for field in REQUIRED_CONFIG_FIELDS[stage]:
                    if field not in cfg:
                        errors.append(f"config.json missing required field: {field}")
                if cfg.get("stage") != stage:
                    errors.append(f"config.json stage={cfg.get('stage')} != {stage}")
                if cfg.get("class_order") != CLASS_ORDER:
                    errors.append(f"config.json class_order mismatch: {cfg.get('class_order')}")
                if cfg.get("checkpoint_file") != ckpt_name:
                    errors.append(f"config.json checkpoint_file={cfg.get('checkpoint_file')} != {ckpt_name}")
                if stage == 3 and cfg.get("bbox_format") != "xyxy_pixel":
                    errors.append(f"config.json bbox_format must be 'xyxy_pixel', got {cfg.get('bbox_format')}")
            except Exception as exc:
                errors.append(f"Invalid config.json: {exc}")

        # Check requirements.txt
        req_path = sub_root / "requirements.txt"
        if req_path.is_file():
            for line in req_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if any(tok in s for tok in ["-e ", "git+", "http://", "https://", "file://", ".whl", "/", "\\"]):
                    errors.append(f"Forbidden entry in requirements.txt: {s}")

        # Scan .py files for hardcoded personal paths
        for py_file in sub_root.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            txt = py_file.read_text(encoding="utf-8", errors="ignore")
            for pat in FORBIDDEN_PATH_PATTERNS:
                if pat.search(txt):
                    errors.append(f"Forbidden absolute path pattern '{pat.pattern}' in {py_file.relative_to(sub_root)}")

        # Run inference check from external working directory
        if run_inference and not errors:
            repo_root = Path(__file__).resolve().parent.parent
            ext_cwd = repo_root / "workspace_outputs" / "val_check"
            ext_cwd.mkdir(parents=True, exist_ok=True)
            script_abs = (sub_root / script_name).resolve()

            if stage in (1, 2):
                val_dir = (repo_root / "data/STUDENT_PUBLIC/classification/validation").resolve()
                out_csv = ext_cwd / f"val_stage{stage}_preds.csv"
                if out_csv.exists():
                    out_csv.unlink()
                cmd = [sys.executable, str(script_abs), "--input_dir", str(val_dir), "--output_csv", str(out_csv)]
                proc = subprocess.run(cmd, cwd=str(ext_cwd), capture_output=True, text=True, timeout=300)
                if proc.returncode != 0:
                    errors.append(f"Inference failed (rc={proc.returncode}): {proc.stderr[-500:]}")
                elif not out_csv.is_file():
                    errors.append(f"Output CSV not created: {out_csv}")
                else:
                    gt_map = {}
                    for cls_name in CLASS_ORDER:
                        for img_p in (val_dir / cls_name).glob("*.png"):
                            gt_map[img_p.stem] = cls_name
                    with open(out_csv, "r", encoding="utf-8") as f:
                        reader = list(csv.reader(f))
                    if not reader or reader[0] != ["image_id", "prediction"]:
                        errors.append(f"CSV header mismatch: {reader[0] if reader else 'EMPTY'}")
                    else:
                        preds_map = {}
                        for row in reader[1:]:
                            if len(row) != 2:
                                errors.append(f"Invalid CSV row length: {row}")
                                continue
                            img_id, pred = row
                            if pred not in CLASS_ORDER:
                                errors.append(f"Invalid predicted class '{pred}' for {img_id}")
                            if img_id in preds_map:
                                errors.append(f"Duplicate image_id in CSV: {img_id}")
                            preds_map[img_id] = pred
                        if set(preds_map.keys()) != set(gt_map.keys()):
                            errors.append(f"CSV image_id set mismatch: got {len(preds_map)} vs expected {len(gt_map)}")
                        else:
                            y_true = [gt_map[k] for k in sorted(gt_map)]
                            y_pred = [preds_map[k] for k in sorted(gt_map)]
                            acc = accuracy_score(y_true, y_pred)
                            mf1 = f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
                            print(f"[Validator] Stage {stage} external inference PASS: N={len(y_true)}, Val Acc={acc:.4f}, Val Macro F1={mf1:.4f}")
            else:
                val_dir = (repo_root / "data/STUDENT_PUBLIC/detection/validation/images").resolve()
                voc_dir = (repo_root / "data/STUDENT_PUBLIC/detection/validation/annotations_voc").resolve()
                out_json = ext_cwd / "val_stage3_dets.json"
                if out_json.exists():
                    out_json.unlink()
                cmd = [sys.executable, str(script_abs), "--input_dir", str(val_dir), "--output_json", str(out_json)]
                proc = subprocess.run(cmd, cwd=str(ext_cwd), capture_output=True, text=True, timeout=300)
                if proc.returncode != 0:
                    errors.append(f"Detection failed (rc={proc.returncode}): {proc.stderr[-500:]}")
                elif not out_json.is_file():
                    errors.append(f"Output JSON not created: {out_json}")
                else:
                    sys.path.insert(0, str((sub_root / "src").resolve()))
                    from evaluate import evaluate_detections_voc, parse_voc_annotations, validate_detections_schema

                    gt_by_image = parse_voc_annotations(voc_dir)
                    preds = json.loads(out_json.read_text(encoding="utf-8"))
                    s_errs = validate_detections_schema(preds, gt_by_image)
                    if s_errs:
                        errors.extend(s_errs[:10])
                    else:
                        ev = evaluate_detections_voc(preds, gt_by_image, iou_threshold=0.5)
                        print(
                            f"[Validator] Stage 3 external detection PASS: N={len(preds)}, "
                            f"Val mAP@0.5={ev['mAP50']:.4f}, P={ev['precision']:.4f}, R={ev['recall']:.4f}"
                        )

        for w in warnings:
            print(f"[WARN] {w}")
        if errors:
            for e in errors:
                print(f"[ERROR] {e}")
            print(f"[FAIL] Stage {stage} validation FAILED with {len(errors)} error(s).")
            return False

        print(f"[PASS] Stage {stage} validation PASSED!")
        return True
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="FactoryVision Submission Validator")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--submission", type=str, required=True)
    parser.add_argument("--no_inference", action="store_true")
    args = parser.parse_args()

    ok = check_stage_submission(args.stage, Path(args.submission), run_inference=(not args.no_inference))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

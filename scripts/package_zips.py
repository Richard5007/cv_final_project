"""Package clean Team21_Stage1.zip, Team21_Stage2.zip, and Team21_Stage3.zip."""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def zip_stage(stage_num: int) -> Path:
    folder_name = f"Team21_Stage{stage_num}"
    src_dir = ROOT / folder_name
    out_zip = ROOT / f"{folder_name}.zip"
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Missing submission directory: {src_dir}")

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if "__pycache__" in p.parts or p.suffix.lower() in {".pyc", ".pyo"}:
                continue
            if p.is_file():
                arcname = f"{folder_name}/{p.relative_to(src_dir).as_posix()}"
                zf.write(p, arcname=arcname)

    mb = out_zip.stat().st_size / (1024.0 * 1024.0)
    print(f"[Package] Created {out_zip.name}: {mb:.2f} MB")
    return out_zip


def main() -> None:
    for s in (1, 2, 3):
        zip_stage(s)


if __name__ == "__main__":
    main()

"""One-click script to rebuild Team21_Final_Report.pdf after updating scripts/benchmark_scores.json."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "build_reports.py"), "--only_final"]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()

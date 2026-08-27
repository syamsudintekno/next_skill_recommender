from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Stage-3 bounded validation budget")
    parser.add_argument("--family", choices=["integrated", "posthoc", "all"], default="all")
    args = parser.parse_args()
    config_dir = PROJECT_ROOT / "stages/stage3_proposed/configs/bounded"
    configs = sorted(config_dir.glob("integrated_*.json")) if args.family in {"integrated", "all"} else []
    if args.family in {"posthoc", "all"}:
        configs += sorted(config_dir.glob("posthoc_*.json"))
    if not configs:
        raise RuntimeError("No bounded configs found")
    for config in configs:
        payload = json.loads(config.read_text(encoding="utf-8"))
        result = PROJECT_ROOT / "runs/stage3" / payload["run_id"] / "result.json"
        if result.exists():
            print(f"SKIP existing immutable result: {result}")
            continue
        runner = "run_posthoc.py" if payload["mode"] == "posthoc" else "run_stage3.py"
        command = [sys.executable, str(PROJECT_ROOT / "stages/stage3_proposed" / runner), "--config", str(config), "--quiet"]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

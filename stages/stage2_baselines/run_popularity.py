from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / ".deps"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.integrity import run_development_integrity
from src.popularity import run_popularity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/canonical")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs/stage2/popularity/POP_DEV_001.json",
    )
    args = parser.parse_args()

    integrity = run_development_integrity(args.data)
    result, accessed = run_popularity(args.data)
    payload = {
        "run_id": "POP_DEV_001",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "development",
        "model": "global_user_support_popularity",
        "k": 10,
        "tie_break": "skill_id_lexicographic_ascending",
        "integrity": integrity,
        "metrics": asdict(result),
        "model_files_accessed": accessed,
        "manifest_verification": "NOT_RUN: Stage 1 per-file manifest absent from ZIP/workspace",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

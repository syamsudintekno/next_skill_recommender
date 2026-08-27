from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / ".deps"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.integrity import metadata_inventory


def main() -> None:
    root = PROJECT_ROOT / "data/canonical"
    report = {
        "scope": "structural_and_provenance_audit_only",
        "note": (
            "Parquet footers and raw bytes were inspected for row counts, schemas, and hashes. "
            "Test-target columns/rows were not decoded. This audit is separate from development runs."
        ),
        "manifest_comparison": "BLOCKED: manifest absent from supplied ZIP and workspace",
        "artifacts": metadata_inventory(root),
    }
    output = PROJECT_ROOT / "runs/stage2/audits/S2_1_ARTIFACT_AUDIT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

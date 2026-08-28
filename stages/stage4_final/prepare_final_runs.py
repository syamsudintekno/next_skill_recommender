from __future__ import annotations

import json
from pathlib import Path

from final_protocol import ROOT, TRAINED_FAMILIES, load_manifest, verify_artifact_snapshot


def main() -> None:
    manifest = load_manifest()
    # Deliberately excludes test_targets.parquet.
    verified = verify_artifact_snapshot(include_test=False)
    destination = ROOT / "runs/stage4/config_snapshots"
    destination.mkdir(parents=True, exist_ok=True)
    common = {"max_epochs": 100, "test_access": False}
    count = 0
    for family in TRAINED_FAMILIES:
        model = dict(manifest["models"][family])
        if family == "integrated_asymmetric_squared":
            model = {**manifest["models"]["lightgcn"], **model}
        for seed in manifest["seeds"]:
            run_id = f"{family.upper()}_FINAL_{seed}"
            config = {"run_id": run_id, "family": family, "seed": seed, **common, **model}
            (destination / f"{run_id}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            count += 1
    payload = {
        "status": "PREPARED_WITHOUT_TEST_ACCESS",
        "training_configs": count,
        "verified_non_test_artifacts": verified,
        "test_accessed": False,
    }
    (destination / "PREPARATION.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

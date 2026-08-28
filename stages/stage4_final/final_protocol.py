from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "stages/stage4_final/configs/final_protocol_manifest.json"
TRAINED_FAMILIES = ("bpr_mf", "lightgcn", "xsimgcl", "integrated_asymmetric_squared")


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_BEFORE_TEST_ACCESS" or manifest["test_accessed"] is not False:
        raise PermissionError("Final protocol is not in the frozen pre-test state")
    return manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact_snapshot(*, include_test: bool = False) -> list[str]:
    """Verify frozen bytes; test hash is optional so training never opens it."""
    manifest = load_manifest()
    checked = []
    for filename, expected in manifest["artifact_sha256"].items():
        if filename == "test_targets.parquet" and not include_test:
            continue
        actual = sha256(ROOT / "data/canonical" / filename)
        if actual != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {filename}")
        checked.append(filename)
    return checked


def expected_training_runs() -> list[str]:
    seeds = load_manifest()["seeds"]
    return [f"{family.upper()}_FINAL_{seed}" for family in TRAINED_FAMILIES for seed in seeds]


def assert_global_training_barrier() -> list[Path]:
    """Require all 20 stochastic checkpoints before any test evaluation."""
    receipts = []
    for run_id in expected_training_runs():
        run_dir = ROOT / "runs/stage4/training" / run_id
        receipt_path = run_dir / "training_complete.json"
        checkpoint_candidates = [run_dir / "checkpoint.pt", run_dir / "checkpoint.npz"]
        checkpoint = next((p for p in checkpoint_candidates if p.is_file()), None)
        if checkpoint is None or not receipt_path.is_file():
            raise PermissionError(f"Global training barrier incomplete: {run_id}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("run_id") != run_id or receipt.get("completed_epochs") != 100:
            raise PermissionError(f"Invalid completion receipt: {run_id}")
        if receipt.get("checkpoint_sha256") != sha256(checkpoint):
            raise PermissionError(f"Checkpoint/receipt mismatch: {run_id}")
        if receipt.get("status") != "TRAINING_COMPLETE_TEST_NOT_ACCESSED" or receipt.get("test_accessed") is not False:
            raise PermissionError(f"Run is not eligible for first test access: {run_id}")
        receipts.append(receipt_path)
    return receipts

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "stages/stage2_baselines/configs"
RUNNER = ROOT / "stages/stage2_baselines/run_lightgcn.py"
CONFIGS = [CONFIG_DIR / f"lightgcn_bound_{i:03d}.json" for i in range(1, 5)]


def main() -> None:
    summaries = []
    for config in CONFIGS:
        print(f"Running {config.stem} ...", flush=True)
        subprocess.run(
            [sys.executable, str(RUNNER), "--config", str(config), "--quiet"],
            cwd=ROOT,
            check=True,
        )
        run_id = json.loads(config.read_text(encoding="utf-8"))["run_id"]
        result_path = ROOT / "runs/stage2/lightgcn" / run_id / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        summaries.append(
            {
                "run_id": run_id,
                "best_epoch": result["best_epoch"],
                **result["metrics"],
                "runtime_seconds": result["runtime_seconds"],
            }
        )
    output = ROOT / "runs/stage2/lightgcn/bounded_tuning_summary.json"
    output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    winner = max(summaries, key=lambda row: row["ndcg_at_10"])
    print("Completed. Summary:", output)
    print("Best validation run:", json.dumps(winner))


if __name__ == "__main__":
    main()

# Stage 4 — Final experiment

Status: protocol frozen; test execution is not yet enabled.

`configs/final_protocol_manifest.json` is authoritative. The final loader has no
generic targets method. Test targets require a persisted checkpoint and matching
100-epoch completion receipt with state
`TRAINING_COMPLETE_TEST_NOT_ACCESSED`.

Do not manually inspect `test_targets.parquet`. Execution commands will be
published only after the isolated runner and summarizer pass their safeguards.

The preparation step verifies the eleven non-test frozen artifacts and creates
20 immutable stochastic-training config snapshots. A global barrier requires
all 20 checkpoints and matching receipts before any final evaluator can open
the test artifact.

Target-free training may be run by family and resumed safely:

```powershell
python stages/stage4_final/run_all_training.py --family bpr_mf
python stages/stage4_final/run_all_training.py --family lightgcn
python stages/stage4_final/run_all_training.py --family xsimgcl
python stages/stage4_final/run_all_training.py --family integrated_asymmetric_squared
```

These commands cannot evaluate test targets. Do not run any test command yet;
the batch evaluator and final summarizer are still pending safeguard audit.

After all 20 receipts pass the global barrier, the one-time batch command is:

```powershell
python stages/stage4_final/evaluate_final_once.py
python stages/stage4_final/summarize_final.py
```

The evaluator creates a lock before its single test read, evaluates all 26
eligible outputs in memory, and refuses a second invocation. Never run it until
the pre-execution audit explicitly reports PASS.

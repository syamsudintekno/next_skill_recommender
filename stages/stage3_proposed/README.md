# Stage 3 — Difficulty-Regularized LightGCN

Status: development implementation; hyperparameters are not frozen and no test target may be used.

The three runners preserve the selected Stage-2 LightGCN architecture, optimizer budget,
full-ranking candidate set, validation evaluator, early stopping rule, and seed policy.
`lambda_zero` is the mechanical equivalence control, `integrated` adds exact expected
asymmetric squared risk over every development learner's full unseen catalog, and `posthoc`
trains relevance-only LightGCN before applying `score - rerank_weight * risk`.

Run the leakage/proxy/gradient audit first:

```powershell
python stages/stage3_proposed/audit_stage3.py
```

Then run one configuration from the project root in the VS Code terminal:

```powershell
python stages/stage3_proposed/run_stage3.py --config stages/stage3_proposed/configs/lambda_zero_dev.json --quiet
python stages/stage3_proposed/run_stage3.py --config stages/stage3_proposed/configs/integrated_dev.json --quiet
python stages/stage3_proposed/run_stage3.py --config stages/stage3_proposed/configs/posthoc_dev.json --quiet
```

Each run writes an immutable checkpoint, complete per-epoch history, configuration,
proxy audit, accessed-file list, relevance metrics, DVR@10, MED@10, squared risk,
and item exposure counts under `runs/stage3/<run_id>/result.json`.

The example values `tolerance=0.1`, `temperature=0.2`, `difficulty_weight=0.1`, and
`rerank_weight=0.1` are **PROPOSAL — OPEN**, not selected settings. Do not compare
integrated and post-hoc results until a matched, predeclared validation budget is recorded.

## Frozen bounded coefficient selection

The matched budget is now frozen in `configs/bounded_tuning_manifest.json`.
It contains four integrated coefficients and four post-hoc coefficients, all at
the same selection seed, tolerance, candidate set, evaluator, and validation-only
selection rule. Post-hoc runs reuse the bit-identical lambda-zero checkpoint and
do not retrain LightGCN.

Run the complete budget:

```powershell
python stages/stage3_proposed/run_bounded_tuning.py --family all
```

Or run one family at a time:

```powershell
python stages/stage3_proposed/run_bounded_tuning.py --family integrated
python stages/stage3_proposed/run_bounded_tuning.py --family posthoc
```

Existing immutable result directories are skipped. Send back the eight
`runs/stage3/*_BOUND_*/result.json` files after completion. Do not alter the
manifest or add trials after viewing results.

After all eight results exist, apply the frozen selection rule with:

```powershell
python stages/stage3_proposed/select_bounded.py
```

This writes `runs/stage3/STAGE3_BOUNDED_SELECTION.json`; please send that file
with the eight result files.

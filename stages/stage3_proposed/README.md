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

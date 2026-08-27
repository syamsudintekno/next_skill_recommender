# Difficulty-Regularized LightGCN for Educational Recommendation

Research code for the JOIN paper **“Balancing Relevance and Overchallenge in Graph-Based Educational Recommendation: An Asymmetric Difficulty-Regularized LightGCN.”**

## Project layout

- `stages/stage1_data_design/`: raw-data audit and frozen canonical artifact construction.
- `stages/stage2_baselines/`: development-only baseline entry points and configurations.
- `stages/stage3_proposed/`: reserved for integrated difficulty regularization, reranking, and ablations.
- `src/`: shared data loading, integrity, candidate construction, evaluation, metrics, and models.
- `data/canonical/`: the 12 frozen Stage 1 Parquet artifacts.
- `data/manifests/`: artifact manifests and checksums.
- `runs/`: immutable run records organized by stage and model.
- `docs/`: decisions, design freeze, experiment log, results ledger, and manuscript changelog.
- `sources/`: read-only research references.

## Current status

- Stage 1: frozen.
- Stage 2: conditionally complete; the complete 12-file Stage 1 hash manifest remains open.
- Stage 3: development-only loader/objective audit passed; runners are ready for local execution.
- Test targets are not available to development runners.

Run Stage 2 checks from the project root with:

```powershell
python -m unittest discover -s tests -v
python stages/stage2_baselines/audit_artifacts.py
python stages/stage2_baselines/run_popularity.py
```

Stage 3 commands and configuration status are documented in
`stages/stage3_proposed/README.md`. Test targets remain unavailable to all
development runners.

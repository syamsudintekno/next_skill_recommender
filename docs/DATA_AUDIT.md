# DATA_AUDIT

## 2026-08-27 — Stage 2 canonical handoff

- **FACT — VERIFIED:** supplied ZIP contains exactly the 12 expected canonical Parquet filenames.
- **FACT — VERIFIED:** structural row counts and schemas are recorded in `runs/stage2/audits/S2_1_ARTIFACT_AUDIT.json`.
- **FACT — VERIFIED:** development graph has 22,241 learners, 262 skills, and 320,119 unique learner–skill edges; duplicate edges = 0.
- **FACT — VERIFIED:** validation has 22,239 evaluable targets and 2 globally cold targets. Target/seen overlap = 0; visible targets outside the development catalog = 0.
- **FACT — VERIFIED:** mean full-ranking candidate count is 247.606807 (min 176; max 261).
- **FACT — PARTIAL PASS:** `stage1_proxy_audit.py` contains frozen hashes for three development artifacts (`development_train_events`, `development_graph_edges`, and `development_ability_inputs`); all three match the current files exactly.
- **DISCREPANCY — OPEN:** the ZIP contains no complete Stage 1 12-file hash manifest. Current SHA-256 values are captured as handoff provenance, but full equality against the frozen Stage 1 materialization cannot be asserted.
- **FIREWALL:** structural audit inspected Parquet footers and hashes for all 12 files without decoding test-target rows. Development evaluation reads only `development_graph_edges.parquet`, `development_catalog.parquet`, and `validation_targets.parquet`.

## 2026-08-27 — Stage 3 development-proxy loader audit

- **FACT — VERIFIED:** Stage 3 proxy construction read only the development graph, catalog, train events, difficulty inputs, and ability inputs; neither validation nor test targets were read by the audit.
- **FACT — VERIFIED:** all-event counts/successes in `development_difficulty_inputs.parquet` match the development event prefix, and its learner support matches 320,119 reconstructed first learner–skill exposures.
- **FACT — VERIFIED:** `development_ability_inputs.parquet` contains 320,119 learner–skill rows, matching the binary development graph pairs.
- **FACT — VERIFIED:** reconstructed empirical-Bayes prior is alpha=3.7439989264 and beta=2.6928005966; the ability population center is 0.3591255543 and estimated shrinkage kappa is 15.4729027778.
- **FACT — VERIFIED:** 620 development learners have no successful prefix skill and receive the population-center fallback before shrinkage.
- Trace: `runs/stage3/audits/STAGE3_LOADER_OBJECTIVE_AUDIT.json`.

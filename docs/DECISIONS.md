# DECISIONS

## 2026-08-27

- **DECISION — LOCKED:** common development evaluator uses deterministic full ranking and seen-item masking.
- **DECISION — LOCKED:** Popularity score is development graph user support; ties are broken by ascending lexical `skill_id`.
- **DECISION — LOCKED:** run `POP_DEV_001` is a diagnostic sanity baseline, not a manuscript contribution.
- **DECISION — REVISED:** artifact-wide structural/provenance audit is a separate command from every development model run. This makes the no-test-access firewall mechanically auditable.
- **DECISION — PROVISIONAL:** BPR-MF uses uniform sampling from learner-unseen development-catalog items. These are optimization samples, not negative-preference labels.
- **DECISION — REVISED/LOCKED:** BPR-MF Stage 2 configuration is selected from a bounded four-run budget: dimension 64, LR 0.02, L2 0.001, batch size 4,096, epoch cap 100. The cap is not extended after observing validation.
- **DECISION — OPEN:** final multi-seed execution of the selected BPR-MF configuration remains pending.
- **DECISION — REJECTED:** full-batch LightGCN training with one optimizer update per epoch is not eligible for baseline selection because the optimization budget is not comparable to mini-batch BPR.
- **FACT — DIAGNOSTIC:** two-layer LightGCN strongly underperformed one-layer LightGCN on the current development graph, consistent with possible oversmoothing but not sufficient to claim oversmoothing causally.
- **DECISION — OPEN:** LightGCN tuning will proceed from the valid one-layer stochastic protocol; no LightGCN configuration is frozen yet.
- **DECISION — REVISED/LOCKED:** LightGCN Stage 2 configuration is selected from the bounded four-run budget: one layer, dimension 64, LR 0.005, L2 0.0001, batch 65,536, five steps per epoch, epoch cap 100.
- **DECISION — OPEN:** final multi-seed execution of selected BPR-MF and LightGCN configurations remains pending.
- **DECISION — REVISED/COMPLETE:** selected BPR-MF and LightGCN configurations have completed five-seed validation. BPR-MF is consistently stronger on all three relevance metrics.
- **DECISION — LOCKED:** XSimGCL is selected as the modern graph recommender baseline. Rationale: official PyTorch implementation, direct LightGCN lineage, noise-based contrastive augmentation without graph-structure dropout, and feasible adaptation to the shared development-only evaluator.
- **DECISION — REJECTED:** LightGCL is not selected because its truncated-SVD view adds a distinct low-rank design and additional rank/dropout tuning burden on a 262-item catalog.
- **DECISION — REJECTED:** SGL is not selected because stochastic node/edge/random-walk graph augmentation expands preprocessing and fairness risks relative to the frozen graph.
- **DECISION — LOCKED:** selected XSimGCL configuration is LR 0.002, contrastive weight 0.2, epsilon 0.2, temperature 0.2, two layers, and contrastive layer 1. It was selected from the bounded four-run validation budget.
- **DECISION — COMPLETE:** selected XSimGCL completed five-seed validation. It improves NDCG/MRR over LightGCN but not Recall, and remains below BPR-MF on all three relevance metrics.
- **STAGE STATUS — CONDITIONALLY COMPLETE:** Stage 2 implementation, bounded tuning, five-seed validation, modern baseline, and leakage audit are complete. Full closure remains conditional only on the missing 12-file Stage 1 hash manifest (S2-A01 partial pass).

## 2026-08-27 — Stage 3 start

- **DECISION — LOCKED:** Stage 3 inherits the selected LightGCN architecture and optimization protocol unchanged: dimension 64, one propagation layer, LR 0.005, L2 0.0001, batch 65,536, five steps per epoch, epoch cap 100, validation NDCG@10 checkpointing, and seeds 20260827–20260831.
- **DECISION — LOCKED:** difficulty is reconstructed from first learner–skill exposure correctness with the Stage 1 empirical-Bayes Beta-Binomial method; the all-event difficulty artifact is an audit input, not the final proxy definition.
- **DECISION — LOCKED:** ability uses equal weighting over unique skills with any prefix success, method-of-moments shrinkage, and population-center fallback, all from the development prefix.
- **DECISION — LOCKED:** integrated regularization uses the exact softmax expectation over the frozen full unseen candidate catalog, averaged uniformly over all development learners at every optimizer step. This avoids degree-biased learner sampling in the pedagogical term. A fixed unweighted risk sum is rejected because it has no gradient path.
- **DECISION — LOCKED:** `difficulty_weight=0` skips risk computation in the objective exactly, preserving the Stage 2 relevance-only update path.
- **DECISION — LOCKED:** post-hoc comparison trains relevance-only LightGCN and changes ranking scores only as `score - rerank_weight * asymmetric_squared_risk`.
- **DECISION — OPEN:** tolerance, temperature, difficulty weight, rerank weight, and the bounded matched tuning grid are not frozen. Values in the initial development configs are implementation probes only.
- **DECISION — OPEN:** operational definitions of DVR@10 and MED@10 are implemented as violation share and mean positive excess, respectively, but require explicit design-freeze confirmation before manuscript use.
- **FACT — VERIFIED:** the Stage 3 λd=0 run is bit-identical to selected-seed Stage 2 LightGCN, including checkpoint SHA-256 and relevance metrics.
- **DECISION — REJECTED:** the initial integrated and post-hoc probes are not eligible for model selection because they were not generated under a predeclared bounded matched budget.
- **DECISION — REJECTED:** extending the 100-epoch cap after all initial Stage 3 probes reached the ceiling is not allowed.
- **DECISION — OPEN:** the next run set requires an equal-sized coefficient budget for integrated and post-hoc variants and a validation-only Pareto selection rule frozen before execution.

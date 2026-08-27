# EXPERIMENT_LOG

## POP_DEV_001 — 2026-08-27

- Status: completed, development only.
- Inputs: development graph edges, development catalog, validation targets.
- Model: global item popularity measured by unique learner support.
- Evaluation: full candidate ranking, seen-prefix masking, K=10, deterministic lexical tie-break.
- Result source: `runs/stage2/popularity/POP_DEV_001.json`.
- Test targets: not accessed by the accepted development run.

## BPRMF_DEV_001 — 2026-08-27

- Status: completed, provisional development diagnostic.
- Inputs: development graph edges, development catalog, validation targets.
- Configuration: seed 20260827; 64 dimensions; learning rate 0.01; L2 0.0001; batch size 4,096; maximum 50 epochs.
- Negative sampling semantics: uniformly sampled unobserved items for pairwise optimization, not negative preferences.
- Selection criterion: validation NDCG@10 evaluated every five epochs.
- Best checkpoint: epoch 50; the metric was still improving at the configured ceiling, so convergence/early stopping was not established.
- Runtime: 137.269 seconds.
- Checkpoint SHA-256: `e5473881a915e05a89a545492e9c6afd07dbf05184f0562e88b0b84014659eae`.
- Result source: `runs/stage2/bpr_mf/BPRMF_DEV_001/result.json`.
- Test targets: not accessed.

## BPR-MF bounded tuning — 2026-08-27

- Budget: four configurations, one development seed (20260827), maximum 100 epochs each; no post-result grid expansion.
- TUNE_001: dim 32, LR 0.02, L2 0.0001; best epoch 90; NDCG@10 0.292339.
- TUNE_002: dim 64, LR 0.02, L2 0.0001; best epoch 100; NDCG@10 0.307544.
- TUNE_003: dim 64, LR 0.01, L2 0.0001; best epoch 100; NDCG@10 0.294770.
- TUNE_004: dim 64, LR 0.02, L2 0.001; best epoch 100; NDCG@10 0.313874 (**selected**).
- Selection caveat: the winner reached the epoch cap; the cap is retained rather than extended after observing validation.
- Full final-seed evaluation remains pending; no test run exists.

## LightGCN diagnostics — 2026-08-27

- `LIGHTGCN_DEV_001`: two layers, LR 0.01, full-batch update; best epoch 5, NDCG@10 0.143859. **Rejected training protocol** because it performed only one optimizer update per epoch and was not computationally comparable to stochastic BPR-MF.
- `LIGHTGCN_TUNE_001`: one layer, LR 0.001, full-batch update; best epoch 20, NDCG@10 0.148680. **Invalidated for selection** with the full-batch protocol.
- `LIGHTGCN_DEV_002`: two layers, LR 0.001, stochastic mini-batch BPR (5 × 65,536 per epoch); best epoch 40, NDCG@10 0.142917. Valid diagnostic, but severe underperformance/oversmoothing suspected.
- `LIGHTGCN_DEV_003`: one layer, LR 0.005, same stochastic protocol; best epoch 40, Recall@10 0.434957, NDCG@10 0.237573, MRR@10 0.177699. Metric was still improving at the cap.
- All LightGCN runs were relevance-only and accessed development graph, development catalog, and validation targets only.
- LightGCN tuning remains open; no configuration is frozen.

## Selected baseline multi-seed validation — 2026-08-27

- Seeds: 20260827–20260831 (five predeclared seeds).
- BPR-MF: Recall@10 0.531993±0.000719; NDCG@10 0.311951±0.001684; MRR@10 0.244718±0.002124.
- LightGCN: Recall@10 0.484626±0.002695; NDCG@10 0.286187±0.001738; MRR@10 0.225614±0.001822.
- All values are validation-only; test remains unopened for evaluation.

## Modern graph baseline selection — 2026-08-27

- Selected: XSimGCL (TKDE 2023), using the authors' official SELFRec implementation as the primary implementation reference.
- Required controls: identical development graph, candidates, evaluator, validation selection, seed set, and no test access.
- Implementation and bounded tuning: pending.

## XSimGCL selected multi-seed validation — 2026-08-27

- Recall@10: 0.480579±0.003957.
- NDCG@10: 0.294421±0.001904.
- MRR@10: 0.237233±0.001384.
- Validation only; test targets were not used.

## Stage 2 closure audit — 2026-08-27

- Seven automated tests passed.
- Thirty result files audited; zero development→test/final access violations.
- Popularity, BPR-MF, LightGCN, and XSimGCL complete; learned baselines have five validation seeds.
- Status: **CONDITIONALLY COMPLETE** because the complete 12-file Stage 1 manifest remains unavailable.
- Test evaluation has not been performed.

## STAGE3_LOADER_OBJECTIVE_AUDIT — 2026-08-27

- Status: passed, development-prefix audit only; no validation or test target was read.
- Verified reconstruction of first-exposure empirical difficulty and shrunk learner ability against the supplied development aggregates/graph.
- Fitted prior: alpha 3.7439989264, beta 2.6928005966; population ability 0.3591255543; kappa 15.4729027778; 620 population-fallback learners.
- Synthetic objective check: expected-risk loss 0.1823261827 and parameter-gradient L1 0.3959889412; a raw fixed-risk sum has no gradient path.
- Eleven automated Stage 2–3 tests passed.
- Result source: `runs/stage3/audits/STAGE3_LOADER_OBJECTIVE_AUDIT.json`.
- Multi-epoch Stage 3 runs: not executed; configs are ready for user execution in VS Code.

## Stage 3 initial development probes — 2026-08-27

- Runs: `DRLGCN_LAMBDA0_DEV_20260827`, `DRLGCN_INTEGRATED_DEV_20260827`, and `LIGHTGCN_POSTHOC_DEV_20260827`; seed 20260827, validation only.
- **FACT — VERIFIED:** λd=0 exactly reproduces `LIGHTGCN_BOUND_004`: identical checkpoint SHA-256 and identical Recall/NDCG/MRR at 10.
- Integrated probe (τ=0.1, T=0.2, λd=0.1): Recall 0.487477, NDCG 0.283190, MRR 0.220719, DVR 0.183629, MED 0.015806, and mean squared risk 0.002193.
- Post-hoc probe (τ=0.1, γ=0.1): Recall 0.485004, NDCG 0.286016, MRR 0.225218, DVR 0.186678, MED 0.016433, and mean squared risk 0.002317.
- Relative to λd=0, integrated changed Recall by +0.002473, NDCG by -0.002888, MRR by -0.004582, DVR by -0.003197, MED by -0.000658, and squared risk by -0.000130.
- **INFERENCE — PROVISIONAL:** integrated regularization produced the intended risk direction at this one setting, with a ranking-quality cost except for Recall. The single result does not establish superiority or statistical significance.
- **INFERENCE — PROVISIONAL:** γ=0.1 is too weak to make the post-hoc probe an informative comparator.
- All three runs reached epoch 100; the inherited cap must not be extended in response.
- Trace: `runs/stage3/audits/STAGE3_DEV_RUN_AUDIT.json`.

## Stage 3 score/risk scale audit and bounded-budget freeze — 2026-08-27

- Audit used the λd=0 checkpoint and unseen development candidates without reading validation or test targets.
- Unseen LightGCN score SD: 2.525652. Positive asymmetric squared-risk median/p90: 0.018004/0.104013; positive candidate fraction: 0.355415.
- Based on scale rather than target performance, the post-hoc bounded range is γ `{1, 5, 10, 20}`.
- Matched four-run integrated range is λd `{0.03, 0.1, 0.3, 1.0}`; shared τ=0.1, T=0.2, and seed 20260827.
- Status: frozen before execution. Result-driven expansion is prohibited.
- Trace: `runs/stage3/audits/STAGE3_SCORE_RISK_SCALE_AUDIT.json` and `stages/stage3_proposed/configs/bounded_tuning_manifest.json`.

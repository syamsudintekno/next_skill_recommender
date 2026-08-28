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
- Code snapshot before bounded execution: Git commit `7231af6` (`git rev-parse --short HEAD`, reported from project root).
- Trace: `runs/stage3/audits/STAGE3_SCORE_RISK_SCALE_AUDIT.json` and `stages/stage3_proposed/configs/bounded_tuning_manifest.json`.

## Stage 3 bounded coefficient selection — 2026-08-28

- Status: completed, validation-only, seed 20260827; eight frozen configurations audited.
- Integrated λd `{0.03, 0.1, 0.3, 1.0}` formed a monotonic NDCG–DVR trade-off; every point was non-dominated. All checkpoints selected epoch 100.
- Post-hoc γ `{1, 5, 10, 20}` also formed a monotonic NDCG–DVR trade-off; every point was non-dominated.
- Under the frozen 1% relative NDCG-loss guardrail, integrated selected λd=0.03 (`DRLGCN_BOUND_001`; loss 0.6482%) and post-hoc selected γ=5 (`POSTHOC_BOUND_002`; loss 0.5920%).
- Selected integrated: Recall 0.485453, NDCG 0.284224, MRR 0.222666, DVR 0.185306, MED 0.016170, squared risk 0.002265.
- Selected post-hoc: Recall 0.482890, NDCG 0.284385, MRR 0.223731, DVR 0.178836, MED 0.014879, squared risk 0.001989.
- **INFERENCE — PROVISIONAL:** at this selection seed, post-hoc gives the stronger NDCG–risk point, whereas integrated retains higher Recall. Multi-seed validation is required before comparative claims.
- No test or `final_*` artifact was accessed. Selection was independently recomputed and matched `STAGE3_BOUNDED_SELECTION.json`.
- Trace: `runs/stage3/audits/STAGE3_BOUNDED_SELECTION_AUDIT.json`.

## Stage 3 selected-model five-seed validation freeze — 2026-08-28

- Status: frozen before execution; no new multi-epoch run performed by the preparation step.
- Integrated configuration: λd=0.03, τ=0.1, T=0.2, inherited selected LightGCN training protocol.
- Post-hoc configuration: γ=5, τ=0.1, matched selected LightGCN checkpoint per seed.
- Seeds: 20260827–20260831. The accepted bounded results supply seed 20260827; four new runs per family remain.
- Reporting: mean±sample SD and paired post-hoc-minus-integrated differences with two-sided 95% paired-t confidence intervals.
- Checkpoint paths and SHA-256 values are frozen in `stages/stage3_proposed/configs/multiseed_validation_manifest.json`.
- Test targets remain unopened; coefficient revision from multi-seed results is prohibited.

## Stage 3 selected-model five-seed validation — 2026-08-28

- Status: completed, five matched validation seeds; no test or `final_*` artifact access.
- Integrated λd=0.03: Recall 0.484860±0.002974; NDCG 0.284372±0.001572; MRR 0.223112±0.001432; DVR 0.183588±0.001670; MED 0.016091±0.000193; squared risk 0.002257±0.000029.
- Post-hoc γ=5: Recall 0.482765±0.003199; NDCG 0.284507±0.001675; MRR 0.223997±0.001598; DVR 0.177589±0.001401; MED 0.014851±0.000182; squared risk 0.001985±0.000030.
- Paired post-hoc-minus-integrated Recall difference: -0.002095, 95% CI [-0.003015, -0.001176].
- Paired NDCG difference: +0.000135, 95% CI [-0.000440, +0.000711].
- Paired MRR difference: +0.000886, 95% CI [+0.000235, +0.001537].
- Paired DVR difference: -0.005999, 95% CI [-0.006539, -0.005458].
- Paired MED difference: -0.001240, 95% CI [-0.001305, -0.001175].
- Paired squared-risk difference: -0.000272, 95% CI [-0.000285, -0.000259].
- **INFERENCE — VALIDATION ONLY:** post-hoc provides consistently lower overchallenge exposure, integrated retains higher Recall, and NDCG is not clearly separated by the paired interval. These are not final/test claims.
- All five integrated runs selected epoch 100; the cap remains unchanged.
- Trace: `runs/stage3/STAGE3_MULTI_SEED_VALIDATION.json` and `runs/stage3/audits/STAGE3_MULTI_SEED_AUDIT.json`.

## Stage 3 τ-sensitivity freeze — 2026-08-28

- Status: frozen before execution; multi-epoch sensitivity runs not executed by the preparation step.
- τ values: 0, 0.05, 0.1, and 0.2; five seeds; integrated λd=0.03; post-hoc γ=5; T=0.2.
- Completed τ=0.1 runs are reused. New budget: 15 integrated trainings and 15 post-hoc evaluations.
- Every result reports native-τ risk metrics plus fixed-anchor τ_eval=0.1 metrics to separate ranking changes from threshold-definition changes.
- Sensitivity cannot revise τ, λd, γ, or any frozen training/evaluation choice. Test targets remain closed.
- Manifest: `stages/stage3_proposed/configs/tau_sensitivity_manifest.json`.

## Stage 3 τ sensitivity — 2026-08-28

- Status: completed and audited; τ `{0, 0.05, 0.1, 0.2}`, five seeds, integrated λd=0.03 and post-hoc γ=5.
- Audit scope: 30 new results plus 10 reused τ=0.1 results. All configs, seed mappings, checkpoint/source hashes, exposure counts, and fixed-anchor outputs passed; no test or `final_*` access.
- Native DVR means for integrated across increasing τ: 0.510943, 0.347173, 0.183588, 0.072580. Post-hoc: 0.499338, 0.338119, 0.177589, 0.071109.
- Fixed-anchor τ_eval=0.1 DVR means for integrated: 0.180369, 0.182178, 0.183588, 0.185206. Post-hoc: 0.162182, 0.170876, 0.177589, 0.183959.
- Fixed-anchor NDCG means for integrated: 0.282073, 0.283253, 0.284372, 0.285926. Post-hoc: 0.281769, 0.283346, 0.284507, 0.285678.
- **INFERENCE — VALIDATION ONLY:** larger τ improves relevance while making ranked lists less conservative under a common τ=0.1 definition. The sharp native-risk decline is partly mechanical and must not be presented alone.
- Mean integrated runtime for the three new τ groups was approximately 294–309 seconds per run.
- Frozen τ=0.1 and selected coefficients remain unchanged.
- Trace: `runs/stage3/STAGE3_TAU_SENSITIVITY.json` and `runs/stage3/audits/STAGE3_TAU_SENSITIVITY_AUDIT.json`.

## Stage 3 risk-form ablation freeze — 2026-08-28

- Status: frozen before execution; ten new integrated trainings remain.
- Reference: asymmetric-squared, τ=0.1, T=0.2, λd=0.03, five existing selected runs.
- Ablations: asymmetric-linear and symmetric-squared, each on the same five seeds.
- Target-free unseen-candidate means were 0.01275518 (asymmetric-squared), 0.05411641 (linear), and 0.01435418 (symmetric-squared).
- Frozen scale factors are 0.2356989647 for linear and 0.8886040153 for symmetric-squared.
- Reporting includes relevance, frozen overchallenge metrics, paired 95% CIs, and runtime. Test targets remain closed.
- Manifest: `stages/stage3_proposed/configs/risk_ablation_manifest.json`.

## Stage 3 risk-form ablations — 2026-08-28

- Status: passed audit; ten new validation runs plus five reused asymmetric-squared reference runs, using five matched seeds.
- All ten new checkpoints matched their reported SHA-256 values; configurations matched the frozen OFAT manifest; independently recomputed means and sample SDs matched exactly.
- Asymmetric-linear minus asymmetric-squared: Recall +0.000072 (95% CI [-0.000588, +0.000732]); NDCG -0.000709 ([-0.001197, -0.000220]); MRR -0.000968 ([-0.001543, -0.000394]); DVR -0.001168 ([-0.001338, -0.000998]); squared risk -0.0000183 ([-0.0000220, -0.0000146]).
- Symmetric-squared minus asymmetric-squared: Recall +0.000072 (95% CI [-0.000112, +0.000255]); NDCG +0.000349 ([+0.000193, +0.000505]); MRR +0.000438 ([+0.000227, +0.000649]); DVR +0.000221 ([+0.000135, +0.000308]); squared risk +0.00000489 ([+0.00000347, +0.00000632]).
- **INFERENCE — VALIDATION ONLY:** linear is more conservative but loses ranking quality; symmetric slightly improves ranking quality while worsening overchallenge risk. Neither universally dominates the frozen asymmetric-squared formulation.
- Selected settings remain unchanged. No test or `final_*` artifact was accessed.
- Trace: `runs/stage3/STAGE3_RISK_ABLATIONS.json` and `runs/stage3/audits/STAGE3_RISK_ABLATIONS_AUDIT.json`.

## Final experiment protocol freeze — 2026-08-28

- Stage 3 development is complete and the final protocol is frozen before runner execution.
- Eligible final methods: Popularity, BPR-MF, LightGCN, XSimGCL, integrated asymmetric-squared LightGCN, and post-hoc asymmetric-squared LightGCN.
- Training uses the final prefix for exactly 100 epochs without validation or early stopping; each stochastic method uses the five established seeds.
- Test evaluation is permitted only after a checkpoint is saved and only once per eligible model seed. No sensitivity or tuning cell is eligible.
- The 12 hashes from the earlier footer/byte-level structural audit are frozen as a pre-test drift guard. The absent original Stage 1 manifest remains an explicit provenance limitation.
- No final training or test evaluation was performed by this freeze step.
- Manifest: `stages/stage4_final/configs/final_protocol_manifest.json`.

## Final-access safeguard implementation — 2026-08-28

- Added an isolated final-prefix loader with no unguarded targets method.
- Test targets require a non-empty persisted checkpoint plus a matching receipt recording exactly 100 completed epochs, checkpoint SHA-256, `test_accessed=false`, and state `TRAINING_COMPLETE_TEST_NOT_ACCESSED`.
- Failed authorization does not append `test_targets.parquet` to the loader access ledger.
- Sixteen automated Stage 2–4 tests passed. No final training or test evaluation was performed.
- Target-free final training infrastructure now prepares 20 frozen configurations and enforces exactly 100 epochs without validation or target loading. A resumable family runner was added; 18 automated tests pass. No final training was executed by Codex.
- Final training completed externally: 20/20 receipts passed the 100-epoch, checkpoint-hash, non-target-access, and global-barrier audit. The one-time batch evaluator and result summarizer were implemented but not executed; test targets remain unopened.
- Final evaluator preflight passed with 20 receipts, no prior lock/result, and `test_accessed=false`; 18 automated tests passed. Trace: `runs/stage4/FINAL_PREEXECUTION_AUDIT.json`.

## One-time final test evaluation — 2026-08-28

- Status: completed once and locked; 26 eligible outputs, including five matched seeds for every stochastic family and one deterministic Popularity result.
- Test access ledger contains one `test_targets.parquet` occurrence. All results share 22,241 evaluable users, zero cold targets, mean 248.606807 candidates, 264 catalog items, and 222,410 Top-10 exposures.
- Recomputed means and sample SDs match `STAGE4_FINAL_SUMMARY.json` exactly.
- Integrated minus LightGCN: Recall +0.000180 (95% CI [-0.000530, +0.000889]); NDCG -0.000874 ([-0.001289, -0.000459]); DVR -0.001103 ([-0.001265, -0.000942]); squared risk -0.0000510 ([-0.0000591, -0.0000430]).
- Post-hoc minus integrated: Recall -0.002392 (95% CI [-0.003077, -0.001707]); NDCG -0.000679 ([-0.001085, -0.000274]); DVR -0.005960 ([-0.006391, -0.005529]); squared risk -0.000304 ([-0.000323, -0.000285]).
- **INFERENCE — FINAL:** integrated control provides a modest accuracy-risk trade-off relative to LightGCN; post-hoc is more conservative but sacrifices more Recall/NDCG. No method universally dominates.
- Trace: `runs/stage4/final/STAGE4_FINAL_RESULTS.json`, `STAGE4_FINAL_SUMMARY.json`, and `STAGE4_FINAL_AUDIT.json`.

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
- **DECISION — REVISED/LOCKED:** the matched coefficient budget and selection rule are frozen in `stages/stage3_proposed/configs/bounded_tuning_manifest.json`; no configuration may be added after results are observed.
- **DECISION — REVISED/LOCKED:** DVR@10 is violation exposure share; MED@10 is unconditional mean positive excess over all Top-10 exposures. These definitions supersede their previous OPEN status.
- **FACT — VERIFIED:** target-free scale audit found unseen-score SD 2.525652 and positive-risk median/p90 0.018004/0.104013 at τ=0.1. This supports replacing the ineffective γ=0.1 probe with the frozen range 1–20 without using validation outcomes.

## 2026-08-28 — Stage 3 bounded selection

- **DECISION — COMPLETE:** the frozen bounded budget completed with all eight expected validation results and no forbidden/test artifact access.
- **DECISION — LOCKED:** integrated selection is `DRLGCN_BOUND_001`, λd=0.03, under the primary 1% NDCG guardrail.
- **DECISION — LOCKED:** post-hoc selection is `POSTHOC_BOUND_002`, γ=5, under the same guardrail.
- **FACT — DIAGNOSTIC:** at seed 20260827, selected post-hoc has higher NDCG/MRR and lower DVR/MED/squared risk than selected integrated, while integrated has higher Recall. This is not yet a multi-seed conclusion.
- **DECISION — REJECTED:** no bounded coefficient grid expansion and no epoch-cap extension are permitted after observing these results.
- **DECISION — OPEN:** five-seed validation of both selected variants remains required before τ sensitivity and before any final/test access.
- **DECISION — REVISED/LOCKED:** five-seed validation manifest is frozen at `stages/stage3_proposed/configs/multiseed_validation_manifest.json`; seed 20260827 is reused and four new seeds per selected variant remain to execute.
- **DECISION — LOCKED:** paired comparison direction is post-hoc minus integrated. Mean±sample SD and two-sided paired-t 95% CI are the predeclared summaries; these results cannot revise selected coefficients.

## 2026-08-28 — Stage 3 five-seed validation

- **DECISION — COMPLETE:** selected integrated λd=0.03 and post-hoc γ=5 completed validation on all five predeclared matched seeds.
- **FACT — VALIDATION:** post-hoc has lower DVR, MED, and squared risk on all five seeds; the paired 95% CIs for post-hoc-minus-integrated exclude zero.
- **FACT — VALIDATION:** integrated has higher Recall on all five seeds; its paired difference interval excludes zero.
- **FACT — VALIDATION:** the NDCG difference is small and its paired interval includes zero. Post-hoc MRR is higher on all five seeds and its paired interval excludes zero.
- **DECISION — LOCKED:** these findings do not reopen λd or γ selection and are not final/test results.
- **DECISION — OPEN:** frozen τ sensitivity remains required before complete Stage 3 design freeze and any test access.
- **DECISION — REVISED/LOCKED:** τ sensitivity is frozen in `tau_sensitivity_manifest.json` for four τ values and five seeds, with native and fixed-anchor τ_eval=0.1 reporting. It is analysis-only and cannot revise selected settings.

## 2026-08-28 — Stage 3 τ sensitivity

- **DECISION — COMPLETE:** τ sensitivity completed for τ `{0, 0.05, 0.1, 0.2}`, both selected variants, and all five seeds; 30 new runs and 10 reused τ=0.1 runs passed audit.
- **FACT — VALIDATION:** native DVR/MED/risk decrease sharply with larger τ, but fixed-anchor τ_eval=0.1 results show anchor risk increases with τ. The native trend is therefore partly definitional rather than solely a recommendation improvement.
- **FACT — VALIDATION:** smaller τ produces more conservative ranked lists under the common anchor, with a relevance cost that is strongest for post-hoc Recall and integrated NDCG/MRR.
- **DECISION — LOCKED:** sensitivity does not revise τ=0.1, λd=0.03, or γ=5.
- **DECISION — OPEN:** asymmetric-vs-symmetric and linear-vs-squared ablations remain required before Stage 3 design freeze can close.
- **DECISION — REVISED/LOCKED:** risk-form ablations are frozen as two OFAT five-seed comparisons with target-free mean-risk scale matching; no symmetric-linear factorial expansion is permitted.
- **DECISION — COMPLETE:** both frozen risk-form ablations completed across all five matched validation seeds and passed configuration, aggregation, checkpoint-hash, and leakage audit.
- **FACT — VALIDATION:** asymmetric-linear reduces DVR, MED, and squared risk but lowers NDCG and MRR relative to asymmetric-squared; Recall is not clearly separated.
- **FACT — VALIDATION:** symmetric-squared slightly raises NDCG and MRR but also raises DVR, MED, and squared risk relative to asymmetric-squared; Recall is not clearly separated.
- **DECISION — LOCKED:** asymmetric-squared remains selected. Ablation results cannot trigger reselection, and the one-sided form matches the stated overchallenge construct.
- **STAGE STATUS — COMPLETE:** Stage 3 development, bounded selection, five-seed validation, τ sensitivity, and risk-form ablations are complete. The next gate is a frozen final-experiment and one-time test protocol; test targets remain unopened.

## 2026-08-28 — Final experiment freeze

- **DECISION — LOCKED:** final eligible methods, five-seed policy, 100-epoch fixed training budget, final-prefix inputs, candidate set, metrics, comparisons, and one-time test access are frozen in `stages/stage4_final/configs/final_protocol_manifest.json`.
- **DECISION — LOCKED:** test targets may be read only after training and checkpoint persistence for a model seed. Test evaluation is single-use and cannot trigger retraining or any analytical choice.
- **DECISION — REJECTED:** final test evaluation of tuning cells, τ-sensitivity cells, risk-form ablations, or a duplicate λd=0 model.
- **DECISION — REVISED/LOCKED:** the complete hash set from the earlier structural audit is now the immutable pre-test byte snapshot. This controls drift but does not repair or replace the missing original Stage 1 handoff manifest; that provenance limitation remains reportable.
- **EXECUTION STATUS:** final runner implementation and safeguard audit remain pending; test access has not been authorized or performed.
- **FACT — VERIFIED:** the isolated final loader has no generic targets method and rejects test access unless a persisted checkpoint and matching 100-epoch completion receipt are present. Safeguard tests pass; test execution remains disabled pending the complete runner and summarizer.
- **FINAL GATE — PASS/READY:** all 20 fixed-epoch training runs passed receipt, checkpoint-hash, epoch-count, and no-target-access checks. Evaluator preflight passed without test access; the frozen one-time batch evaluation is now eligible for user execution.
- **FINAL STATUS — COMPLETE/LOCKED:** the one-time test evaluation completed for all 26 eligible outputs and passed lock, access-ledger, seed, candidate, exposure, aggregation, and paired-comparison audit. No rerun, reselection, or retraining is permitted.
- **FACT — FINAL:** integrated asymmetric-squared regularization gives a small but consistent reduction in DVR, MED, and squared risk versus LightGCN, with lower NDCG and MRR and no clearly separated Recall difference.
- **FACT — FINAL:** post-hoc reranking gives a larger risk reduction than integrated, while integrated retains higher Recall and NDCG; their MRR difference is not clearly separated.
- **FACT — FINAL:** BPR-MF has the strongest mean Recall/NDCG, whereas XSimGCL has the strongest mean MRR and the highest overchallenge metrics among learned models.
- **CLAIM GUARD — LOCKED:** results support a relevance-overchallenge trade-off and implementation-level integrated control; they do not support universal superiority, causal learning benefit, or learning-outcome claims.

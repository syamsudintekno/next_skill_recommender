# DESIGN_FREEZE

## Stage 1 decisions carried into Stage 2

- **LOCKED:** task is Top-K next newly encountered skill recommendation.
- **LOCKED:** learner–skill graph edges represent binary observed exposure, not negative/positive preference.
- **LOCKED:** development evaluation uses the training-visible catalog minus each learner's seen-prefix skills, with full ranking and no sampled negatives.
- **LOCKED:** primary K is 10; validation cold targets are reported but excluded from relevance denominators.
- **LOCKED:** development model selection must not access final/test artifacts.
- **OPEN:** Stage 1 complete 12-file manifest is required to close hash gate S2-A01. Three development hashes embedded in the Stage 1 proxy-audit runner have matched (partial pass only).

This file records the Stage 2 carry-forward only; it does not revise any frozen Stage 1 methodological choice.

## Stage 3 development selection freeze — 2026-08-27

- **LOCKED:** DVR@10 is the proportion of all Top-10 recommendation exposures with `d_i > a_u + τ`, averaged over all development learners.
- **LOCKED:** MED@10 is the mean positive excess `max(0, d_i - a_u - τ)` over all Top-10 recommendation exposures; it is not conditioned on violations.
- **LOCKED:** squared risk@10 is the corresponding mean squared positive excess.
- **LOCKED:** item exposure distribution is the complete vector of Top-10 counts in ascending lexical skill-ID order, with min/median/max summaries.
- **LOCKED:** coefficient selection uses seed 20260827, τ=0.1, T=0.2, the inherited LightGCN/evaluator/candidate contract, and exactly four configurations per method family.
- **LOCKED:** integrated λd candidates are 0.03, 0.1, 0.3, and 1.0. Post-hoc γ candidates are 1, 5, 10, and 20 and reuse the frozen λd=0 checkpoint.
- **LOCKED:** first restrict to the NDCG–DVR non-dominated set. Among points with relative validation NDCG loss no greater than 1% versus λd=0, select minimum DVR; ties prefer higher NDCG, lower MED, then the smaller coefficient.
- **LOCKED:** if no point satisfies the 1% guardrail, choose the smallest NDCG loss among points with strictly lower DVR; reject a variant if none lowers DVR.
- **LOCKED:** after coefficient selection, run the selected configurations on seeds 20260827–20260831. τ sensitivity `{0, 0.05, 0.1, 0.2}` is analysis-only and cannot revise coefficient selection.
- **FIREWALL:** test targets remain closed until bounded selection, multi-seed validation, τ sensitivity, and complete Stage 3 design freeze are finished.

## Stage 3 five-seed validation freeze — 2026-08-28

- **LOCKED:** compare selected integrated λd=0.03 and selected post-hoc γ=5 on seeds 20260827–20260831, with τ=0.1 and T=0.2 unchanged.
- **LOCKED:** seed 20260827 reuses `DRLGCN_BOUND_001` and `POSTHOC_BOUND_002`; these accepted selection runs must not be retrained.
- **LOCKED:** post-hoc seeds 20260828–20260831 reuse the corresponding selected Stage 2 LightGCN checkpoints, whose SHA-256 values are frozen in `multiseed_validation_manifest.json`.
- **LOCKED:** report mean and sample SD per method. Report paired per-seed differences as post-hoc minus integrated with two-sided 95% paired-t confidence intervals.
- **LOCKED:** multi-seed validation cannot change λd, γ, τ, T, architecture, optimizer budget, candidate set, evaluator, or epoch cap.
- **FIREWALL:** no test or `final_*` access is permitted. τ sensitivity begins only after this five-seed validation is complete.

## Stage 3 τ-sensitivity freeze — 2026-08-28

- **LOCKED:** τ values are `{0, 0.05, 0.1, 0.2}` on all five seeds for integrated λd=0.03 and post-hoc γ=5; T remains 0.2.
- **LOCKED:** τ=0.1 reuses the completed five-seed results. The other three values require 15 integrated trainings and 15 post-hoc evaluations.
- **LOCKED:** each run reports native-τ DVR/MED/squared risk and fixed-anchor metrics at τ_eval=0.1 on the same ranked lists.
- **RATIONALE — LOCKED:** fixed-anchor reporting prevents the mechanical decrease caused solely by relaxing the violation definition from being misread as a recommendation improvement.
- **LOCKED:** report mean±sample SD and matched-seed post-hoc-minus-integrated paired-t 95% CIs for both views.
- **LOCKED:** this is analysis-only. It cannot select a preferred τ or revise λd, γ, model architecture, candidates, evaluator, or epoch cap.
- **FIREWALL:** test targets remain closed until τ sensitivity is audited and Stage 3 is completely frozen.

## Stage 3 τ-sensitivity completion — 2026-08-28

- **COMPLETE:** all 30 new runs and 10 reused τ=0.1 run-views passed manifest, hash, exposure, summary, and no-test-access audits.
- **LOCKED:** native-threshold results must always be accompanied by fixed-anchor τ_eval=0.1 results when interpreting τ sensitivity.
- **LOCKED:** τ=0.1 remains the frozen setting; the analysis-only sweep does not reopen selection.
- **OPEN:** Stage 3 remains incomplete pending asymmetric-vs-symmetric and linear-vs-squared ablations plus their computational-cost audit.

## Stage 3 risk-form ablation freeze — 2026-08-28

- **LOCKED:** use OFAT comparisons against the selected asymmetric-squared reference; do not expand to a symmetric-linear factorial cell.
- **LOCKED:** asymmetric-linear is `max(0, d_i-a_u-τ)` and symmetric-squared is `max(0, |d_i-a_u|-τ)^2`.
- **LOCKED:** τ=0.1, T=0.2, λd=0.03, architecture, optimizer budget, candidates, evaluator, and seeds 20260827–20260831 remain unchanged.
- **LOCKED:** target-free unseen-candidate mean-risk matching scales asymmetric-linear by 0.2356989647 and symmetric-squared by 0.8886040153 relative to asymmetric-squared.
- **RATIONALE — LOCKED:** scale matching prevents the ablation from merely comparing different numerical penalty magnitudes.
- **LOCKED:** report mean±sample SD, paired ablation-minus-reference 95% CIs, and runtime. Ablations cannot revise selected settings.
- **FIREWALL:** test and `final_*` artifacts remain closed until ablations and Stage 3 closure audit complete.

## Final experiment protocol freeze — 2026-08-28

- **LOCKED:** final training uses the frozen final prefix and exactly 100 epochs without validation evaluation, checkpoint selection, or early stopping.
- **LOCKED:** eligible outputs are Popularity, BPR-MF, LightGCN, XSimGCL, integrated asymmetric-squared LightGCN, and asymmetric-squared post-hoc LightGCN. Sensitivity, risk-form ablation, and rejected tuning cells do not receive test evaluation.
- **LOCKED:** stochastic models use seeds 20260827–20260831. Popularity is deterministic and evaluated once.
- **LOCKED:** post-hoc reuses the same-seed final LightGCN checkpoint. A separate final λd=0 run is prohibited because it is equivalent to LightGCN.
- **LOCKED:** the candidate set is the final catalog minus each learner's final-prefix seen skills, using full ranking at K=10 and the frozen cold-target rule.
- **LOCKED:** each model-seed checkpoint is saved before `test_targets.parquet` is opened, then evaluated exactly once. Test outcomes cannot cause retries, reselection, or configuration changes.
- **LOCKED:** report all frozen relevance and pedagogy metrics, complete exposure distributions, runtime, mean±sample SD, and matched-seed paired-t 95% CIs.
- **PROVENANCE LIMITATION:** the original complete Stage 1 manifest is unavailable. The 12 hashes recorded earlier in `S2_1_ARTIFACT_AUDIT.json` are frozen as the immutable pre-test snapshot, but are not represented as the missing original manifest.
- Manifest: `stages/stage4_final/configs/final_protocol_manifest.json`.

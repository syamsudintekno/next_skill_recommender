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

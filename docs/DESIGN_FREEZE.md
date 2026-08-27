# DESIGN_FREEZE

## Stage 1 decisions carried into Stage 2

- **LOCKED:** task is Top-K next newly encountered skill recommendation.
- **LOCKED:** learner–skill graph edges represent binary observed exposure, not negative/positive preference.
- **LOCKED:** development evaluation uses the training-visible catalog minus each learner's seen-prefix skills, with full ranking and no sampled negatives.
- **LOCKED:** primary K is 10; validation cold targets are reported but excluded from relevance denominators.
- **LOCKED:** development model selection must not access final/test artifacts.
- **OPEN:** Stage 1 complete 12-file manifest is required to close hash gate S2-A01. Three development hashes embedded in the Stage 1 proxy-audit runner have matched (partial pass only).

This file records the Stage 2 carry-forward only; it does not revise any frozen Stage 1 methodological choice.

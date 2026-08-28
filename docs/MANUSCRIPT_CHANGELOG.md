# MANUSCRIPT_CHANGELOG

## 2026-08-27

- No manuscript text or numerical claim changed.
- Stage 2 diagnostic infrastructure and Popularity validation result were recorded; this is not yet a manuscript baseline result.
- LightGCN diagnostic runs were added to the experiment ledger only. Rejected full-batch runs must not be reported as manuscript baseline results.
- Stage 3 loader/objective audit and runner infrastructure were added; no manuscript result or claim was changed.
- Three Stage 3 single-seed validation probes were logged as development evidence only; no manuscript table, result claim, or test result was added.
- Frozen Stage 3 bounded selection was completed and logged as validation-only evidence; no manuscript result claim or test result was added.
- Five-seed Stage 3 validation protocol and reporting contract were frozen; no new numerical or manuscript claim was added.
- Five-seed selected-variant validation results were logged as development evidence; no test result or final manuscript claim was added.
- Five-seed τ-sensitivity protocol, including fixed-anchor risk reporting, was frozen; no sensitivity result or manuscript claim was added.
- Audited τ-sensitivity results were logged as validation-only evidence; no test result or final manuscript claim was added.
- Risk-form ablation protocol was frozen with target-free scale matching; no ablation result or manuscript claim was added.
- Audited five-seed risk-form ablation results were logged as validation-only evidence; no test result or final manuscript claim was added.
- Final experiment and one-time test protocol was frozen; no final training, test result, or manuscript claim was added.
- One-time final test results were audited and locked. The manuscript may now use the final relevance-risk trade-off evidence, subject to the recorded non-causal and non-superiority claim guards.
- Added the first English Results draft at `manuscript/RESULTS.md`, covering the locked final table, paired RQ1–RQ2 comparisons, validation-only λd/τ sensitivity, and risk-form ablations. No new numerical result or citation was introduced.
- Added a deterministic exporter and generated table source at `manuscript/generated/RESULTS_TABLES.md`; the manuscript tables are now traceable directly to the locked Stage 4 summary.
- Added the first English Method and Experimental Setup draft at `manuscript/METHOD_AND_EXPERIMENTAL_SETUP.md`, grounded in the frozen split builder, proxy implementation, model configs, evaluator, and final protocol. The missing original Stage 1 handoff manifest remains explicit.
- Added the first English Discussion and Limitations draft at `manuscript/DISCUSSION_AND_LIMITATIONS.md`. Factual findings are separated from untested explanations, and the draft explicitly covers proxy validity, implicit-feedback semantics, statistical uncertainty, scalability, generalizability, provenance, and non-causal claim limits.
- Added a primary-source-grounded Related Work draft and `docs/NOVELTY_MATRIX.md`. The novelty statement is deliberately bounded to the task–objective–evaluation combination; “first difficulty-aware” and learning-path equivalence remain rejected.
- Added the first English Introduction draft at `manuscript/INTRODUCTION.md`. It states the three frozen research questions and three evidence-bounded contributions, describes the result as a relevance–overchallenge trade-off, and excludes priority, causal, learning-outcome, and learning-path claims.
- Added the first English Conclusion draft at `manuscript/CONCLUSION.md`. It answers RQ1–RQ3 from the locked final and validation-only evidence, preserves the non-dominance interpretation, and limits future-work claims to proxy validation, external replication, scalability, and prospective evaluation.
- Added the first English Abstract and keywords draft at `manuscript/ABSTRACT.md`. Every reported quantity is traced to the locked Stage 4 summary; validation-only sensitivity is labeled, and the closing claim is restricted to offline exposure trade-offs.
- Added verified drafting references, a deterministic full-manuscript builder and generated `manuscript/FULL_MANUSCRIPT_DRAFT.md`, plus `docs/MANUSCRIPT_INTEGRATION_AUDIT.md`. The audit separates completed scientific drafting from unresolved dataset provenance, literature coverage, author metadata, availability statements, JOIN formatting, and figure work.

# MANUSCRIPT_INTEGRATION_AUDIT

Audit date: 2026-08-29

## Status

- **COMPLETE — DRAFT ASSEMBLY:** Abstract, Introduction, Related Work, Method
  and Experimental Setup, Results, Discussion and Limitations, Conclusion, and
  References have source files and a deterministic assembly order.
- **NOT SUBMISSION-READY:** unresolved provenance, literature-coverage, author,
  and journal-format items remain below.

## Cross-section checks

- **PASS:** the title is consistent with `README.md` and the frozen project
  scope.
- **PASS:** RQ1–RQ3 in the Introduction match the Results and Conclusion
  structure: integrated-vs-LightGCN effect, integrated-vs-post-hoc comparison,
  and \(\lambda_d\)/\(\tau\) sensitivity.
- **PASS:** the recommendation unit is consistently a next newly encountered
  skill and the output is Top-K, not a sequential learning path.
- **PASS:** exposure, correctness, preference, mastery, and learning outcome are
  not treated as interchangeable constructs.
- **PASS:** final numerical claims trace to
  `runs/stage4/final/STAGE4_FINAL_SUMMARY.json`; sensitivity and risk-form
  findings are labeled as validation-only.
- **PASS:** integrated control is described as a modest relevance–risk
  trade-off. The manuscript does not claim universal superiority over post-hoc
  reranking.
- **PASS:** difficulty and ability are described as cohort-dependent behavioral
  proxies, not intrinsic or ground-truth quantities.
- **PASS:** the manuscript does not claim causal learning benefit, improved
  learning outcomes, a hard safety constraint, or JOIN acceptance.
- **PASS:** Table 1 is the split summary, Table 2 is the final model comparison,
  and Table 3 is the paired integrated/post-hoc comparison.

## Submission blockers

1. **BLOCKED — dataset provenance:** recover the official ASSISTments download
   URL, exact public dataset record/variant, access date, and primary dataset
   citation. Remove the drafting note in the Method only after verification.
2. **OPEN — literature coverage:** reconcile the targeted primary-source
   comparison with the author's SLR and perform the broader database search
   described in `docs/NOVELTY_MATRIX.md`. No priority claim is allowed in the
   meantime.
3. **OPEN — author metadata:** supply the final author order, affiliations,
   corresponding-author details, acknowledgments/funding, conflict-of-interest
   statement, and any required ethics statement.
4. **OPEN — availability statements:** decide the repository/archive URL and
   write data-availability and code-availability statements that accurately
   reflect redistribution constraints.
5. **OPEN — journal formatting:** apply the current JOIN template, section and
   abstract limits, reference style, figure/table placement rules, and required
   declarations only after checking the official author guidelines.
6. **COMPLETE — result visualization:** the NDCG–DVR operating-point figure and
   exposure-concentration figure are generated from locked Stage 4 summary and
   result artifacts. Their source hashes and derived values are recorded in
   `manuscript/generated/FIGURE_DATA.json`; no target, training, evaluation, or
   reselection path is used.

## Reportable reproducibility limitation

- The original complete 12-file Stage 1 handoff manifest is unavailable. The
  pre-test snapshot hashes control subsequent artifact drift but cannot prove
  equality to that missing original handoff. This is already disclosed in the
  Method and Discussion and must not be silently marked as repaired.

## Build contract

- Edit the eight section sources under `manuscript/`.
- Run `python manuscript/generate_figures.py` after changing a locked figure
  source or plotting code. It must read only the two recorded Stage 4 outputs.
- Run `powershell -ExecutionPolicy Bypass -File
  manuscript/build_full_manuscript.ps1` from the project root.
- Treat `manuscript/FULL_MANUSCRIPT_DRAFT.md` as generated output.

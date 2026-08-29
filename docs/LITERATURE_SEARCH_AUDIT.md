# LITERATURE_SEARCH_AUDIT

Audit date: 2026-08-29

## Purpose and boundary

This audit tests the manuscript's bounded novelty positioning. It does not
replace the author's systematic literature review, constitute a new PRISMA
review, or establish a priority claim. The comparison unit is the combination
of task, representation, pedagogical control, control location, and evaluation.

## Inputs

- The author's SLR covering 84 studies published from 2019 through 2025.
- The supplied full text of Zhang et al. (KDD 2024), DOI
  `10.1145/3637528.3671947`.
- Primary publisher records and available author/arXiv full texts checked on
  2026-08-29.

## Search concepts

The targeted update combined the following concept blocks:

1. `educational`, `exercise`, `skill`, `learning path`, `recommendation`;
2. `difficulty`, `ability`, `challenge`, `cognitive diagnosis`, `pedagogical
   constraint`;
3. `LightGCN`, `graph collaborative filtering`, `knowledge graph`;
4. `multi-objective`, `regularization`, `constraint`, `filtering`, `reranking`.

Queries were applied to ACM Digital Library, IEEE Xplore, journal-publisher
pages, and author/arXiv versions of accepted papers. The final indexed-database
refresh must additionally preserve Scopus and Web of Science exports.

## Reconciliation with the author's SLR

| SLR evidence | Interpretation for this manuscript |
|---|---|
| 84 studies, 2019--2025 | Provides broad field coverage but predates the complete 2026 update. |
| 44.0% post-hoc filtering and 33.3% feature engineering | Motivates testing where a pedagogical signal enters the pipeline. |
| 2.4% formal MCDM/constraint optimization | Shows limited formal multi-criteria use under the SLR coding scheme, not absence of multi-objective recommendation. |
| Accuracy-centric evaluation remains common | Supports joint relevance/risk reporting, but not a claim that no prior work measured pedagogical properties. |

The SLR and this audit answer different questions. The SLR codes broad
pedagogical integration patterns; this audit checks whether close papers share
the manuscript's specific task--objective--evaluation combination.

## Closest-work decisions

| Work | Include? | Evidence-based reason |
|---|---|---|
| Huang et al. (2019), DRE | Yes | Multi-objective adaptive exercise recommendation includes difficulty smoothness; direct precedent against broad novelty claims. |
| Du et al. (2022) | Yes | Cognitive diagnosis explicitly relates learner ability and exercise difficulty. |
| Yan et al. (2023) | Yes | Educational knowledge graphs are combined with difficulty/diversity/novelty filtering. |
| Yang et al. (2023), PEGA | Yes | Cognitive-diagnosis-informed constrained multi-objective exercise-group assembly. |
| Zhang et al. (2024), DLPR | Yes | Difficulty-driven hierarchical RL generates sequential learning paths. |
| Cheng et al. (2025), NR4DER | Yes | Appropriate-difficulty candidate filtering precedes neural reranking; closest post-hoc precedent found. |
| Zhu et al. (2026) | Yes | Deep knowledge tracing and DRL use a difficulty-adaptive ability-alignment constraint. |
| Zhang et al. (2026), LT-MKT | Context only | Models cognitive load and transfer for knowledge tracing, not ranked recommendation. |

## Resulting claim boundary

- **Supported:** evaluation of a training-integrated asymmetric expected-
  overchallenge term in LightGCN, compared with its matched score-level form,
  for temporal full-ranking next-new-skill recommendation.
- **Not supported:** any “first,” field-wide absence, universal superiority,
  hard-safety, or learning-outcome claim.
- **Residual uncertainty:** publications outside the targeted sources, papers
  indexed after 2026-08-29, and terminology variants not captured by these
  concept blocks.

## Required pre-submission refresh

1. Run the concept blocks in Scopus and Web of Science, plus ACM and IEEE.
2. Export complete query strings, timestamps, result counts, and RIS/BibTeX.
3. Screen titles/abstracts for an identical LightGCN + asymmetric
   overchallenge + integrated/post-hoc comparison.
4. Update `docs/NOVELTY_MATRIX.md`, Related Work, and References.
5. Keep combination wording even if no identical paper is found.

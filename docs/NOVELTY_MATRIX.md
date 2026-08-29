# NOVELTY_MATRIX

Status: targeted primary-source comparison, reconciled on 2026-08-29 with the
author's 2019--2025 SLR and a targeted update through 2026. This is not an
exhaustive systematic-search claim and must be refreshed before submission.

| Work | Primary task | Representation / optimizer | Difficulty or pedagogical role | Control location | Distinction from this study |
|---|---|---|---|---|---|
| Rendle et al., BPR (2009) | Personalized ranking from implicit feedback | Latent pairwise ranking; instantiated with MF | None | Training objective | Supplies the relevance loss, but has no learner-specific overchallenge objective. |
| Huang et al., DRE (2019) | Adaptive exercise recommendation | Recurrent Q-networks and deep reinforcement learning | Multi-objective reward includes Review and Explore goals, difficulty smoothness, and engagement | Sequential policy learning | Directly establishes multi-objective difficulty-aware recommendation, but not graph CF, a full-catalog Top-K objective, or one-sided excess-risk exposure. |
| He et al., LightGCN (2020) | Top-K collaborative filtering | Linear propagation on a user--item bipartite graph | None | BPR-trained graph embeddings | Supplies the graph backbone; no educational difficulty signal is integrated. |
| Gong et al., ACKRec (2020) | MOOC knowledge-concept recommendation | Heterogeneous graph, meta-path propagation, and attention | Learner interest/context rather than asymmetric empirical-difficulty risk | End-to-end heterogeneous GNN | Educational graph recommendation, but not the same learner--skill graph, risk construct, or candidate-softmax regularizer. |
| Du et al. (2022) | Personalized exercise recommendation | Cognitive-diagnosis-based recommendation | Student ability and exercise difficulty inform exercise selection | Recommendation algorithm | Ability--difficulty matching predates this study; the work does not use LightGCN or optimize expected one-sided overchallenge over the unseen catalog. |
| Yan et al. (2023) | Personalized exercise recommendation | Deep knowledge tracing, course knowledge graph, and learner knowledge-structure graph | Difficulty, diversity, and novelty filter candidate exercises | Candidate generation/filtering | Combines educational graphs and difficulty-aware filtering, but not graph collaborative filtering or matched integrated-versus-score-level risk control. |
| Yang et al., PEGA (2023) | Personalized exercise-group assembly | Cognitive diagnosis plus constrained multi-objective evolutionary optimization | Ability estimates and pedagogical objectives constrain group assembly | Set-level evolutionary search | Optimizes exercise groups rather than learned full-ranking scores and does not use the asymmetric excess-risk expectation studied here. |
| Yu et al., XSimGCL (2023) | Graph contrastive recommendation | LightGCN-style propagation with noise-based views and cross-layer contrast | None in the cited formulation | BPR plus InfoNCE | Strengthens relevance learning but does not optimize learner-specific overchallenge. |
| Zhang et al., DLPR (2024) | Step-by-step learning-path recommendation | Hierarchical learning/practice-item graph and hierarchical reinforcement learning | Difficulty-aware state and decisions | Sequential policy learning | Differs in task, action space, feedback, output, and simulator-based evaluation. |
| Cheng et al., NR4DER (2025) | Diversified exercise recommendation | Mastery prediction, difficulty-based candidate filtering, and neural reranking | Filters exercises to appropriate difficulty and models learning-pace diversity | Candidate filter plus post-hoc neural reranker | Particularly close operationally, but targets exercise sequences/diversity and does not test a LightGCN training-integrated asymmetric risk term against its matched score-level form. |
| Zhu et al. (2026) | Personalized exercise recommendation | Deep knowledge tracing plus deep reinforcement learning | Difficulty-adaptive constraint aligns exercises with estimated learner ability | Sequential policy learning | Confirms recent ability-aligned control, but not graph CF or the same Top-K exposure-risk formulation and comparison. |
| This study | Next newly encountered skill recommendation | Binary learner--skill exposure graph with LightGCN | Prefix-only empirical difficulty, shrunk ability, and one-sided excess risk | Exact candidate-aware expected-risk regularization during training; matched score-level comparator | Evaluates the relevance--overchallenge trade-off for one specific task--objective--evaluation combination under temporal full ranking. |

## SLR reconciliation

- **FACT:** the author's SLR analyzes 84 studies from 2019--2025 and reports
  that 77.3% use post-hoc filtering or feature engineering (44.0% and 33.3%,
  respectively), while 2.4% use formal MCDM or constraint optimization.
- **INFERENCE:** those aggregate codes support the importance of comparing
  objective integration with post-hoc control, but they do not establish that
  the exact method in this study is absent. The SLR has a broader construct and
  different inclusion/coding goals from this task-specific novelty audit.
- **UPDATE:** the targeted search added works omitted from the earlier matrix,
  including multi-objective RL, cognitive-diagnosis selection, graph-plus-
  difficulty filtering, constrained exercise-group assembly, neural reranking,
  and a 2026 difficulty-adaptive DRL method.
- **OUT OF SCOPE AS DIRECT PRIOR ART:** LT-MKT (Zhang et al., CIKM 2026) models
  cognitive load and transfer for multi-domain knowledge tracing. It predicts
  future performance rather than recommending a ranked candidate list, so it
  informs terminology but is not a direct recommender comparator.

## Defensible positioning

- **REJECTED:** “the first difficulty-aware educational recommender.”
- **REJECTED:** “the first ability-aligned” or “the first objective-integrated
  pedagogical recommender.” Multi-objective RL and constrained optimization
  precede this study.
- **REJECTED:** “the first to contrast training integration with post-hoc
  control.” The search does not support a field-wide priority statement.
- **REJECTED:** equating Top-K recommendation with exercise-group assembly or
  sequential learning-path generation.
- **SUPPORTED, BOUNDED:** this study evaluates a training-integrated asymmetric
  expected-overchallenge term in a LightGCN Top-K learner--skill setting and
  compares it with a matched score-level comparator under one temporal,
  full-ranking relevance-and-risk protocol.
- **OPEN:** whether an earlier or newly released paper implements the exact
  combination. Refresh Scopus/Web of Science/ACM/IEEE searches immediately
  before submission; do not turn the bounded combination claim into a priority
  claim even if that refresh finds no match.

## Search record

- **Author SLR checked:** 84 included studies, publication window 2019--2025.
- **Targeted update date:** 2026-08-29.
- **Concept blocks:** educational/exercise/skill recommendation; difficulty,
  ability, challenge, cognitive diagnosis, pedagogical constraint; LightGCN,
  graph collaborative filtering, multi-objective, regularization, reranking.
- **Primary-source venues checked:** ACM Digital Library, IEEE Xplore, journal
  publisher pages, and author-hosted/arXiv full text for accepted ACM papers.
- **Limitation:** this update is a reproducible targeted comparison, not a new
  PRISMA review or a complete citation-index export.

## Verified primary sources

- Rendle et al., “BPR: Bayesian Personalized Ranking from Implicit Feedback,”
  UAI 2009: https://auai.org/uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf
- Huang et al., “Exploring Multi-Objective Exercise Recommendations in Online
  Education Systems,” CIKM 2019: https://doi.org/10.1145/3357384.3357995
- He et al., “LightGCN: Simplifying and Powering Graph Convolution Network for
  Recommendation,” SIGIR 2020: https://doi.org/10.1145/3397271.3401063
- Gong et al., “Attentional Graph Convolutional Networks for Knowledge Concept
  Recommendation in MOOCs in a Heterogeneous View,” SIGIR 2020:
  https://doi.org/10.1145/3397271.3401057
- Du et al., “Personalization Exercise Recommendation Based on Cognitive
  Diagnosis,” CSAE 2022: https://doi.org/10.1145/3565387.3565416
- Yan et al., “Personalization Exercise Recommendation Framework based on
  Knowledge Concept Graph,” ComSIS 2023:
  https://doi.org/10.2298/CSIS220706024Y
- Yang et al., “Cognitive Diagnosis-Based Personalized Exercise Group Assembly
  via a Multi-Objective Evolutionary Algorithm,” IEEE TETCI 2023:
  https://doi.org/10.1109/TETCI.2022.3220812
- Yu et al., “XSimGCL: Towards Extremely Simple Graph Contrastive Learning for
  Recommendation,” IEEE TKDE: https://doi.org/10.1109/TKDE.2023.3288135
- Zhang et al., “Item-Difficulty-Aware Learning Path Recommendation: From a
  Real Walking Perspective,” KDD 2024:
  https://doi.org/10.1145/3637528.3671947
- Cheng et al., “NR4DER: Neural Re-ranking for Diversified Exercise
  Recommendation,” SIGIR 2025: https://doi.org/10.1145/3726302.3730046
- Zhu et al., “A Sustainable Personalized Education Recommendation Method Based
  on Deep Reinforcement Learning,” IECA 2026:
  https://doi.org/10.1145/3802133.3802217

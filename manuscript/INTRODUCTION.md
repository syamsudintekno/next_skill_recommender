# Introduction

Educational recommenders can help learners navigate a large set of available
resources, but relevance and pedagogical suitability are not necessarily the
same objective. In an implicit-feedback setting, an observed interaction may
indicate that content was assigned or encountered rather than preferred, and an
unobserved interaction does not establish dislike or inability. A model trained
only to recover future interactions can therefore rank content that is
behaviorally relevant while remaining silent about whether its empirical
difficulty substantially exceeds a learner-specific reference point.

Graph collaborative filtering provides a natural representation for this
setting. LightGCN simplifies graph recommendation to normalized neighborhood
aggregation over a user–item interaction graph and is commonly optimized with
pairwise ranking loss
([He et al., 2020](https://doi.org/10.1145/3397271.3401063)). Educational graph
models have also incorporated richer relations for knowledge-concept
recommendation, as in ACKRec
([Gong et al., 2020](https://doi.org/10.1145/3397271.3401057)). These lines of
work establish graph-based relevance modeling, but their cited formulations do
not directly control learner-specific overchallenge in the exposure induced by
a Top-K ranking.

Difficulty-aware recommendation is not new. For example, Zhang et al. model
difficulty within hierarchical reinforcement learning for sequential
learning-path generation
([2024](https://doi.org/10.1145/3637528.3671947)). That setting differs from
ranking the complete unseen catalog of skills from a collaborative-exposure
graph: a path policy selects successive learning and practice actions and is
evaluated in a simulated environment, whereas a Top-K recommender produces a
single ranked set. The distinction motivates a narrower question: can
learner-specific difficulty risk be incorporated into graph collaborative
filtering itself, and how does doing so compare with modifying scores only after
training?

We study next-new-skill recommendation using a temporally split ASSISTments
interaction dataset. Exposure defines the learner–skill graph, while
correctness is used separately to construct behavioral proxies. Skill
difficulty is estimated from first-exposure success rates using empirical-Bayes
smoothing. Learner ability is summarized from successfully encountered skills
and shrunk toward the population center. Both quantities are calculated only
from the applicable training prefix. For learner \(u\) and skill \(i\), we
define a one-sided squared risk that is positive only when empirical difficulty
exceeds estimated ability plus tolerance. This quantity characterizes
overchallenge exposure; it is not an intrinsic difficulty label or a measure of
learning benefit.

A fixed sum of such risks would provide no gradient to a recommender because
the proxy values do not depend on model parameters. We instead take their
expectation under a temperature-controlled softmax over the model scores of all
unseen candidates and add that expectation to the LightGCN training objective.
This candidate-aware construction creates a gradient path from overchallenge
risk to the ranking scores. We compare the integrated formulation with standard
LightGCN, conventional and modern relevance baselines, and a matched post-hoc
method that subtracts the same asymmetric risk from trained LightGCN scores.

This study addresses the following research questions:

- **RQ1:** How does training-integrated asymmetric difficulty regularization
  affect Top-K ranking accuracy and learner-specific overchallenge exposure
  relative to relevance-only LightGCN?
- **RQ2:** How does integrated regularization compare with matched post-hoc
  reranking in the relevance–overchallenge trade-off?
- **RQ3:** How sensitive is that trade-off to the regularization weight
  \(\lambda_d\) and tolerance \(\tau\)?

The contributions are threefold:

1. We formulate a differentiable, candidate-aware asymmetric overchallenge
   objective for LightGCN. It penalizes only exposure above an estimated
   learner-ability-plus-tolerance boundary and retains the standard model
   exactly when \(\lambda_d=0\).
2. We provide a controlled comparison between training integration and
   score-level post-hoc reranking using the same empirical risk, backbone,
   temporal split, full candidate sets, evaluation metrics, tuning discipline,
   and matched seeds.
3. We characterize the relevance–risk frontier using ranking metrics together
   with violation rate, mean excess difficulty, and squared exposure risk, and
   examine coefficient, tolerance, and risk-form sensitivity under a
   leakage-controlled protocol.

The empirical results show a trade-off rather than universal dominance.
Integrated regularization produced a small reduction in overchallenge exposure
relative to LightGCN, accompanied by small ranking concessions, while post-hoc
reranking achieved a larger risk reduction at a larger Recall/NDCG cost than
the integrated model. These findings concern offline recommendation exposure
under cohort-dependent behavioral proxies. They do not establish improved
learning outcomes, a causal educational benefit, or learning-path quality.

# Related Work

## Collaborative filtering and graph recommendation

Bayesian Personalized Ranking (BPR) formalized personalized ranking from
implicit feedback as pairwise optimization over observed and unobserved items
([Rendle et al., 2009](https://auai.org/uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf)).
Its treatment of unobserved interactions as ranking samples rather than explicit
negative ratings is relevant to educational exposure logs, where absence of an
interaction does not establish dislike or inability. We use BPR both as the
matrix-factorization baseline and as the relevance component of the graph
models.

LightGCN removes feature transformations and nonlinear activations from graph
collaborative filtering, retaining normalized neighborhood aggregation and a
weighted combination of layer-wise embeddings
([He et al., 2020](https://doi.org/10.1145/3397271.3401063)). Its deliberately
simple propagation makes it a useful backbone for isolating an additional
educational objective. XSimGCL adds noise-based embedding augmentation and
cross-layer contrast to a LightGCN-style recommendation pipeline
([Yu et al., 2023](https://doi.org/10.1109/TKDE.2023.3288135)). We include it as
a modern graph baseline to distinguish difficulty-aware control from a stronger
relevance-only representation learner.

These methods optimize general implicit-feedback ranking. In their cited
formulations, they do not represent learner ability and empirical skill
difficulty as an asymmetric Top-K exposure cost. Our work retains their
collaborative-ranking foundation but adds a candidate-aware expected-risk term
to LightGCN training.

## Educational graphs and learner-state modeling

Educational recommendation has used graphs to represent relations beyond a
homogeneous learner--item matrix. ACKRec constructs a heterogeneous network of
learners, knowledge concepts, courses, videos, and teachers, then applies
meta-path-guided graph convolution and attention for MOOC knowledge-concept
recommendation ([Gong et al., 2020](https://doi.org/10.1145/3397271.3401057)).
Yan et al. combine deep knowledge tracing, a course knowledge graph, and a
learner knowledge-structure graph; candidate exercises are subsequently filtered
using difficulty, diversity, and novelty
([2023](https://doi.org/10.2298/CSIS220706024Y)). These studies establish that
graph structure and difficulty-aware selection can coexist in educational
recommendation. They do not, however, test the same graph-collaborative-filtering
objective or a matched integrated-versus-score-level risk intervention.

Learner-state models address a related but different problem. EKT combines
exercise content with recurrent representations of knowledge acquisition to
predict performance on future exercises
([Liu et al., 2021](https://doi.org/10.1109/TKDE.2019.2924374)). More recently,
LT-MKT has modeled cognitive load and knowledge transfer for multi-domain
knowledge tracing ([Zhang et al., 2026](https://doi.org/10.1145/3799682.3841120)).
Such models can provide richer learner-state estimates than our transparent,
prefix-only behavioral proxy. Predicting correctness for a supplied exercise,
however, is not equivalent to ranking the complete unseen catalog, and
predictive accuracy alone does not characterize the difficulty exposure of a
Top-K list.

## Difficulty-aware and multi-objective educational recommendation

Ability--difficulty matching and multi-objective control precede this study.
Huang et al. formulate adaptive exercise recommendation as deep reinforcement
learning with Review and Explore goals, difficulty smoothness, and engagement
([2019](https://doi.org/10.1145/3357384.3357995)). Du et al. use cognitive
diagnosis to incorporate student ability and exercise difficulty in exercise
selection ([2022](https://doi.org/10.1145/3565387.3565416)). Yang et al. cast
personalized exercise-group assembly as a constrained multi-objective
evolutionary problem informed by cognitive diagnosis
([2023](https://doi.org/10.1109/TETCI.2022.3220812)). These approaches establish
important precedents for adaptive, ability-aware, and explicitly multi-objective
recommendation, while differing from learned full-ranking graph scores and the
one-sided excess-risk expectation used here.

Difficulty has also been modeled in sequential learning-path recommendation.
The Difficulty-constrained Learning Path Recommendation framework separates
learning and practice items, constructs a hierarchical graph, and applies
difficulty-driven hierarchical reinforcement learning to generate paths
step-by-step ([Zhang et al., 2024](https://doi.org/10.1145/3637528.3671947)). A
2026 personalized exercise framework likewise combines deep knowledge tracing,
deep reinforcement learning, and a difficulty-adaptive constraint
([Zhu et al., 2026](https://doi.org/10.1145/3802133.3802217)). These sequential
policies differ from our static Top-10 next-new-skill ranking task in action
space, feedback assumptions, output, and evaluation.

Post-hoc control is an especially relevant comparator. NR4DER first predicts
mastery and filters exercises to an appropriate-difficulty candidate subset,
then applies neural reranking to balance relevance and learning-pattern
diversity ([Cheng et al., 2025](https://doi.org/10.1145/3726302.3730046)). This
shows that difficulty-aware filtering and educational reranking are already
established. Our question is consequently narrower: under one LightGCN backbone
and one asymmetric risk definition, does placing the signal in the training
objective yield a different relevance--risk operating point from applying the
matched risk only to final scores?

## Positioning of this study

Prior work therefore spans multi-objective reinforcement learning,
cognitive-diagnosis-based selection, graph-supported filtering, constrained
exercise-group assembly, sequential difficulty-aware paths, and neural
reranking. The contribution examined here is one specific combination: an
empirical, learner-specific, one-sided overchallenge risk is integrated into a
LightGCN objective through the model-induced distribution over all unseen
candidates, then compared with the matched score-level intervention under a
temporal full-ranking protocol that reports both relevance and risk exposure.

This is a bounded combination claim, not a priority claim. The targeted
primary-source comparison, reconciled with the author's 2019--2025 systematic
review and updated through August 2026, did not identify an identical
task--objective--evaluation combination. It cannot establish absence from the
literature, and the search must be refreshed before submission.

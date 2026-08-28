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
simple propagation makes it a useful backbone for isolating the effect of an
additional educational objective. More recent graph contrastive recommenders
seek stronger representations through self-supervision. XSimGCL uses
noise-based embedding augmentation and cross-layer contrast while sharing the
recommendation and contrastive propagation pipeline
([Yu et al., 2023](https://doi.org/10.1109/TKDE.2023.3288135)). We include it as
a modern graph baseline to distinguish gains from difficulty-aware control from
gains obtainable through a stronger relevance-only graph model.

These methods optimize general implicit-feedback ranking. They do not, in their
cited formulations, represent learner ability and empirical skill difficulty as
an asymmetric Top-K exposure cost. Our work retains their collaborative-ranking
foundation but adds a candidate-aware expected-risk term to the LightGCN
training objective.

## Graph and learner-state modeling in education

Educational recommendation research has used graphs to incorporate entities and
relations beyond a homogeneous learner–item interaction matrix. ACKRec constructs
a heterogeneous information network of learners, knowledge concepts, courses,
videos, and teachers, then uses meta-path-guided graph convolution and attention
for MOOC knowledge-concept recommendation
([Gong et al., 2020](https://doi.org/10.1145/3397271.3401057)). This demonstrates
that graph representation learning can support educational recommendation, but
its emphasis is interest/context propagation across heterogeneous relations.
The present study instead deliberately uses a minimal binary learner–skill graph
to isolate an overchallenge term in the ranking objective.

Learner-state models address a related but different problem. EKT combines
exercise content with recurrent representations of knowledge acquisition to
predict student performance on future exercises
([Liu et al., 2021](https://doi.org/10.1109/TKDE.2019.2924374)). Such prediction models
can provide rich estimates of learner state, whereas our ability variable is a
transparent, prefix-only behavioral proxy. However, predicting correctness for
a supplied exercise is not equivalent to ranking the complete unseen catalog,
and predictive accuracy alone does not characterize which difficulty levels a
Top-K recommender exposes.

## Difficulty-aware educational recommendation

Difficulty has been modeled explicitly in learning-path recommendation. The
closest verified anchor is the Difficulty-constrained Learning Path
Recommendation framework of Zhang et al.
([2024](https://doi.org/10.1145/3637528.3671947)). That work separates learning
and practice items, constructs a hierarchical graph, and applies
difficulty-driven hierarchical reinforcement learning to generate paths
step-by-step. Its experiments use simulators based on benchmark datasets and
evaluate path efficiency and smoothness.

Our setting differs along four dimensions. First, the output is a Top-10 list of
next-unseen skills rather than a sequential path containing learning and practice
actions. Second, the model is trained from a binary collaborative-exposure graph
rather than through a hierarchical reinforcement-learning environment. Third,
difficulty control is a soft asymmetric expected-exposure regularizer over the
full candidate catalog. Fourth, evaluation reports conventional ranking
relevance together with learner-specific overchallenge exposure and compares
training integration against score-level reranking. These distinctions prevent
claims about learning-path quality or simulated learning effectiveness from
being transferred to our offline Top-K task.

## Positioning of this study

Prior work establishes pairwise implicit-feedback ranking, simplified graph
collaborative filtering, contrastive graph representation learning, educational
knowledge-concept recommendation, learner-state prediction, and difficulty-aware
sequential path generation. The contribution examined here is narrower than
“difficulty-aware recommendation” in general: an empirical, learner-specific,
one-sided overchallenge risk is integrated into the LightGCN training objective
through the model-induced distribution over unseen candidates. A matched
post-hoc comparator then tests whether changing the training objective occupies
a different relevance–risk point from changing scores after training.

This positioning is a combination claim, not a priority claim. The targeted
primary-source comparison did not identify an identical task–objective–evaluation
combination, but it is not sufficient to establish that no such work exists. A
broader database search and reconciliation with the author's systematic review
remain necessary before submission.

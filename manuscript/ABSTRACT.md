# Abstract

Top-K educational recommenders optimized only for interaction relevance may
expose learners to skills whose empirical difficulty substantially exceeds a
learner-specific reference point. This study investigates training-integrated
asymmetric difficulty regularization for LightGCN and compares it with matched
post-hoc reranking. Skill difficulty was estimated from first-exposure
correctness with empirical-Bayes smoothing, while learner ability was derived
from successfully encountered skills with population shrinkage. Both proxies
were computed strictly from the applicable temporal training prefix. A
one-sided squared excess risk was integrated into training as its expectation
under a temperature-controlled softmax over every unseen candidate, thereby
preserving a gradient path to the ranking scores. Evaluation used full ranking
over each learner's unseen candidates on a temporally split ASSISTments dataset
with 22,241 test
learners, 264 prefix-visible skills, and five matched seeds. Relative to
LightGCN, integrated regularization reduced DVR@10 by 0.001103, MED@10 by
0.000214, and squared risk@10 by 0.000051, while NDCG@10 and MRR@10 decreased
by 0.000874 and 0.001192; the paired Recall@10 interval included zero. Post-hoc
reranking was more conservative than integrated regularization, reducing
DVR@10 by a further 0.005960, but it also reduced Recall@10 by 0.002392 and
NDCG@10 by 0.000679. Validation-only sensitivity analyses showed that stronger
difficulty-control coefficients reduced overchallenge at a relevance cost and
that larger tolerance increased both relevance and fixed-threshold risk. These
results demonstrate distinct operating points on a relevance–overchallenge
frontier rather than universal dominance by either control strategy. The
findings characterize offline recommendation exposure under cohort-dependent
behavioral proxies and do not establish causal learning benefits.

**Keywords:** educational recommendation; graph collaborative filtering;
LightGCN; empirical difficulty; overchallenge regularization

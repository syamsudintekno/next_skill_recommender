# Method and Experimental Setup

## Problem formulation

We formulate the task as Top-K recommendation of the next newly encountered
skill. Let \(\mathcal{U}\) denote learners and \(\mathcal{I}\) denote skills.
An edge \((u,i)\) indicates that learner \(u\) was exposed to skill \(i\) in the
training prefix. The edge is an exposure signal rather than a claim that the
learner preferred or mastered the skill. Correctness is retained separately for
constructing empirical difficulty and learner-ability proxies.

For learner \(u\), the candidate set is

\[
\mathcal{C}_u = \mathcal{I}_{\mathrm{prefix}} \setminus
\mathcal{I}^{\mathrm{seen}}_u,
\]

where \(\mathcal{I}_{\mathrm{prefix}}\) contains only skills visible in the
corresponding training prefix. Models rank every member of \(\mathcal{C}_u\);
no sampled negatives are used during evaluation. The target is the learner's
next first exposure to a skill.

## Temporal data construction

We used the ASSISTments 2012--2013 School Data with Affect, distributed by the
official ASSISTments data site and obtained through the Kaggle mirror record
*ASSISTments Data Set 2012-2013* (version 4; records reverified on August 28,
2026; ASSISTmentsData, n.d.; Wattiez, 2021). The mirror's sole version-4 file is
`2012-2013-data-with-predictions-4-final.csv`; the analyzed local copy was named
`dataset_skill.csv`. The public file metadata and local copy have the same exact
size of 3,009,494,391 bytes.
The local input is additionally fixed by SHA-256
`1d06aee9e649c5ba9db49052fe9a86e68d0c3ccb4709a7487289d7721f5464db`.
Because the public mirror does not expose a cryptographic checksum, the public
record-to-local-copy match is supported by the variant description, sole-file
metadata, exact byte size, compatible schema, and local row audit, whereas the SHA-256
identifies the exact bytes used in this study. We cite Feng et al. (2009), as
requested by the official data page for uses that do not analyze the affect
columns.

We retained events with valid learner, skill, event, and timestamp fields;
numeric correctness in \([0,1]\); `original = 1`; and a problem type other than
open response. A value of `correct = 1` was treated as success, while values
below 1 were treated as non-success. Learners required at least three distinct
skills.

Events were ordered within learner by timestamp and event identifier. We then
identified the first exposure to every learner–skill pair. The penultimate first
skill exposure was held out as the validation target, and the last first skill
exposure was held out as the test target. Event-level temporal cutoffs were
applied before constructing graphs, catalogs, popularity, difficulty, ability,
or any other aggregate. Consequently, the development prefix ended before the
validation target, whereas the final prefix included the development data and
validation-period information but ended before the test target.

The development artifacts contained 22,241 learners, 262 skills, 2,132,943
events, and 320,119 unique learner–skill edges. Validation contained 22,239
evaluable targets and two skills that were globally invisible in the development
prefix. The final-training artifacts contained 22,241 learners, 264 skills,
2,297,527 events, and 342,360 unique learner–skill edges. All 22,241 test targets
were evaluable. Table 1 summarizes the model-facing data.

**Table 1. Model-facing temporal-split statistics.**

| Statistic | Development/validation | Final training/test |
|---|---:|---:|
| Learners | 22,241 | 22,241 |
| Prefix-visible skills | 262 | 264 |
| Prefix events | 2,132,943 | 2,297,527 |
| Unique learner–skill edges | 320,119 | 342,360 |
| Evaluable targets | 22,239 | 22,241 |
| Globally prefix-invisible targets | 2 | 0 |
| Mean unseen candidates | 247.607 | 248.607 |

The original complete 12-file Stage 1 handoff manifest was unavailable in the
received archive. Before test access, we therefore froze and reverified the full
set of byte hashes recorded by the structural audit. This controls subsequent
artifact drift but cannot establish equality to the missing original handoff
manifest.

## Recommendation models

### Baselines

Popularity ranks unseen skills by learner support in the current training graph,
with lexical skill identifier as the deterministic tie-breaker. BPR-MF learns
learner and skill embeddings from observed learner–skill exposure pairs using a
pairwise ranking loss. An unobserved skill sampled for optimization is not
interpreted as an explicit negative preference.

LightGCN propagates learner and skill embeddings over the normalized bipartite
exposure graph without feature transformations or nonlinear activations. With
initial embedding matrix \(\mathbf{E}^{(0)}\), propagation is

\[
\mathbf{E}^{(\ell+1)}=\tilde{\mathbf{A}}\mathbf{E}^{(\ell)},
\]

and the final representation averages the layer-wise embeddings. The ranking
score is the inner product \(y_{ui}=\mathbf{e}_u^\top\mathbf{e}_i\). Training
uses Bayesian Personalized Ranking,

\[
\mathcal{L}_{\mathrm{BPR}}=-\mathbb{E}_{(u,i,j)}
\log \sigma(y_{ui}-y_{uj}),
\]

where \(i\) is an observed exposure and \(j\) is sampled from the learner's
unseen prefix catalog.

XSimGCL serves as the modern graph baseline. It retains the LightGCN graph
backbone and adds noise-based contrastive views and an InfoNCE term. This
baseline tests whether a stronger graph representation alone improves the
relevance–risk profile without an explicit difficulty objective.

### Empirical skill difficulty

Difficulty is estimated strictly from first learner–skill exposures in the
applicable training prefix. For skill \(i\), let \(n_i\) be the number of first
exposures and \(c_i\) the number that were successful. A Beta prior is fitted to
the collection of skill-level binomial observations by empirical Bayes. The
smoothed success probability and behavioral difficulty proxy are

\[
p_i=\frac{c_i+\alpha}{n_i+\alpha+\beta},\qquad d_i=1-p_i.
\]

Using first exposures aligns the proxy with recommending a newly encountered
skill and prevents repeated practice from dominating the estimate. The quantity
\(d_i\) is cohort-dependent empirical difficulty, not intrinsic or ground-truth
difficulty.

### Learner ability proxy

Let \(\mathcal{S}_u\) be the unique prefix skills on which learner \(u\) had at
least one successful event, and let \(m_u=|\mathcal{S}_u|\). Raw ability is the
equally weighted mean difficulty of these skills,

\[
\bar a_u=\frac{1}{m_u}\sum_{i\in\mathcal{S}_u}d_i.
\]

To stabilize short histories, we use method-of-moments shrinkage toward the
population mean \(\mu\):

\[
a_u=\frac{m_u\bar a_u+\kappa\mu}{m_u+\kappa}.
\]

Learners without a successful prefix skill receive the population center. The
development audit estimated \(\alpha=3.7440\), \(\beta=2.6928\),
\(\mu=0.3591\), and \(\kappa=15.4729\); 620 development learners used the
population fallback. For final training, the same estimation procedure—not the
development parameter values—was applied to the final prefix.

### Asymmetric overchallenge risk

For tolerance \(\tau\), the learner–skill risk is

\[
r_{ui}=\left[\max(0,d_i-a_u-\tau)\right]^2.
\]

The penalty is zero for a skill at or below the learner's estimated ability plus
tolerance. Easier skills are not penalized symmetrically because the construct
of interest is overchallenge rather than absolute difficulty mismatch.

### Training-integrated regularization

A direct sum of fixed \(r_{ui}\) values would have zero gradient with respect to
the recommender parameters. We instead define a temperature-controlled
distribution over every unseen candidate:

\[
q_\Theta(i\mid u,\mathcal{C}_u)=
\frac{\exp(y_{ui}/T)}{\sum_{j\in\mathcal{C}_u}\exp(y_{uj}/T)}.
\]

The expected overchallenge exposure is

\[
\mathcal{L}_{\mathrm{over}}=
\frac{1}{|\mathcal{U}|}\sum_{u\in\mathcal{U}}
\sum_{i\in\mathcal{C}_u}q_\Theta(i\mid u,\mathcal{C}_u)r_{ui},
\]

and the complete objective is

\[
\mathcal{L}=\mathcal{L}_{\mathrm{BPR}}+
\lambda_d\mathcal{L}_{\mathrm{over}}+
\lambda_2\|\Theta\|_2^2.
\]

The expectation is computed exactly over the full unseen prefix catalog and is
averaged uniformly over learners at every optimizer step. This prevents the
pedagogical term from inheriting the degree-biased learner distribution of BPR
triples. Automatic-differentiation and synthetic gradient checks verified a
non-zero gradient path through the candidate softmax. Setting \(\lambda_d=0\)
skips the risk computation and produced a checkpoint bit-identical to the
selected LightGCN implementation.

### Post-hoc comparator

The post-hoc method leaves the trained LightGCN embeddings unchanged and adjusts
only ranking scores:

\[
y'_{ui}=y_{ui}-\gamma r_{ui}.
\]

It therefore provides a matched comparison between difficulty control during
optimization and difficulty control applied after relevance-only training.

## Evaluation metrics

Ranking relevance is measured using Recall@10, NDCG@10, and MRR@10. Globally
prefix-invisible targets are counted and excluded from relevance denominators.
For the Top-10 list \(R_u\), the Difficulty Violation Rate is

\[
\mathrm{DVR@10}=\frac{1}{10|\mathcal{U}|}
\sum_u\sum_{i\in R_u}\mathbb{1}[d_i>a_u+\tau].
\]

Mean Excess Difficulty is the unconditional mean positive excess over all
recommendation exposures,

\[
\mathrm{MED@10}=\frac{1}{10|\mathcal{U}|}
\sum_u\sum_{i\in R_u}\max(0,d_i-a_u-\tau),
\]

and squared risk@10 replaces the final excess with its square. We additionally
retain the full vector of item exposure counts in lexical skill order. All
metrics use full ranking over the same unseen candidate sets.

## Experimental protocol

All selection was performed on validation data. Popularity was deterministic.
BPR-MF, LightGCN, and XSimGCL each received a bounded four-configuration tuning
budget. The selected BPR-MF used 64-dimensional embeddings, learning rate 0.02,
and L2 weight 0.001. The selected LightGCN used one propagation layer,
64-dimensional embeddings, learning rate 0.005, L2 weight 0.0001, batch size
65,536, and five optimizer steps per epoch. XSimGCL used two layers,
contrastive layer 1, learning rate 0.002, contrastive weight 0.2, perturbation
magnitude 0.2, temperature 0.2, and the same LightGCN batch and L2 settings.

Integrated and post-hoc coefficient selection used four predeclared settings per
family. Candidate points were first restricted to the NDCG–DVR Pareto set. The
primary rule then selected the minimum DVR among configurations whose relative
NDCG loss from relevance-only LightGCN did not exceed 1%. This selected
\(\lambda_d=0.03\) for integrated training and \(\gamma=5\) for post-hoc
reranking. Both used \(\tau=0.1\); integrated training used \(T=0.2\).

Development runs used validation NDCG@10 for checkpointing under a frozen
100-epoch cap. Selected configurations were evaluated with seeds 20260827–
20260831. Sensitivity analyses varied \(\tau\in\{0,0.05,0.1,0.2\}\); the
integrated coefficient over \(\{0.03,0.1,0.3,1.0\}\); and the post-hoc
coefficient over \(\{1,5,10,20\}\). Risk-form ablations compared
asymmetric-linear and symmetric-squared objectives with target-free mean-risk
scale matching. None of these analyses could revise the selected settings.

For the final experiment, every stochastic method was retrained on the final
prefix for exactly 100 epochs without validation evaluation, early stopping, or
checkpoint selection. The same five seeds were used. Post-hoc reranking reused
the same-seed final LightGCN checkpoint, and a duplicate \(\lambda_d=0\) final
run was omitted because equivalence had already been verified. All 20 training
checkpoints and hashes were completed before a single guarded batch evaluation
opened the test targets. Test results were locked against rerunning or
reselection.

We report mean and sample standard deviation across five seeds. Primary
comparisons use matched-seed mean differences and two-sided 95% paired-t
confidence intervals. The intervals are interpreted together with effect
magnitudes and trade-offs rather than as standalone evidence of practical
importance. All experiments ran deterministically on CPU with fixed software
versions recorded in each run artifact.

## Computational considerations

For this dataset, exact candidate-aware regularization was feasible because the
catalog contained only 262 development skills and 264 final skills. LightGCN
propagation scales with the graph edges and embedding dimension. The additional
integrated term materializes learner–candidate scores and risks, adding an
approximately \(O(|\mathcal{U}||\mathcal{I}|d)\) score computation per optimizer
step and \(O(|\mathcal{U}||\mathcal{I}|)\) risk/softmax storage. This exact
implementation avoids candidate-sampling bias, but its cost would require
approximation for substantially larger catalogs.

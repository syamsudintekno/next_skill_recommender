# Discussion and Limitations

## Interpreting the relevance–overchallenge trade-off

The central result is not that difficulty-aware control improves every metric.
Instead, both integrated regularization and post-hoc reranking moved the
recommendations toward lower overchallenge exposure while conceding some
ranking relevance. This pattern appeared in the final comparison, the
coefficient sweeps, the tolerance analysis, and the risk-form ablations. The
consistency across these analyses supports the interpretation that relevance
and learner-specific overchallenge are distinct objectives in this setting.

Integrated regularization produced a small but consistent reduction in DVR,
MED, and squared risk relative to relevance-only LightGCN. Its Recall difference
was not clearly separated from zero, whereas NDCG and MRR decreased. This is
compatible with the role of the proposed loss: it changes the score distribution
over all unseen candidates rather than imposing a hard Top-10 constraint. The
selected coefficient was also deliberately conservative because it was chosen
under a predeclared 1% validation NDCG-loss guardrail. These observations explain
the size and direction of the effect, but they do not establish that the
guardrail or coefficient is optimal for another educational context.

Post-hoc reranking reduced overchallenge more strongly than the integrated
method at the frozen operating points. Its intervention acts directly on the
scores used to form the final ranking, whereas the integrated objective shapes
a temperature-smoothed candidate distribution during optimization. The two
coefficients therefore operate on different numerical scales and should not be
compared by magnitude. The final evidence shows that post-hoc occupied the
lower-risk point and integrated regularization occupied the higher-Recall and
higher-NDCG point; it does not show universal superiority of either mechanism.

The distinction matters operationally. Post-hoc control can be adjusted without
retraining and may be preferable when exposure policy must be changed rapidly.
Integrated control is relevant when the scoring model itself should internalize
the risk signal or when downstream reranking is unavailable. In either case,
the operating point should be selected from an explicit relevance–risk frontier
rather than from relevance alone.

## Baseline behavior

BPR-MF achieved the highest mean Recall and NDCG, while XSimGCL achieved the
highest mean MRR. The graph models therefore did not uniformly outperform the
simpler factorization model. One plausible explanation is that the learner–skill
catalog is small and the binary exposure graph already provides sufficient
collaborative information for direct factorization. Another is that propagation
or contrastive learning may emphasize graph structure that is not aligned with
the next-new-skill target. These are hypotheses rather than tested mechanisms;
the current experiment cannot attribute BPR-MF's advantage to graph smoothing,
density, or any single architectural property.

XSimGCL's relevance profile also illustrates why a stronger relevance model is
not automatically a safer educational recommender. It produced the highest mean
MRR but also the highest DVR, MED, and squared risk among learned models. This
does not imply that contrastive learning intrinsically increases difficulty.
It shows only that the relevance-only XSimGCL configuration selected on
validation NDCG did not control the frozen overchallenge proxy.

Popularity produced the lowest risk values but substantially lower relevance.
Because it is non-personalized, this result should not be interpreted as a
desirable pedagogical solution. It is instead an extreme reference point showing
that reducing the reported risk metric without preserving learner-specific
ranking utility is insufficient.

## Validation-to-test behavior

The overall trade-off transferred from validation to test, but not every paired
relevance comparison remained identical. On validation, integrated and post-hoc
NDCG were not clearly separated; on the final test, integrated retained higher
NDCG. The final models were retrained on the longer final prefix, the catalog
expanded from 262 to 264 skills, and the unseen candidate distribution changed
slightly. These observed protocol differences may contribute to the change, but
the study was not designed to isolate which factor caused it. In accordance with
the frozen protocol, no setting was revised after observing test performance.

## Sensitivity and asymmetric design

The coefficient sweeps formed monotonic NDCG–DVR trade-offs: stronger integrated
or post-hoc penalties reduced violations while reducing NDCG. The tolerance
analysis revealed an important measurement issue. When each ranked list was
evaluated using its own tolerance, risk declined mechanically as tolerance
increased because the violation boundary was relaxed. Under the common
\(\tau_{eval}=0.1\) anchor, the direction reversed: larger training or reranking
tolerance improved relevance but increased exposure beyond the shared
overchallenge threshold. Reporting only native-tolerance risk would therefore
overstate the apparent safety improvement.

The risk-form ablations did not establish that asymmetric-squared risk dominates
all alternatives. The asymmetric-linear form was more conservative but lost
NDCG and MRR, whereas the symmetric-squared form slightly improved those
relevance metrics while increasing overchallenge. The selected asymmetric form
is justified primarily by construct alignment: the research question concerns
items above ability plus tolerance, not equal penalization of easier and harder
items. Squaring the excess concentrates the penalty on larger violations. The
empirical evidence should thus be presented as support for a chosen trade-off
and construct, not proof that this mathematical form is universally best.

## Threats to construct validity

Difficulty and ability are behavioral proxies. Skill difficulty is estimated
from first-exposure correctness within the observed cohort and may reflect prior
instruction, opportunity to learn, interface effects, guessing, or population
composition. It is neither an intrinsic property of a skill nor expert-validated
curriculum difficulty. Learner ability is derived from the difficulties of
skills with any successful prefix event and shrunk toward a population center.
It does not model knowledge state transitions, forgetting, prerequisite mastery,
or uncertainty at the individual prediction level.

The two proxies are related by construction because ability aggregates the same
cohort-based skill difficulties used in the risk function. Temporal isolation
prevents validation or test leakage, and shrinkage reduces instability, but
neither step removes this conceptual dependence. Future work should compare the
behavioral proxy with external assessments or independently estimated knowledge
states.

DVR measures whether an exposure crosses a threshold, MED measures the
unconditional positive excess, and squared risk emphasizes larger excesses.
These metrics characterize recommendation exposure; they do not measure whether
a learner attempted, completed, mastered, or benefited from a recommendation.
Accordingly, lower offline risk cannot be claimed to improve learning outcomes.

## Threats to internal validity

The split and all derived statistics were constructed temporally within the
training prefix. Incorrect responses were not converted into negative
preferences, and unobserved skills served only as optimization samples. These
choices reduce leakage and semantic mismatch, but binary exposure remains an
imperfect implicit-feedback signal: an observed skill may have been assigned by
the platform rather than chosen by the learner.

Hyperparameter budgets were bounded and frozen before observing the associated
validation results. Nevertheless, the search spaces are necessarily selective.
A larger or differently scaled search could yield other Pareto points. The
integrated and post-hoc methods received equal numbers of coefficient settings,
but exact equivalence of optimization opportunity is impossible because one
coefficient weights a training loss and the other modifies final scores.

All selected development configurations reached the frozen 100-epoch cap.
Extending the cap after observing this behavior was prohibited, and final models
were consequently trained for exactly 100 epochs without test-based checkpoint
selection. This protects the test protocol but leaves open whether a separately
predeclared larger training budget would change absolute performance.

## Threats to conclusion validity

Stochastic methods were evaluated with five matched seeds. We report sample
standard deviations and paired-t confidence intervals, emphasizing effect sizes
and directions. With only five pairs, interval estimates remain sensitive to
seed-level variation, especially for XSimGCL. Multiple metrics and comparisons
were examined without treating unadjusted intervals as a family-wise hypothesis
testing procedure. The intervals should therefore support estimation and
trade-off interpretation rather than binary declarations of significance.

Popularity was evaluated once because it is deterministic, so it has no
seed-based uncertainty estimate. In addition, model selection used NDCG and DVR;
the remaining metrics are complementary outcomes rather than independent
selection criteria.

## Threats to external validity

The study uses one ASSISTments dataset and a learner–skill recommendation unit.
The catalog is small enough to permit exact full ranking and an exact
learner-by-candidate expected-risk term. Results may differ for exercise-level
recommendation, larger catalogs, other educational platforms, different age
groups, or systems in which learners freely choose content. The method produces
a Top-K recommendation list, not a sequential learning path, prerequisite plan,
or curriculum policy.

The exact expected-risk implementation has learner-by-catalog
\(O(|\mathcal{U}||\mathcal{I}|)\) score storage and score-computation cost
proportional to learners, items, and embedding dimension. Candidate sampling or
other approximations would be needed at substantially larger scale, and those
approximations could change both the gradient and the resulting trade-off.

No subgroup claim is made for learner ability or interaction sparsity because a
subgroup protocol was not frozen before final test access. Similarly, this study
does not evaluate fairness across demographic groups, classroom deployment, or
causal effects on achievement.

## Reproducibility limitation

The current workspace contains verified raw-file identity information and a
complete pre-test snapshot of all 12 canonical artifact hashes. However, the
original complete Stage 1 handoff manifest and the official download URL/access
date were not present in the received archive. The recorded hashes prevent
subsequent artifact drift and all final files passed the pre-test gate, but this
does not reconstruct the missing provenance document. The official dataset
source, access date, and primary citation must be recovered before submission.

## Practical implication

Educational recommendation should expose the relevance–risk frontier rather
than collapse both objectives into a single unqualified “best model.” In this
experiment, integrated regularization offered a modest reduction in
overchallenge with limited ranking changes, whereas post-hoc reranking enabled a
larger risk reduction with a larger relevance concession. The appropriate point
depends on the cost assigned to overchallenge relative to missed relevant
skills. Determining that cost requires educational and stakeholder evidence
beyond the offline interaction data used here.

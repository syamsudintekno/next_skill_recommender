# Results

## Final test performance

Table 2 reports the locked test results. All stochastic methods were trained with
five matched random seeds and are reported as mean ± sample standard deviation.
Popularity is deterministic and was evaluated once. The final evaluation
included 22,241 learners, no globally prefix-invisible test targets, 264 candidate
skills, and an average of 248.61 unseen candidates per learner.

**Table 2. Final Top-10 relevance and overchallenge results.**

_Reproducibility source: `manuscript/generated/RESULTS_TABLES.md`, generated
directly from the locked final summary._

| Method | Recall@10 | NDCG@10 | MRR@10 | DVR@10 | MED@10 | Squared risk@10 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.287847 | 0.145124 | 0.101892 | 0.076485 | 0.002479 | 0.000228 |
| BPR-MF | **0.543186 ± 0.000981** | **0.321878 ± 0.000825** | 0.254217 ± 0.000920 | 0.180145 ± 0.001170 | 0.014737 ± 0.000048 | 0.002077 ± 0.000021 |
| LightGCN | 0.497792 ± 0.004085 | 0.293262 ± 0.001625 | 0.230797 ± 0.001494 | 0.187554 ± 0.001649 | 0.015498 ± 0.000114 | 0.002171 ± 0.000021 |
| XSimGCL | 0.509590 ± 0.006555 | 0.316049 ± 0.007898 | **0.256290 ± 0.008427** | 0.221725 ± 0.005924 | 0.021326 ± 0.000748 | 0.003899 ± 0.000149 |
| Integrated asymmetric-squared | 0.497972 ± 0.003583 | 0.292389 ± 0.001591 | 0.229605 ± 0.001465 | 0.186450 ± 0.001694 | 0.015284 ± 0.000120 | 0.002120 ± 0.000022 |
| Post-hoc asymmetric-squared | 0.495580 ± 0.004100 | 0.291709 ± 0.001557 | 0.229462 ± 0.001360 | 0.180490 ± 0.001547 | 0.014018 ± 0.000086 | 0.001816 ± 0.000026 |

BPR-MF obtained the highest mean Recall@10 and NDCG@10. XSimGCL obtained the
highest mean MRR@10, but it also produced the highest overchallenge exposure
among the learned models. Popularity produced the lowest risk values overall,
but its substantially lower relevance shows that low overchallenge alone does
not constitute a competitive personalized recommendation result.

## RQ1: Effect of integrated difficulty regularization

Compared with standard LightGCN, integrated asymmetric regularization reduced
DVR@10 by 0.001103 (95% CI [−0.001265, −0.000942]), MED@10 by 0.000214
([−0.000239, −0.000190]), and squared risk@10 by 0.0000510
([−0.0000591, −0.0000430]). This reduction was accompanied by decreases of
0.000874 in NDCG@10 ([−0.001289, −0.000459]) and 0.001192 in MRR@10
([−0.001781, −0.000604]). The Recall@10 difference was 0.000180
([−0.000530, 0.000889]), for which the paired interval included zero.

These results show that the training-integrated term affected the ranked
exposure distribution in the intended direction, but the effect was a modest
relevance–overchallenge trade-off rather than a simultaneous improvement in
both objectives.

## RQ2: Integrated regularization versus post-hoc reranking

Table 3 reports matched-seed differences for the two difficulty-control
strategies. Negative risk differences favor the post-hoc method, whereas
negative relevance differences indicate a loss relative to the integrated
model.

**Table 3. Paired post-hoc minus integrated differences on the final test set.**

_Reproducibility source: `manuscript/generated/RESULTS_TABLES.md`._

| Metric | Mean difference | 95% paired-t CI |
|---|---:|---:|
| Recall@10 | −0.002392 | [−0.003077, −0.001707] |
| NDCG@10 | −0.000679 | [−0.001085, −0.000274] |
| MRR@10 | −0.000143 | [−0.000719, 0.000433] |
| DVR@10 | −0.005960 | [−0.006391, −0.005529] |
| MED@10 | −0.001266 | [−0.001355, −0.001176] |
| Squared risk@10 | −0.000304 | [−0.000323, −0.000285] |

Post-hoc reranking was more conservative: it reduced each overchallenge metric
relative to the integrated method. The integrated method retained higher
Recall@10 and NDCG@10, while the paired MRR@10 interval included zero. Thus,
neither control strategy dominated the other. Post-hoc reranking occupied the
lower-risk point, whereas integrated regularization occupied the higher-relevance
point under the frozen coefficients.

## RQ3: Sensitivity to the regularization weight and tolerance

The coefficient and tolerance analyses used validation data only and did not
alter the final configuration. In the bounded single-seed coefficient analysis,
increasing the integrated weight from 0.03 to 1.0 decreased DVR@10 from
0.185306 to 0.170982, while NDCG@10 decreased from 0.284224 to 0.277656.
All four integrated settings were non-dominated on the NDCG–DVR plane. The
predeclared 1% relative-NDCG-loss guardrail selected λd=0.03. The analogous
post-hoc sweep showed the same monotonic pattern: increasing γ from 1 to 20
reduced DVR@10 from 0.185046 to 0.160370 and NDCG@10 from 0.285649 to
0.279204; γ=5 satisfied the frozen selection rule.

Tolerance sensitivity was evaluated across five seeds at τ ∈ {0, 0.05, 0.1,
0.2}. Under a common evaluation anchor of τeval=0.1, increasing the training
tolerance from 0 to 0.2 raised integrated NDCG@10 from 0.282073 to 0.285926
and DVR@10 from 0.180369 to 0.185206. For post-hoc reranking, NDCG@10 rose
from 0.281769 to 0.285678 and DVR@10 from 0.162182 to 0.183959. Therefore,
larger tolerance improved relevance while exposing learners to more items that
violated the common overchallenge threshold. Native-threshold risk decreased as
τ increased, but that trend is partly definitional because the violation
threshold itself becomes less strict.

## Risk-form ablations

Five-seed, scale-matched validation ablations further supported the trade-off
interpretation. Replacing the asymmetric-squared term with an asymmetric-linear
term reduced DVR@10 by 0.001168 and squared risk@10 by 0.0000183, but reduced
NDCG@10 by 0.000709 and MRR@10 by 0.000968. Conversely, the
symmetric-squared form increased NDCG@10 by 0.000349 and MRR@10 by 0.000438,
while increasing DVR@10 by 0.000221 and squared risk@10 by 0.00000489.
Recall@10 was not clearly separated in either comparison. These ablations do
not establish universal superiority of the asymmetric-squared form; rather,
they show that the selected one-sided formulation implements the intended
overchallenge construct at a distinct point on the relevance–risk trade-off.

## Summary of findings

Across the three research questions, difficulty-aware control consistently
changed exposure risk, but stronger risk reduction generally required a
relevance concession. Integrated regularization provided a small reduction in
overchallenge relative to LightGCN, while post-hoc reranking achieved a larger
risk reduction at a larger Recall/NDCG cost. The coefficient, tolerance, and
risk-form analyses all exhibited the same underlying trade-off. These offline
results characterize recommendation exposure under behavioral difficulty and
ability proxies; they do not demonstrate improved learning outcomes.

# Conclusion

This study examined whether learner-specific overchallenge risk can be
integrated into LightGCN training for next-new-skill recommendation. The method
combines the standard pairwise relevance objective with the expected value of a
one-sided squared difficulty risk under the model-induced distribution over all
unseen candidates. Difficulty and ability are prefix-only behavioral proxies,
and the formulation reduces exactly to the selected relevance-only LightGCN
implementation when \(\lambda_d=0\).

For RQ1, integrated asymmetric regularization changed the final ranked exposure
in the intended direction: it produced modest reductions in DVR@10, MED@10, and
squared risk@10 relative to LightGCN. These reductions were accompanied by
lower NDCG@10 and MRR@10, while the paired Recall@10 interval included zero.
The result is therefore a relevance–overchallenge trade-off rather than a
simultaneous improvement across objectives.

For RQ2, post-hoc reranking occupied a more conservative operating point than
integrated regularization under the frozen coefficients. It reduced all three
overchallenge metrics more strongly, whereas the integrated method retained
higher Recall@10 and NDCG@10; their MRR@10 difference was not clearly
separated. Neither strategy dominated the other. Integrated regularization is a
mechanism for incorporating the risk signal into model training, while post-hoc
reranking offers stronger and more easily adjustable score-level control in
this experiment.

For RQ3, the validation-only analyses showed that stronger difficulty-control
coefficients generally traded relevance for lower exposure risk. Tolerance also
changed the operating point: under a fixed evaluation threshold, larger
tolerance retained more relevance but allowed more overchallenge exposure.
This fixed-anchor result is important because risk measured at each setting's
own tolerance decreases partly by definition as the threshold is relaxed. The
risk-form ablations likewise identified different trade-off points rather than
a universally superior penalty form.

Overall, the findings support reporting a relevance–risk frontier instead of a
single unqualified best recommender. They also show that a stronger relevance
model is not necessarily less challenging: BPR-MF achieved the highest mean
Recall@10 and NDCG@10, whereas XSimGCL achieved the highest mean MRR@10 but the
highest overchallenge exposure among the learned models. Model choice and
difficulty-control strength should therefore reflect an explicitly justified
cost of overchallenge rather than ranking accuracy alone.

The conclusions are limited to offline Top-K exposure on one temporally split
ASSISTments dataset and to cohort-dependent behavioral proxies. The results do
not establish intrinsic skill difficulty, learner mastery, improved learning
outcomes, causal educational benefit, or learning-path quality. Future work
should validate the proxies against independent learner-state or assessment
signals, test additional datasets and recommendation units, study scalable
approximations for larger catalogs, and evaluate operating-point choices with
educational stakeholders before prospective deployment.

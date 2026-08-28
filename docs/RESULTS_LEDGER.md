# RESULTS_LEDGER

| Run | Split | Model | Recall@10 | NDCG@10 | MRR@10 | Evaluable | Cold | Mean candidates |
|---|---|---|---:|---:|---:|---:|---:|---:|
| POP_DEV_001 | validation | Global popularity | 0.283421 | 0.139201 | 0.095708 | 22,239 | 2 | 247.606807 |
| BPRMF_DEV_001 | validation | BPR-MF (provisional) | 0.461936 | 0.260206 | 0.199002 | 22,239 | 2 | 247.606807 |
| BPRMF_TUNE_001 | validation | BPR-MF, d=32/lr=.02/l2=.0001 | 0.506812 | 0.292339 | 0.226931 | 22,239 | 2 | 247.606807 |
| BPRMF_TUNE_002 | validation | BPR-MF, d=64/lr=.02/l2=.0001 | 0.520707 | 0.307544 | 0.242144 | 22,239 | 2 | 247.606807 |
| BPRMF_TUNE_003 | validation | BPR-MF, d=64/lr=.01/l2=.0001 | 0.502810 | 0.294770 | 0.231114 | 22,239 | 2 | 247.606807 |
| BPRMF_TUNE_004 | validation | BPR-MF, d=64/lr=.02/l2=.001 | **0.531409** | **0.313874** | **0.247370** | 22,239 | 2 | 247.606807 |
| LIGHTGCN_DEV_001 | validation | LightGCN 2-layer/full-batch (rejected protocol) | 0.291065 | 0.143859 | 0.099424 | 22,239 | 2 | 247.606807 |
| LIGHTGCN_TUNE_001 | validation | LightGCN 1-layer/full-batch (invalidated) | 0.295517 | 0.148680 | 0.104292 | 22,239 | 2 | 247.606807 |
| LIGHTGCN_DEV_002 | validation | LightGCN 2-layer/stochastic diagnostic | 0.294033 | 0.142917 | 0.097521 | 22,239 | 2 | 247.606807 |
| LIGHTGCN_DEV_003 | validation | LightGCN 1-layer/stochastic diagnostic | 0.434957 | 0.237573 | 0.177699 | 22,239 | 2 | 247.606807 |
| LIGHTGCN_BOUND_001 | validation | LightGCN d=32/lr=.002 | 0.407752 | 0.219776 | 0.162702 | 22,239 | 2 | 247.606807 |
| LIGHTGCN_BOUND_002 | validation | LightGCN d=32/lr=.005 | 0.457889 | 0.264235 | 0.205339 | 22,239 | 2 | 247.606807 |
| LIGHTGCN_BOUND_003 | validation | LightGCN d=64/lr=.002 | 0.437160 | 0.239406 | 0.179545 | 22,239 | 2 | 247.606807 |
| LIGHTGCN_BOUND_004 | validation | LightGCN d=64/lr=.005 | **0.485004** | **0.286078** | **0.225301** | 22,239 | 2 | 247.606807 |
| XSIMGCL_BOUND_004 | validation | XSimGCL selected | 0.480147 | **0.294898** | **0.237908** | 22,239 | 2 | 247.606807 |

Values are traced to their run JSON under `runs/stage2/`. No test result exists. `BPRMF_TUNE_004` is frozen for Stage 2 comparison under the bounded tuning budget; final multi-seed evaluation remains pending.

Rows marked rejected/invalidated are diagnostic evidence only and must not enter the manuscript comparison table.

## Selected-model validation summary (5 seeds)

| Model | Recall@10 mean±SD | NDCG@10 mean±SD | MRR@10 mean±SD |
|---|---:|---:|---:|
| BPR-MF | 0.531993±0.000719 | 0.311951±0.001684 | 0.244718±0.002124 |
| LightGCN | 0.484626±0.002695 | 0.286187±0.001738 | 0.225614±0.001822 |
| XSimGCL | 0.480579±0.003957 | 0.294421±0.001904 | 0.237233±0.001384 |

These are validation results, not final test results.

## Stage 3 status

The following are single-seed validation implementation probes, not selected or
final manuscript results.

| Run | Variant | Recall@10 | NDCG@10 | MRR@10 | DVR@10 | MED@10 | Mean squared risk@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| DRLGCN_LAMBDA0_DEV_20260827 | λd=0 equivalence | 0.485004 | 0.286078 | 0.225301 | 0.186826 | 0.016464 | 0.002324 |
| DRLGCN_INTEGRATED_DEV_20260827 | integrated, τ=.1/T=.2/λd=.1 | 0.487477 | 0.283190 | 0.220719 | 0.183629 | 0.015806 | 0.002193 |
| LIGHTGCN_POSTHOC_DEV_20260827 | post-hoc, τ=.1/γ=.1 | 0.485004 | 0.286016 | 0.225218 | 0.186678 | 0.016433 | 0.002317 |

The λd=0 checkpoint is bit-identical to Stage-2 `LIGHTGCN_BOUND_004`.
Hyperparameters and pedagogical metric definitions remain open; no test target
has been accessed. Audit trace:
`runs/stage3/audits/STAGE3_DEV_RUN_AUDIT.json`.

## Stage 3 bounded selection (seed 20260827)

| Selected variant | Coefficient | Recall@10 | NDCG@10 | MRR@10 | DVR@10 | MED@10 | Squared risk@10 | Relative NDCG loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Integrated | λd=0.03 | 0.485453 | 0.284224 | 0.222666 | 0.185306 | 0.016170 | 0.002265 | 0.6482% |
| Post-hoc | γ=5 | 0.482890 | 0.284385 | 0.223731 | 0.178836 | 0.014879 | 0.001989 | 0.5920% |

These rows record validation selection only. They are not multi-seed or final
results. All eight bounded points and the deterministic selection are traced to
`runs/stage3/STAGE3_BOUNDED_SELECTION.json` and
`runs/stage3/audits/STAGE3_BOUNDED_SELECTION_AUDIT.json`.

## Stage 3 selected variants — validation summary (5 seeds)

| Variant | Recall@10 mean±SD | NDCG@10 mean±SD | MRR@10 mean±SD | DVR@10 mean±SD | MED@10 mean±SD | Squared risk@10 mean±SD |
|---|---:|---:|---:|---:|---:|---:|
| Integrated λd=0.03 | 0.484860±0.002974 | 0.284372±0.001572 | 0.223112±0.001432 | 0.183588±0.001670 | 0.016091±0.000193 | 0.002257±0.000029 |
| Post-hoc γ=5 | 0.482765±0.003199 | 0.284507±0.001675 | 0.223997±0.001598 | 0.177589±0.001401 | 0.014851±0.000182 | 0.001985±0.000030 |

Paired differences use post-hoc minus integrated. The 95% CI includes zero for
NDCG only; it excludes zero for Recall, MRR, DVR, MED, and squared risk. These
are validation-only results, traced to `STAGE3_MULTI_SEED_VALIDATION.json` and
`runs/stage3/audits/STAGE3_MULTI_SEED_AUDIT.json`.

## Stage 3 τ sensitivity — fixed-anchor τ_eval=0.1 (5-seed means)

| Training/reranking τ | Variant | Recall@10 | NDCG@10 | MRR@10 | Anchor DVR@10 | Anchor MED@10 | Anchor squared risk@10 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.00 | Integrated | 0.485651 | 0.282073 | 0.219823 | 0.180369 | 0.015733 | 0.002202 |
| 0.00 | Post-hoc | 0.479329 | 0.281769 | 0.221477 | 0.162182 | 0.012917 | 0.001674 |
| 0.05 | Integrated | 0.484932 | 0.283253 | 0.221599 | 0.182178 | 0.015909 | 0.002228 |
| 0.05 | Post-hoc | 0.481209 | 0.283346 | 0.222953 | 0.170876 | 0.013953 | 0.001837 |
| 0.10 | Integrated | 0.484860 | 0.284372 | 0.223112 | 0.183588 | 0.016091 | 0.002257 |
| 0.10 | Post-hoc | 0.482765 | 0.284507 | 0.223997 | 0.177589 | 0.014851 | 0.001985 |
| 0.20 | Integrated | 0.484743 | 0.285926 | 0.225227 | 0.185206 | 0.016337 | 0.002298 |
| 0.20 | Post-hoc | 0.484113 | 0.285678 | 0.225104 | 0.183959 | 0.015982 | 0.002196 |

Native-threshold risk metrics are retained in `STAGE3_TAU_SENSITIVITY.json`,
but must be interpreted together with this fixed-anchor table because increasing
τ mechanically relaxes the violation definition. Sensitivity is validation-only
and does not revise the frozen τ=0.1 setting.

## Stage 3 risk-form ablations — validation summary (5 seeds)

| Integrated risk form | Recall@10 mean±SD | NDCG@10 mean±SD | MRR@10 mean±SD | DVR@10 mean±SD | MED@10 mean±SD | Squared risk@10 mean±SD |
|---|---:|---:|---:|---:|---:|---:|
| Asymmetric squared (frozen reference) | 0.484860±0.002974 | 0.284372±0.001572 | 0.223112±0.001432 | 0.183588±0.001670 | 0.016091±0.000193 | 0.002257±0.000029 |
| Asymmetric linear | 0.484932±0.003166 | 0.283663±0.001457 | 0.222143±0.001275 | 0.182420±0.001688 | 0.015956±0.000202 | 0.002238±0.000031 |
| Symmetric squared | 0.484932±0.002951 | 0.284721±0.001546 | 0.223550±0.001466 | 0.183809±0.001645 | 0.016121±0.000195 | 0.002261±0.000030 |

These scale-matched OFAT comparisons are validation-only. Linear reduces risk
with lower NDCG/MRR; symmetric slightly improves NDCG/MRR while increasing
overchallenge risk. They do not reopen the frozen objective or coefficients.

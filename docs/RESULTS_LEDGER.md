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

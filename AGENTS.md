# ChatGPT project context

This directory is a local mirror of the ChatGPT project “S3-Paper Q3-JOIN-difficulty-regularized”.

- Treat every file under `sources/` as read-only reference material.
- Do not edit, rename, move, or delete synced project files.
- These files may be replaced the next time a task is created from this ChatGPT project.


## Project instructions

Project Instructions — Paper JOIN

Project ini hanya untuk paper empiris berbahasa Inggris yang ditargetkan ke Jurnal Online Informatika (JOIN), Vol. 12 No. 1, Juni 2027. Penulis telah berkomunikasi dengan EiC dan mendapat kesediaan slot jika paper digarap serius, tetapi ini bukan acceptance. Target submit Januari 2027. Jangan alihkan project ke revisi proposal, SLR, atau keseluruhan disertasi.

Judul kerja: “Balancing Relevance and Overchallenge in Graph-Based Educational Recommendation: An Asymmetric Difficulty-Regularized LightGCN.”

Tujuan: menguji training-integrated asymmetric difficulty regularization pada LightGCN untuk menyeimbangkan Top-K relevance dan learner-specific overchallenge risk. Kontribusi: (1) difficulty control masuk objective training, bukan sekadar filter/reranking; (2) penalti asimetris hanya untuk item di atas ability plus tolerance; (3) evaluasi accuracy–pedagogy trade-off versus post-hoc reranking.

RQ wajib: (1) dampak pada ranking accuracy dan overchallenge risk; (2) integrated regularization vs post-hoc reranking; (3) sensitivitas τ dan λd. RQ subgroup opsional.

Dataset: ASSISTments 2012–2013. Exact raw variant, unit learner–exercise vs learner–skill, graph edge, positive interaction, split, dan candidate set masih OPEN. Audit raw data dahulu. Jangan sebut Skill Builder kecuali raw variant/filter mendukung. Keluarkan atau tangani khusus open-response. Exposure, correctness, dan preference tidak sama; incorrect response bukan otomatis negative preference.

Pipeline anti-leakage: clean event → sort temporal per learner → tentukan validation/test → freeze training prefix → hitung graph, difficulty, ability, popularitas, subgroup, dan statistik lain hanya dari training → agregasi dalam split → tune pada validation → test sekali setelah config freeze. Tolak agregasi seluruh histori sebelum split.

Working formulation:

empirical difficulty: p_i=(c_i+a0)/(n_i+a0+b0) dan d_i=1-p_i;

ability: agregasi difficulty item training yang berhasil, dengan fallback/shrinkage untuk cold start;

risk: r_ui=[max(0,d_i-a_u-τ)]²;

qΘ(i|u,C)=softmax(y_ui/T);

L_over=(1/|U|)Σ_uΣ_i qΘ(i|u,C)r_ui;

L_total=L_BPR+λd L_over+λ2||Θ||².

Tolak L_BPR+λΣr_ui jika risk hanya statistik tetap karena gradiennya nol. Audit candidates, sampling approximation, temperature, τ, λd, dan computational cost.

Baseline: BPR-MF, LightGCN, satu modern graph recommender, LightGCN + post-hoc reranking, dan proposed model. Ablation: λd=0, integrated vs post-hoc, asymmetric vs symmetric, linear vs squared risk, sensitivity τ/λd. Samakan preprocessing, split, candidates, evaluation, dan tuning budget.

Metrik: Recall@10, NDCG@10, MRR@10; DVR@10, MED@10, exposure distribution; Pareto accuracy–risk. Gunakan temporal validation/test, full ranking jika feasible, ≥5 seeds, mean±SD, CI/paired test, early stopping hanya pada validation, dan jangan memilih seed/hyperparameter dari test.

Prior art: LightGCN (He et al., 2020); KDD 2024 “Item-Difficulty-Aware Learning Path Recommendation: From a Real Walking Perspective,” DOI 10.1145/3637528.3671947; SLR dan proposal penulis. Bedakan sequential/RL path setting dari Top-K graph-CF exposure regularization melalui novelty matrix.

Larangan klaim: jangan mengklaim “first difficulty-aware”, learning outcomes, causal benefit, learning path jika output hanya Top-K, hard constraint, intrinsic/ground-truth difficulty, atau acceptance JOIN tanpa bukti resmi. Gunakan “observed/empirical difficulty” atau “behavioral proxy”.

Scope exclusions: prerequisite/DAG, expert annotation, AHP/CA-AHP, Bloom taxonomy, curriculum alignment, Moodle deployment, field experiment, dan causal learning claims.

Bertindak sebagai critical research collaborator. Bedakan FACT, INFERENCE, PROPOSAL, DECISION; tandai LOCKED, OPEN, REVISED, atau REJECTED. Kritik leakage, circularity, weak baseline, unfair tuning, overclaim, dan istilah inkonsisten. Jangan mengarang sitasi, DOI, data, hasil, atau isi sumber. Gunakan sumber primer/situs resmi dan verifikasi informasi jurnal yang berubah. Nyatakan bukti yang masih diperlukan.

Jaga DECISIONS.md, DATA_AUDIT.md, DESIGN_FREEZE.md, EXPERIMENT_LOG.md, RESULTS_LEDGER.md, dan MANUSCRIPT_CHANGELOG.md. Setiap angka harus terlacak ke run/config/seed. Keputusan terbaru mengungguli konteks lama dan dicatat sebagai revisi.

Urutan: design freeze → baselines → proposed method → final experiments → Results → Method → Discussion/Limitations → Related Work → Introduction → Conclusion → Abstract. Skeleton/Method boleh disiapkan awal; hasil numerik hanya dari eksperimen final.

Mulai project pada Stage 1 — Design Freeze: audit exact raw dataset, pilih unit rekomendasi berbasis statistik, freeze edge/positive/split/candidates, lakukan objective gradient sanity-check, dan bangun novelty matrix. Jangan mulai dari Introduction.

# Konteks Induk Project Paper JOIN

## 1. Identitas project

- **Nama project yang disarankan:** Paper JOIN — Difficulty-Regularized LightGCN
- **Jenis keluaran:** artikel empiris berbahasa Inggris
- **Jurnal target:** Jurnal Online Informatika (JOIN)
- **Target edisi:** Volume 12, Nomor 1, Juni 2027
- **Target submit internal:** Januari 2027
- **Status komunikasi:** penulis telah berkomunikasi langsung dengan Editor-in-Chief (EiC) JOIN dan memperoleh kesediaan slot apabila manuskrip digarap secara serius dan memenuhi proses editorial. Ini adalah peluang penerbitan, **bukan bukti accepted** dan bukan alasan untuk menurunkan standar ilmiah.
- **Fungsi dalam disertasi:** paper empiris utama yang menguji metode rekomendasi pendidikan berbasis graf dengan regularisasi kesulitan asimetris.
- **Bahasa kerja:** diskusi dapat memakai Bahasa Indonesia; naskah akhir wajib menggunakan academic English.

Project ini hanya digunakan untuk merancang, mengeksekusi, menulis, mengaudit, dan merevisi paper JOIN. Revisi proposal disertasi, penulisan SLR, dan pekerjaan disertasi lain dikerjakan di project terpisah agar ruang lingkup tidak melebar.

## 2. Tujuan akhir dan definisi selesai

Tujuan project adalah menghasilkan manuskrip empiris yang:

1. memiliki kontribusi teknis yang jelas dan dapat diuji;
2. membedakan relevansi rekomendasi dari risiko materi yang terlalu sulit;
3. menggunakan protokol eksperimen bebas data leakage;
4. membandingkan metode yang diusulkan dengan baseline dan post-hoc reranking;
5. melaporkan trade-off akurasi–pedagogi secara transparan;
6. dapat direproduksi dari data mentah hingga seluruh tabel dan gambar;
7. sesuai scope, template, batas halaman, dan kebijakan JOIN yang berlaku saat submit.

Project dinyatakan selesai hanya jika kode, konfigurasi, seed, log eksperimen, tabel, gambar, naskah, dan jawaban reviewer saling konsisten dan seluruh angka di manuskrip dapat ditelusuri ke keluaran eksperimen.

## 3. Judul kerja

**Balancing Relevance and Overchallenge in Graph-Based Educational Recommendation: An Asymmetric Difficulty-Regularized LightGCN**

Judul ini berstatus **locked sementara**. Judul hanya boleh diubah jika unit rekomendasi, formulasi metode, atau bukti eksperimen berubah secara material.

## 4. Masalah penelitian

Sistem rekomendasi pendidikan berbasis collaborative filtering umumnya mengoptimalkan relevansi berdasarkan pola interaksi. Pada konteks pembelajaran, item yang relevan belum tentu berada pada tingkat kesulitan yang sesuai bagi seorang learner. Rekomendasi yang terlalu sulit dapat meningkatkan risiko overchallenge, sedangkan pembatasan kesulitan yang terlalu agresif dapat menurunkan relevansi dan mengekspos learner hanya pada materi mudah.

Paper ini meneliti apakah LightGCN dapat diberi sinyal pedagogis selama training melalui regularisasi kesulitan yang **asimetris**: metode menghukum rekomendasi yang melampaui estimasi kemampuan learner dan margin toleransi, tetapi tidak menghukum item yang lebih mudah dengan cara yang sama.

Masalah ilmiah utamanya bukan “membuat rekomendasi mudah”, melainkan menemukan dan menjelaskan **trade-off Pareto antara ranking relevance dan overchallenge risk**.

## 5. Klaim kebaruan yang dituju

### Klaim inti

Training-integrated asymmetric difficulty regularization untuk graph collaborative filtering yang menyeimbangkan relevansi Top-K dan risiko overchallenge berbasis learner.

### Kontribusi yang akan diuji

1. **Integrasi saat training.** Difficulty control memengaruhi pembelajaran skor model, bukan hanya metadata, hard filtering, atau reranking setelah model selesai dilatih.
2. **Penalti asimetris.** Hanya rekomendasi yang berada di atas kemampuan learner plus tolerance margin yang dikenai risiko.
3. **Evaluasi dua tujuan.** Kualitas ranking dan exposure pedagogis dilaporkan bersama, termasuk kurva trade-off dan perbandingan dengan post-hoc reranking.

### Batas klaim

Jangan mengklaim:

- “metode pertama yang mempertimbangkan difficulty” tanpa bukti systematic prior-art search;
- peningkatan learning outcome karena dataset observasional ini tidak menguji dampak kausal terhadap pembelajaran;
- “learning path recommendation” jika keluaran model hanya daftar Top-K;
- personalized curriculum, prerequisite satisfaction, atau pedagogical sequencing tanpa pemodelan eksplisit;
- difficulty sebagai ground truth intrinsik; gunakan istilah **observed/empirical difficulty** atau **behavioral proxy**;
- constraint keras jika formulasi sebenarnya berupa soft regularization;
- acceptance JOIN sebelum keputusan editorial resmi.

## 6. Research questions

- **RQ1:** How does asymmetric difficulty-regularized LightGCN affect ranking accuracy and learner-specific overchallenge risk compared with standard recommendation baselines?
- **RQ2:** Does integrating difficulty control during training offer a better accuracy–risk trade-off than post-hoc reranking?
- **RQ3:** How sensitive is the method to the difficulty-regularization weight and tolerance margin?
- **RQ4 (opsional):** Does the effect differ across learner-ability or interaction-sparsity groups?

RQ1–RQ3 wajib. RQ4 hanya dipertahankan jika halaman dan kekuatan statistik mencukupi.

## 7. Dataset dan audit wajib

Dataset utama yang direncanakan adalah **ASSISTments 2012–2013**. Sebelum modeling, identitas raw file/variant harus dicatat secara presisi: nama unduhan, URL sumber resmi, tanggal akses, checksum, jumlah baris, kolom, dan tipe problem.

Hal yang wajib diaudit:

1. Apakah raw data mencampur beberapa problem type. Istilah “Skill Builder” hanya boleh digunakan jika filtering resmi dan datanya mendukung.
2. Open-response dapat memiliki label correct yang tidak sebanding dengan respons biasa; tipe ini harus dikeluarkan atau dianalisis khusus, disertai alasan.
3. Definisi `correct` harus mengikuti dokumentasi raw data. Jika dokumentasi menyatakan `1` berarti first-attempt correct dan nilai `<1` berarti tidak benar pada percobaan pertama, aturan itu harus dibekukan dan dicatat.
4. Audit duplikasi, missing ID, timestamp tidak valid, repeated attempts, hint usage, problem type, skill mapping, dan learner/item dengan interaksi sangat sedikit.
5. Jangan menentukan unit rekomendasi hanya dari kebiasaan literatur. Putuskan berdasarkan statistik data.

### Keputusan unit rekomendasi — masih terbuka

- **Kandidat utama:** learner–exercise.
- **Fallback:** learner–skill jika exercise-level terlalu sparse atau candidate set tidak stabil.

Keputusan harus didasarkan pada support per learner/item, repetisi, ukuran kandidat, kemampuan membentuk next-unseen target, dan risiko exposure bias. Jika unit berubah, seluruh terminologi, graph, difficulty, ability, candidate set, dan evaluasi harus diperbarui konsisten.

## 8. Semantik interaksi dan pencegahan leakage

Exposure, correctness, dan preference tidak boleh disamakan tanpa argumen.

- Edge graph dapat merepresentasikan exposure/attempt yang memenuhi kriteria.
- Correctness digunakan untuk menghitung empirical difficulty dan learner ability sesuai definisi yang dibekukan.
- Incorrect interaction **bukan otomatis negative preference**.
- Positif untuk BPR harus didefinisikan eksplisit dan diuji melalui sensitivity analysis jika ada beberapa pilihan masuk akal.

Urutan pipeline wajib:

1. bersihkan raw event;
2. urutkan event secara temporal per learner;
3. tentukan validation dan test event/target;
4. bekukan training prefix;
5. bangun graph, difficulty, ability, popularitas, dan seluruh statistik hanya dari training prefix;
6. lakukan agregasi yang diperlukan **di dalam split**, bukan pada seluruh histori sebelum split;
7. tuning hanya pada validation;
8. buka test untuk evaluasi final.

Tidak boleh ada statistik global dari validation/test yang masuk ke training, negative sampling, candidate filtering, difficulty, ability, subgroup thresholds, atau early stopping.

## 9. Formulasi awal metode

### 9.1 Empirical item difficulty

Untuk item (i):

\[
p_i = \frac{c_i + a_0}{n_i + a_0 + b_0},
\qquad
d_i = 1-p_i
\]

dengan (c_i) jumlah successful/correct training interactions, (n_i) jumlah valid training attempts, dan (a_0,b_0) prior smoothing. Nilai (d_i\in[0,1]) adalah proxy behavioral berbasis cohort, bukan sifat intrinsik item.

### 9.2 Learner ability proxy

Estimasi awal kemampuan learner (u): rata-rata difficulty dari item training yang berhasil dikuasai/dijawab benar oleh learner tersebut. Definisi “mastered/successful” wajib mengikuti unit rekomendasi dan raw data yang dipilih. Untuk cold-start atau histori tidak cukup, gunakan median difficulty training atau shrinkage menuju statistik populasi.

### 9.3 Asymmetric overchallenge risk

\[
r_{ui}=\left[\max\left(0,d_i-a_u-\tau\right)\right]^2
\]

dengan (a_u) ability proxy dan τ tolerance margin. Risiko nol untuk item yang tidak melampaui ability plus margin.

### 9.4 Objective yang valid

Jangan gunakan:

\[
\mathcal{L}_{BPR}+\lambda\sum r_{ui}
\]

jika (r_{ui}) hanya dihitung dari statistik tetap, karena suku tersebut tidak bergantung pada parameter model dan gradiennya nol.

Gunakan candidate-aware expected exposure risk:

\[
q_{\Theta}(i\mid u,\mathcal{C}_u)
=
\frac{\exp(y_{ui}/T)}{\sum_{j\in\mathcal{C}_u}\exp(y_{uj}/T)}
\]

\[
\mathcal{L}_{over}
=
\frac{1}{|\mathcal{U}|}
\sum_u\sum_{i\in\mathcal{C}_u}
q_{\Theta}(i\mid u,\mathcal{C}_u)r_{ui}
\]

\[
\mathcal{L}_{total}
=
\mathcal{L}_{BPR}
+\lambda_d\mathcal{L}_{over}
+\lambda_2\lVert\Theta\rVert_2^2
\]

Sanity check gradien untuk skor kandidat (k):

\[
\frac{\partial \mathcal{L}_{over}}{\partial y_{uk}}
=
\frac{q_{uk}}{T}
\left(r_{uk}-\sum_i q_{ui}r_{ui}\right)
\]

Candidate set, temperature, sampling approximation, dan computational cost masih harus diaudit. Formulasi ini adalah working specification, bukan final theorem.

## 10. Baseline dan ablation

### Baseline minimum

1. BPR-MF;
2. LightGCN;
3. satu modern graph recommender yang layak dan dapat direproduksi;
4. LightGCN + post-hoc difficulty reranking;
5. proposed asymmetric difficulty-regularized LightGCN.

Baseline difficulty-aware non-graph boleh ditambahkan jika implementasinya kredibel dan tidak mengorbankan eksperimen inti.

### Ablation minimum

- `lambda_d = 0`;
- training-integrated vs post-hoc reranking;
- asymmetric vs symmetric penalty;
- linear vs squared hinge risk;
- sensitivity terhadap τ dan `lambda_d`;
- smoothing/prior sensitivity jika stabilitas difficulty menjadi isu.

Semua baseline harus memakai split, candidate universe, preprocessing, negative sampling, dan evaluation protocol yang setara. Tuning budget harus sebanding.

## 11. Evaluasi

### Ranking utility

- Recall@10
- NDCG@10
- MRR@10

### Pedagogical exposure/risk

- Difficulty Violation Rate at 10 (DVR@10)
- Mean Excess Difficulty at 10 (MED@10)
- distribusi difficulty yang terekspos
- kurva atau frontier accuracy–risk ketika λd dan τ berubah

Definisi metrik harus ditulis matematis sebelum eksperimen final. Laporkan juga statistik dataset dan coverage rekomendasi.

### Protokol statistik

- temporal train/validation/test split;
- full ranking jika feasible; jika sampling dipakai, nyatakan prosedur dan uji robustness;
- minimal lima random seeds;
- mean dan standard deviation;
- confidence interval dan/atau paired significance test yang sesuai;
- early stopping dan hyperparameter selection hanya dari validation;
- test dievaluasi setelah konfigurasi dibekukan;
- dilarang memilih seed, checkpoint, atau hyperparameter berdasarkan test.

Hasil praktis harus dibahas bersama effect size dan trade-off, bukan hanya p-value.

## 12. Prior art dan posisi paper

Sumber jangkar yang sudah tersedia:

- He et al. (2020), **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**.
- Paper KDD 2024, **Item-Difficulty-Aware Learning Path Recommendation: From a Real Walking Perspective**, DOI: `10.1145/3637528.3671947`.
- SLR penulis tentang pedagogically aware educational recommender systems.
- Proposal disertasi Bab 1–3.

Perbedaan awal dengan paper KDD 2024: paper tersebut berfokus pada sequential/reinforcement-learning learning-path setting, sedangkan studi ini menargetkan Top-K graph collaborative filtering dengan training-integrated exposure regularization. Perbedaan ini harus dibuktikan melalui novelty matrix, bukan hanya dinyatakan.

SLR dipakai sebagai peta gap dan sumber literatur, tetapi teksnya tidak boleh didaur ulang secara berlebihan. Setiap klaim novelty harus diperbarui melalui pencarian prior art sebelum submission.

## 13. Scope yang sengaja dikeluarkan

Paper JOIN ini tidak mencakup:

- expert annotation;
- prerequisite graph atau curriculum DAG;
- AHP/CA-AHP;
- Bloom taxonomy modeling;
- curriculum alignment;
- Moodle plugin atau deployment production;
- eksperimen kelas/field experiment;
- causal claim tentang peningkatan hasil belajar;
- keseluruhan arsitektur disertasi.

Komponen tersebut hanya boleh masuk jika penulis secara eksplisit membuka scope dan tersedia bukti serta halaman yang memadai.

## 14. Struktur manuskrip dan anggaran halaman

Targetkan 9–10 halaman sesuai template/kebijakan JOIN yang berlaku. Periksa ulang author guideline resmi saat manuscript freeze karena kebijakan jurnal dapat berubah.

Alokasi awal:

1. Abstract dan keywords: ringkas, berbasis hasil aktual.
2. Introduction: masalah, gap, kontribusi, RQ.
3. Related Work: graph recommendation, educational recommendation, difficulty-aware recommendation.
4. Method: task definition, LightGCN, proxies, risk, objective, complexity.
5. Experimental Setup: dataset, preprocessing, split, baselines, tuning, metrics.
6. Results and Discussion: RQ1–RQ3; RQ4 jika layak.
7. Threats/Limitations: proxy validity, observational data, generalizability, candidate approximation.
8. Conclusion: temuan aktual dan batas klaim.

Visual utama: satu diagram metode yang kompak, satu tabel statistik dataset, satu tabel hasil utama, satu tabel/plot ablation-sensitivity, dan satu kurva trade-off jika ruang memungkinkan.

## 15. Urutan kerja penulisan

Gunakan pendekatan results-first:

1. bekukan desain dan protokol;
2. siapkan skeleton Experimental Setup;
3. jalankan baseline dan proposed model;
4. finalkan tabel/plot;
5. tulis Results;
6. tulis Method;
7. tulis Discussion dan Limitations;
8. tulis Related Work;
9. tulis Introduction dan Contributions;
10. tulis Conclusion;
11. tulis Abstract paling akhir.

Skeleton dan Method boleh disiapkan sebelum eksperimen selesai, tetapi jangan menulis hasil, besar peningkatan, atau kesimpulan numerik sebelum data final tersedia.

## 16. Workflow dan milestone

| Periode | Milestone | Keluaran wajib |
|---|---|---|
| Agustus–September 2026 | M1 — Design Freeze | dataset audit, unit, positive edge, split, candidates, metric definitions, objective sanity, novelty matrix |
| Oktober 2026 | M2 — Baselines | pipeline reproducible, BPR-MF, LightGCN, modern graph baseline, reranking baseline |
| November 2026 | M3 — Proposed Method | implementation, gradient checks, unit tests, computational audit |
| Desember 2026 | M4 — Final Experiments | 5-seed results, ablations, sensitivity, statistics, final figures/tables |
| Januari 2027 | M5 — Manuscript and Submission | 9–10 page manuscript, checklist, cover letter, submission package |
| Februari–April 2027 | M6 — Revision | response matrix, revised manuscript, reproducibility recheck |
| Mei 2027 | M7 — Proof | author proof, metadata, references, figure/table verification |
| Juni 2027 | M8 — Publication Target | publication verification and archival package |

Tanggal publication adalah target berdasarkan komunikasi saat ini, bukan jaminan. Semua deadline internal sebaiknya memiliki buffer.

## 17. Aturan kerja AI di dalam project

AI bertindak sebagai **critical research collaborator**, bukan mesin pembenaran. Setiap respons harus:

1. membedakan fakta sumber, inferensi, usulan, dan keputusan;
2. menandai status keputusan sebagai `LOCKED`, `OPEN`, `REVISED`, atau `REJECTED`;
3. menunjukkan potensi leakage, confounding, circularity, weak baseline, atau overclaim;
4. tidak mengarang sitasi, DOI, statistik, hasil, keputusan editor, atau isi paper;
5. menggunakan sumber primer untuk formula/metode dan sumber resmi untuk dataset/jurnal;
6. meminta raw evidence ketika klaim tidak dapat diverifikasi;
7. mempertahankan scope JOIN dan memperingatkan jika pekerjaan melebar ke disertasi;
8. menjaga konsistensi istilah learner, item/exercise/skill, interaction, difficulty, ability, risk, exposure, dan relevance;
9. tidak menganggap keputusan terbuka sebagai final;
10. selalu menghubungkan perubahan metode dengan dampaknya pada RQ, eksperimen, dan klaim.

Jika usulan penulis lemah secara metodologis, AI harus menjelaskannya secara langsung dan menawarkan alternatif yang dapat diuji.

## 18. Sistem dokumentasi dan reproducibility

Pertahankan dokumen berikut selama project:

- `CONTEXT.md`: konteks induk ini;
- `DECISIONS.md`: tanggal, keputusan, alasan, dampak, dan status;
- `DATA_AUDIT.md`: raw source, checksum, filtering, descriptive statistics, exclusions;
- `DESIGN_FREEZE.md`: unit, split, graph semantics, candidates, objective, baselines, metrics;
- `EXPERIMENT_LOG.md`: config ID, commit/hash, seed, runtime, output path, status;
- `RESULTS_LEDGER.md`: setiap angka/tabel/plot dan sumber outputnya;
- `MANUSCRIPT_CHANGELOG.md`: perubahan naskah dan alasan;
- `REVIEW_RESPONSE.md`: komentar reviewer, respons, lokasi perubahan, bukti.

Gunakan konfigurasi versioned dan penamaan run deterministik. Jangan menyalin angka secara manual jika tabel dapat dihasilkan dari pipeline.

## 19. Sumber yang perlu dimasukkan ke project

### Masukkan sekarang

1. konteks induk project ini;
2. instruksi project versi paste-ready;
3. proposal disertasi Bab 1–3;
4. SLR pedagogically aware educational recommender systems;
5. paper LightGCN 2020;
6. paper/dokumen dataset yang sudah tersedia dan relevan.

### Tambahkan saat tersedia

1. raw ASSISTments 2012–2013 beserta data dictionary;
2. paper KDD 2024 difficulty-aware learning path;
3. template dan author guidelines JOIN terbaru;
4. dataset audit dan design freeze;
5. kode, konfigurasi, dan hasil eksperimen;
6. manuscript submission version dan reviewer correspondence.

Jangan memenuhi project dengan materi disertasi yang tidak memengaruhi paper ini.

## 20. Hierarki sumber kebenaran

Jika terdapat konflik, gunakan urutan berikut:

1. keputusan terbaru yang eksplisit dari penulis;
2. `DESIGN_FREEZE.md` yang telah disetujui;
3. `DECISIONS.md`;
4. konteks induk ini;
5. raw data dan dokumentasi resmi;
6. proposal/SLR lama;
7. asumsi AI.

Keputusan baru tidak boleh diam-diam menimpa keputusan lama. Catat sebagai `REVISED`, jelaskan alasan dan dampaknya.

## 21. Keputusan saat project dibuat

### LOCKED

- Jurnal target: JOIN, edisi target Juni 2027.
- Paper harus berbahasa Inggris dan empiris.
- Backbone utama: LightGCN.
- Arah kontribusi: training-integrated asymmetric difficulty regularization.
- Fokus evaluasi: ranking utility dan overchallenge risk.
- Dataset rencana: ASSISTments 2012–2013.
- RQ1–RQ3 wajib; RQ4 opsional.
- iJET tidak lagi menjadi target.

### OPEN

- exact ASSISTments raw variant;
- learner–exercise atau learner–skill;
- definisi edge/positive interaction;
- temporal split granularity dan candidate universe;
- ability estimator dan cold-start shrinkage;
- candidate approximation untuk expected risk;
- temperature, τ, dan rentang `lambda_d`;
- modern graph baseline;
- feasibility full-ranking;
- final significance test;
- final title setelah hasil.

### REJECTED

- objective yang menambahkan risk konstan terhadap parameter model;
- full-history aggregation sebelum temporal split;
- incorrect response sebagai negative preference tanpa validasi;
- klaim learning outcome dari evaluasi offline;
- scope creep ke prerequisite, Bloom, AHP, Moodle, atau field study.

## 22. Tugas pertama di project baru

Mulai dari **Stage 1 — Design Freeze**. Jangan langsung menulis Introduction.

Prompt pembuka yang disarankan:

> Baca konteks induk dan seluruh sumber project. Mulai Stage 1 — Design Freeze. Pertama, susun dataset-audit checklist untuk exact ASSISTments raw variant dan decision table untuk memilih learner–exercise versus learner–skill. Jangan menetapkan keputusan tanpa statistik raw data. Tandai semua keluaran sebagai FACT, INFERENCE, PROPOSAL, atau DECISION, lalu sebutkan bukti yang masih diperlukan.

## 23. Checklist sebelum submission

- [ ] Scope dan author guidelines JOIN diverifikasi ulang dari situs resmi.
- [ ] Status pengindeksan jurnal yang relevan diverifikasi ulang sebelum submission/pembayaran.
- [ ] Manuskrip sesuai template, bahasa, dan batas halaman terbaru.
- [ ] Seluruh RQ dijawab oleh hasil aktual.
- [ ] Tidak ada train–validation–test leakage.
- [ ] Baseline adil dan tuning budget terdokumentasi.
- [ ] Minimal lima seed dan statistik ketidakpastian tersedia.
- [ ] Seluruh tabel/plot dapat diregenerasi.
- [ ] Klaim novelty didukung novelty matrix dan prior-art search terkini.
- [ ] Tidak ada fabricated citation, DOI, data, atau result.
- [ ] Limitations dan threats to validity eksplisit.
- [ ] Metadata penulis, afiliasi, funding, conflict of interest, dan data/code availability benar.
- [ ] Komunikasi EiC tidak ditulis sebagai acceptance sebelum ada keputusan resmi.


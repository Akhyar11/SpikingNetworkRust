# Audit Report (Fase 0): Delta BPTT, MIDI 2026 #3450

Semua jawaban berasal dari pembacaan kode di repo ini (branch `dev`) dan dari berkas hasil yang ada di `experiment/file_model/`. Belum ada eksperimen baru yang dijalankan. Angka di laporan ini hanya angka yang bisa dibaca atau dihitung langsung dari berkas.

## Ringkasan temuan yang mengubah arah paper

1. **Pooler tidak pernah dilatih oleh Delta BPTT.** `train_step_distill_impl` (`src/models/sentence_embedder/train.rs:150-267`) tidak memanggil `pooler.learn_through_time`. Fungsi itu hanya didefinisikan di `src/layers/dense_bptt.rs:127`. Kernel pooler diinisialisasi sebagai matriks identitas (`train_distil_only.rs:56-61`, `full_eval_controlled.rs:45-49`) dan tetap identitas. Pelatihan hanya mengubah **tabel embedding** lewat `embedding.backward`.
2. **Tidak ada AdamW.** Repo tidak memuat optimizer AdamW. Update bobotnya adalah `w += lr * g` dengan clipping ke [-1, 1] (`src/core/delta.rs:42-47, 85-91`). Learning rate "2.0" itu nyata: cosine dari 2.0 ke 0.02 (`train_distil_only.rs:164-168`). Nilainya besar karena gradiennya tidak dinormalisasi dan bobotnya di-clip.
3. **Gradien tidak sama dengan BPTT standar.** Lihat A5. Jadi framing M1 ("implementasi eksak dari surrogate-gradient BPTT") **tidak bisa dipertahankan** untuk kode ini.
4. **Parameter total jauh dari 65K.** 65.536 adalah matriks pooler 256×256. Tabel embedding model HF v2 adalah 32.000×256 = 8,19M. Tabel embedding di Rust memakai vocab BPE multibahasa 64.000 token, jadi 64.000×256 = 16,4M. Lihat A1.
5. **Angka "STS-B" diambil dari set dev, bukan test.** Berkas `sts-b_valid.json` berisi 1.500 pasangan. Itu ukuran dev STS-B, sedangkan test berukuran 1.379.
6. **Ada tumpang tindih data.** Dataset training 250K memuat kalimat yang sama dengan evaluasi: 400 dari 2.909 kalimat unik STS-B-dev (13,8%), 85 kalimat STS-15, dan 98 kalimat SICK-R. Lihat A7.
7. **Pemilihan konfigurasi memakai STS-B.** `full_eval_controlled.rs:398-476` memilih `best_method` dan `best_d_model` dari Pearson STS-B (`stsb_score`). Lihat A8.
8. **Attention dimatikan saat training distilasi** (`embedder.set_use_attention(false)`, `train_distil_only.rs:152`). Kolom "Attention D=256" di Tabel 2 berasal dari jalur lain (`full_eval_controlled.rs`, fase 3), dengan aturan gradien XOR yang bukan gradien sebenarnya.
9. **Metrik di paper adalah Pearson, bukan Spearman.** Ada tambahan clamp `sim.max(0)` pada cosine (`train_distil_only.rs:80`). Berkas `gold_metrics_snn_v2.json` mencatat Spearman yang lebih rendah. Contoh STS-12: Pearson 0,5343 vs Spearman 0,4548. Rata-rata tertimbang Spearman = 0,4788 vs Pearson 0,5038.

Menurut instruksi di `revisi.md`, temuan 1, 3, 4, 5, dan 6 adalah jenis temuan yang menghentikan pekerjaan untuk meminta konfirmasi. Saya berhenti di sini.

---

## Jawaban per pertanyaan

### A1. Total parameter
- Pooler: kernel 256×256 = **65.536**, bias tidak dipakai (`use_bias=false`, `sentence_embedder/mod.rs:42-50`). `count_params` pooler = kernel + bias (`dense_bptt.rs:225-227`). Ditambah beta dan threshold per neuron (2×256), yang tidak dihitung.
- Embedding: `vocab × 256`.
  - Rust: `vocab_multilingual.json` berisi 64.000 token, jadi **16.384.000**.
  - Ekspor HF v2: `config.json` `vocab_size=32000`, `embedding_weights [32000,256]` = **8.192.000**.
- Attention (hanya varian attention): lihat `self_attention.rs`, belum dihitung rinci di audit ini. Klaim paper "~196K" belum saya verifikasi.
- Jadi angka "65K" **mengecualikan embedding**. Total non-embedding ≈ 66K, total keseluruhan ≈ 8,26M (HF) atau 16,45M (Rust). "770× lebih sedikit parameter" dibanding SpikingBERT 50M tidak berlaku. Rasio yang benar sekitar 6× (HF v2) dan 3× (Rust).
- Paper juga menyebut tokenizer "WordPiece", sedangkan kode memuat BPE (`src/core/bpe.rs`). Ini perlu dikoreksi.

### A2. β dilatih atau tetap, dan bentuknya
- Pooler: `beta[i] = 1 - 1/2^shift`, shift acak 2..5 (`dense_bptt.rs:56-60`). Embedding: shift 2..7 (`embedding.rs:34-37`).
- β **tidak dilatih**. `learn_through_time` hanya memakainya sebagai pengali, dan tidak ada update ke β.
- β diterapkan sebagai **perkalian float**: `(*p * b) + d` (`lifStep.rs:~28`). Tidak ada shift-add. Di backward, `masked_error_data[idx] * self.beta[i]` (`dense_bptt.rs:~164`).
- Ekspor HF v2 memakai β homogen 0,9 (`config.json: "beta": 0.9`, `neuron_type: homogeneous_lif`). Itu bukan bentuk 1−2⁻ᵏ, dan tidak sama dengan model heterogen yang dideskripsikan paper. Perlu dipastikan bobot mana yang dipakai untuk angka di paper.
- Dampak: klaim "0 multiplications" gugur untuk β, kecuali E8 membuktikan sebaliknya.

### A3. Pooler: threshold, spike, reset
- Pooler punya threshold dan soft reset (`lifStep.rs`: `*p = current_p - t` saat spike). Potensial di-clip atas ke 1,0: `min(1.0, p*b + d)`. Ini berbeda dari Persamaan 1 di paper, yang meng-clip sebelum mengurangi threshold.
- **Embedding kalimat memakai `history_potentials` (potensial pra-reset, `pot_at_t`)**, bukan spike (`train.rs:~216`, `encode.rs:~89`). Jadi E = Σ_t V̂ᵗ_pool, di mana V̂ adalah potensial setelah clip dan sebelum reset.
- Di pooler, σ′ dihitung dari `history_potentials[t]` dengan jendela persegi |V−θ|<1 (`dense_bptt.rs:~145`, `surrogate.rs`). Artinya σ′ ada di kode. Tetapi, karena `learn_through_time` tidak dipanggil di training (temuan 1), σ′ pooler **tidak pernah dipakai**. Sebagai gantinya, `maskSurrogate` hanya dipanggil di `embedding.backward` dengan potensial dan threshold embedding.
- Kernel pooler identitas, jadi input pooler = spike embedding apa adanya, dan pooler praktis berperan sebagai integrator LIF per kanal.

### A4. Inventaris perkalian per langkah training
Forward:
- Embedding: lookup baris (tanpa perkalian), lalu LIF: `p*b` (1 perkalian float per neuron per waktu).
- Pooler: `dot_product_add_only` (penjumlahan bersyarat, `dotProductAddOnly.rs`), lalu LIF `p*b`. Dengan kernel identitas, ini secara efektif tidak berubah.
- Normalisasi: mean, selisih, kuadrat (D perkalian), `sqrt` (1), pembagian D kali per kalimat.
Loss (`poolerDistillation`): dot produk D perkalian per pasang, `error * b_val * margin` = 2 perkalian per elemen.
Backward:
- `applyEmbeddingDelta`: `learning_rate * acc_err` = 1 perkalian per elemen yang disentuh.
- `maskSurrogate`: hanya perbandingan.
- (Tidak terpanggil di training: `learn_through_time` memuat `inp * err` di `delta.rs:30`, yaitu **perkalian**, bukan penjumlahan murni. Binary input membuat `inp` selalu 1,0, tapi kodenya tetap mengalikan. Selain itu `error_wrt_inputs` di `dense_bptt.rs:~170-180` adalah matmul penuh `masked_error × kernel`, yakni MAC padat.)
Optimizer: tidak ada AdamW. Yang ada SGD dengan clip: 1 perkalian per bobot yang diperbarui.
- Kesimpulan: forward dan backward **mengandung perkalian float**. Klaim "0 multiplications" tidak berlaku untuk keseluruhan langkah training.

### A5. Gradien loss, normalisasi L2, dan peran m
- Normalisasi adalah z-score + L2: `(x - mean)/sqrt(Σ(x-mean)²)` (`train.rs:~228-240`).
- Gradien loss (`contrastiveHebbian.rs`, `poolerDistillation`): `err[a] += error * b_val * margin`, `err[b] += error * a_val * margin`, dengan `error = target - sim`. Ini adalah gradien dari ½(target−sim)² terhadap vektor **ternormalisasi**, tanpa Jacobian proyeksi (I − ẼẼᵀ)/‖E‖ dan tanpa Jacobian centering. Jadi **Jacobian normalisasi diabaikan**.
- Tanda: untuk loss L = ½(t−s)², dL/dã = −(t−s)·b̃. Kode menambahkan `+error*b_val`, dan `embedding.backward` kemudian **menambah** gradien ke bobot (`w += lr*g`). Secara tanda ini konsisten sebagai descent, tetapi bukan gradien eksak.
- Error yang sama di-broadcast ke semua langkah waktu `s < actual_length` (`train.rs:~245-265`), tanpa propagasi temporal lewat pooler.
- `m = 0,5` hanyalah faktor skala konstan pada gradien dan loss (`margin`). Bukan margin kontrastif, sehingga frasa "contrastive dampening margin" di paper menyesatkan. Karena lr cosine 2,0 → 0,02 dan margin 0,5 saling mengalikan, m efektif memperkecil lr.

### A6. Learning rate
- Nilai 2,0 nyata, dengan cosine annealing ke 0,01×2,0 (`train_distil_only.rs:164-168`). Bukan typo. Tetapi **bukan AdamW**, melainkan SGD tanpa momentum dengan clip bobot ±1. Teks paper harus diganti.
- Batch: 32 pasang (64 kalimat), 10 epoch, 250.000 pasang → 7.810 langkah/epoch (`train_distil_only.rs:128-132`; catatan: skrip itu masih menunjuk `dataset_augmented_5000_scored.json` yang berisi 295.000 pasang, sedangkan `full_eval_controlled.rs:200` memakai `teacher_distillation_dataset_scored.json` dengan **250.000** pasang).

### A7. Pembentukan data 250K, distribusi skor, kebocoran
- File: `experiment/file_model/teacher_distillation_dataset_scored.json`, 250.000 pasang, 351.721 kalimat unik, diskor MiniLM-L6-v2 lewat `pre-dataset/generate_teacher_dataset.js`. Skrip itu membuat varian word-dropout dan substring (`createWordDropout`, `createSubstring`) dan memuat data manusia STS-B (`loadHumanDatasets` memuat `sts-b_train.json` dan `data_stsb.train.modified_indo.csv`). Deskripsi paper "Wikipedia, BookCorpus, OpenSubtitles" belum terbukti dari kode yang saya baca (asal kalimat belum saya telusuri sampai ke sumber mentah).
- Histogram skor teacher (bin: <0, 0-.1, .1-.2, …, .9-1.0): `0, 9733, 12870, 21744, 30119, 37071, 42442, 41304, 31225, 17331, 6161`, rata-rata 0,516. Tidak ada skor negatif sama sekali (kemungkinan dipotong ke ≥0).
- Tumpang tindih (setelah lowercase + normalisasi spasi), jumlah kalimat evaluasi yang juga ada di training:

| Set | Pasangan | Kalimat unik | Beririsan dengan train |
|---|---|---|---|
| STS-B dev (`sts-b_valid`) | 1.500 | 2.909 | **400 (13,8%)** |
| STS-12 | 3.108 | 4.946 | 0 |
| STS-13 | 1.500 | 2.638 | 0 |
| STS-14 | 3.750 | 6.351 | 0 |
| STS-15 | 3.000 | 5.170 | 85 |
| STS-16 | 1.186 | 1.867 | 0 |
| SICK-R | 9.927 | 6.077 | 98 |
| STS-B train (referensi) | 5.749 | 10.518 | 960 |

  Karena training memuat data STS-B train, domain STS-B jelas terwakili. Ini menjelaskan sebagian besar anomali STS-B (0,765 vs 0,37-0,53 di subset lain). Penyebab kebocoran 400 kalimat dev belum saya telusuri: kemungkinan STS-B train dan dev berbagi kalimat sumber, atau dataset dibangun dari sumber yang mencakup keduanya.
- Overlap kalimat belum tentu overlap pasangan. Overlap pasangan belum saya hitung.

### A8. Pemilihan berbasis STS-B
- Ya. `full_eval_controlled.rs:~440-476`: fase 1 memilih metode terbaik dan fase 2 memilih d_model terbaik berdasarkan `stsb_score` (Pearson STS-B dev). Pemilihan terjadi pada set yang sama yang dilaporkan sebagai hasil. Tidak ada early stopping atau checkpoint terpisah yang saya temukan.
- Angka D=64 di Tabel 2 paper identik dengan hasil `Phase1_Distillation` di `full_eval_controlled.json` (STS-12 0,4166, STS-13 0,3106, dst., STS-B 0,6656). Jadi satu kolom di tabel berasal dari fase seleksi.

### A9. Definisi sparsity s ≈ 0,67
- Metrik spike dihitung di `encode.rs`: `val > 0.0` pada output tensor `[batch×max_seq_len, D]` **termasuk posisi padding** karena iterasi mencakup seluruh `emb_out` (`encode.rs:~33-36`). Jadi token pad ikut masuk penyebut. Namun bobot pad = −1,0 sehingga tidak menembak, yang memperbesar kesan sparsity.
- `s ≈ 0,67` belum bisa saya telusuri ke satu rumus tertentu di kode. Dibutuhkan eksekusi untuk menghitung ulang (akan dilakukan di Fase 1).
- Anomali D=128 vs D=256 (3.358 vs 2.157 spike, 2.025 s vs 1.165 s): kemungkinan penyebab adalah inisialisasi berbeda. Tiap d_model memakai seed tetap 42 tetapi rentang β/threshold dan pemuatan `init_weights` ikut berubah (`train_distil_only.rs:122-125` hanya memakai init terkontrol bila d_model cocok). Belum terverifikasi, perlu eksperimen. Selain itu waktu training dari satu run tanpa seed lain tidak bisa dibandingkan.

### A10. Baseline LSTM
- `baseline_lstm.py`: **JAX + optax**, komentar kode menyebut Kaggle GPU T4. Embedding 32.000×256 (8,19M) + LSTM (512×1024 + 1024 = 525.312), total sekitar 8,72M. LSTM ini dilatih di GPU dengan Adam, sedangkan Delta BPTT di CPU Rust. Perbandingan kecepatan tidak setara platform. Pembanding 0,7326 memakai kode evaluasi dan metrik yang belum saya verifikasi.

### A11. Sumber GloVe 0,555 dan Word2Vec 0,725
- Dari `experiment/file_model/baseline_results.json` (dibuat `experiment/baseline_nonSNN.py`): GloVe-300d STS-B Pearson 0,5549, Word2Vec-300d 0,7249, dievaluasi pada **STS-B dev (1.500 pasangan)**, mean pooling, teks cased. Ini bukan angka baku SBERT (GloVe avg STS-B test = 58,02 Spearman). Word2Vec 0,725 pada dev sangat tinggi dibanding angka SBERT-style, dan perlu divalidasi ulang dengan protokol Fase 1.
- Pearson dan set dev ≠ Spearman pada test, jadi angka-angka ini tidak sebanding dengan angka SpikingBERT dan DistilBERT.

---

## Dampak pada klaim paper

| Klaim | Status berdasarkan kode |
|---|---|
| Delta BPTT = aturan update add-only, 0 MAC | Tidak benar: ada perkalian di LIF, normalisasi, loss, update; backward ke input memakai matmul padat; dan pooler tidak dilatih |
| Gradien identik dengan BPTT standar | Tidak benar: Jacobian normalisasi diabaikan, error di-broadcast tanpa propagasi temporal pooler |
| AdamW lr=2,0 | Salah: SGD+clip, lr cosine 2,0→0,02 |
| 65K parameter, 770× lebih sedikit | Menyesatkan: embedding 8,19M (HF) / 16,4M (Rust) tidak dihitung |
| 93,4% SpikingBERT | Tidak sebanding: Pearson vs Spearman, dev vs test, zero-shot bi-encoder vs cross-encoder fine-tuned |
| STS-B 0,765 zero-shot | STS-B dev, dengan training yang memuat STS-B train dan 400 kalimat dev beririsan |
| Spearman | Paper melaporkan Pearson (+clamp); Spearman lebih rendah (rata-rata tertimbang 0,4788 vs 0,5038) |
| GloVe 0,555 / Word2Vec 0,725 | Dari STS-B dev dan Pearson buatan sendiri, bukan angka baku |
| Speedup 4,13× / 2,09× | Hanya benar untuk kode ini (kemungkinan varian attention yang lambat), bukan untuk LSTM; platform tidak setara |

## Pertanyaan keputusan sebelum Fase 1 sampai 3

1. **Arah paper.** Karena pooler tidak dilatih dan gradiennya bukan BPTT eksak, pilih satu:
   - (a) Menulis ulang paper secara jujur sebagai "event-driven sparse update untuk melatih tabel embedding spiking dengan LIF pooler tetap", dengan judul dan klaim yang jauh lebih sempit. Ini yang paling cepat dan paling jujur.
   - (b) Memperbaiki kode dulu: benar-benar melatih pooler dengan `learn_through_time` dan gradien melalui normalisasi, lalu memverifikasi kesetaraan dengan BPTT standar (misalnya cek numerik terhadap PyTorch) sebelum klaim "eksak". Ini jauh lebih besar.
2. **Kebocoran.** Training ulang tanpa kalimat yang beririsan dengan semua set evaluasi (dan tanpa STS-B train bila ingin STS-B benar-benar zero-shot)? Rekomendasi saya: ya, dan pakai STS-B **test** untuk pelaporan, dev hanya untuk seleksi.
3. **Model mana yang menjadi model utama**: bobot Rust heterogen (vocab BPE 64K) atau ekspor HF v2 (β homogen 0,9, vocab 32K)? Keduanya berbeda dari deskripsi di paper.

Saya menunggu jawaban ini sebelum menjalankan eksperimen (Fase 2) atau menulis ulang paper (Fase 3).

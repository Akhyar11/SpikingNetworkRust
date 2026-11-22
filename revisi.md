# Prompt: Revisi Paper "Delta BPTT" (MIDI 2026, paper #3450)

## Peran
Kamu adalah peneliti SNN/NLP senior sekaligus research engineer. Tugasmu memperbaiki paper **"Delta BPTT: Efficient Add-Only Backpropagation Through Time for Spiking Sequence Representations"** (Muhammad Akhyar) sampai lolos review yang ketat. Kerjakan dengan cara mengaudit kode, menjalankan eksperimen yang kurang, lalu menulis ulang paper dengan klaim yang jujur dan bisa dipertahankan.

## Bahan yang tersedia
- PDF paper saat ini: `MIDI2026_paper__3450.pdf` (lampirkan)
- Kode Rust: https://github.com/Akhyar11/SpikingNetworkRust (clone, atau gunakan salinan lokal)
- Bobot model: https://huggingface.co/PulseNet-Labs/spiking-sentence-embedder-v2
- Teacher: `sentence-transformers/all-MiniLM-L6-v2`
- Dataset evaluasi (publik): STS12–16, STS-B, SICK-R (via MTEB / SentEval / Hugging Face)
- Hardware: laptop Intel i5-1135G7, RAM 16 GB, tanpa GPU (GPU T4 opsional, hanya untuk baseline)

## Aturan wajib (jangan dilanggar)
1. **Jangan pernah mengarang angka.** Semua angka di paper hasil revisi harus berasal dari eksperimen yang benar-benar kamu jalankan, atau dari sumber yang dikutip lengkap dengan tabel/halamannya. Kalau sebuah eksperimen tidak bisa dijalankan, tulis `[TODO: belum dijalankan]` dan jelaskan alasannya.
2. **Kalau hasil baru lebih buruk dari klaim lama, laporkan apa adanya** dan sesuaikan klaim di paper. Tujuannya paper yang jujur, bukan angka yang tinggi.
3. **Sebelum mengubah klaim, periksa kodenya dulu.** Beberapa masalah di bawah masih berupa dugaan. Konfirmasi atau bantah dari kode, lalu catat hasilnya.
4. Paper ditulis dalam **bahasa Inggris**, format **LNCS**, dan tidak boleh melebihi batas halaman konferensi (saat ini 8 halaman termasuk referensi; cek CFP MIDI 2026).
5. Semua eksperimen baru dijalankan dengan **minimal 3 seed**. Laporkan rata-rata ± std.
6. Metrik utama STS adalah **Spearman ρ × 100** (cosine similarity, protokol "all" seperti SBERT/SimCSE). Pearson boleh ditambahkan sebagai pelengkap.

---

## Fase 0: Audit kode (kerjakan pertama, laporkan sebelum lanjut)
Jawab setiap pertanyaan di bawah berdasarkan kode, sertakan file dan baris rujukannya:

| # | Pertanyaan | Tujuan |
|---|---|---|
| A1 | Berapa **total parameter** model D=256, dipecah menjadi embedding, pooler, dan lainnya? Apakah angka "65K" mengecualikan tabel embedding WordPiece (sekitar 30.522 × 256 ≈ 7,8M)? | Klaim "770× lebih sedikit parameter" |
| A2 | Apakah β (decay) **dilatih** atau tetap? Apakah nilainya persis berbentuk 1 − 2⁻ᵏ dan diimplementasikan sebagai shift, atau sebagai perkalian float? | Klaim "0 multiplications" |
| A3 | Apakah pooler punya **threshold, spike, dan reset**? Kalau E = Σ V_pool, dari mana σ′(V_pool) di Persamaan 3 berasal? | Konsistensi metode |
| A4 | Inventaris **semua operasi perkalian** dalam satu langkah training: forward, loss, normalisasi, backward (Persamaan 3 dan 4), dan optimizer AdamW. | Hitungan operasi yang jujur |
| A5 | Apakah gradien loss menyertakan turunan melalui normalisasi L2 (proyeksi (I − ẼẼᵀ)/‖E‖)? Apa peran "m = 0.5" sebenarnya? | Kebenaran gradien |
| A6 | Berapa **learning rate** sebenarnya (paper menulis AdamW lr = 2.0)? | Kemungkinan typo |
| A7 | Bagaimana **250K pasangan kalimat** dibentuk? Bagaimana distribusi skor teacher-nya (histogram)? Apakah ada kalimat STS/SICK yang ikut masuk data training? | Kebocoran data |
| A8 | Apakah ada hyperparameter, checkpoint, atau early stopping yang dipilih berdasarkan **STS-B** (dev atau test)? | Anomali STS-B |
| A9 | Bagaimana s ≈ 0.67 dihitung? Apakah token pad ikut dihitung? Kenapa D=256 punya spike lebih sedikit dan lebih cepat daripada D=128 (Tabel 2)? | Konsistensi Tabel 2 |
| A10 | Baseline LSTM: framework apa, di hardware apa, berapa parameternya? | Perbandingan kecepatan yang adil |
| A11 | Sumber angka GloVe 0.555 dan Word2Vec 0.725 di Tabel 3? | Validitas baseline |

**Output Fase 0:** `audit_report.md` berisi jawaban, bukti dari kode, dan dampaknya terhadap setiap klaim di paper.

---

## Fase 1: Perbaiki pengukuran
1. **Evaluasi ulang standar.** Buat skrip evaluasi Python (`eval_sts.py`) yang memuat embedding dari model Rust (lewat ekspor atau binding) dan menghitung Spearman/Pearson untuk STS12, 13, 14, 15, 16, STS-B (test), dan SICK-R (test). Gunakan protokol "all" dan cosine similarity. Validasi skripnya dengan mereproduksi angka avg-GloVe di paper SBERT (55.14 / 70.66 / 59.73 / 68.25 / 63.66 / 58.02 / 53.76) dengan toleransi sekitar 1 poin.
2. **Tabel parameter lengkap:** total, embedding, dan non-embedding untuk model sendiri, SpikingBERT, Spikingformer (versi NLP), DistilBERT, dan MiniLM.
3. **Hitungan operasi dan energi yang jujur** untuk satu langkah training dan satu inferensi. Pisahkan: AC sparse, MAC float, operasi non-linear (sqrt/div), dan operasi optimizer. Hitung ulang klaim 15.5× dan batasi cakupannya ke "akumulasi ΔW" saja. Tambahkan catatan bahwa akses memori tidak dihitung.
4. Laporkan **skor teacher** all-MiniLM-L6-v2 dengan protokol yang sama (ganti ">0.85" dengan angka pasti).

## Fase 2: Eksperimen baru
| # | Eksperimen | Pertanyaan yang dijawab |
|---|---|---|
| E1 | **Baseline parameter-matched:** (a) mean-pool embedding yang sama tanpa LIF, (b) mean-pool embedding statis teacher, (c) avg GloVe, (d) GRU/BiLSTM kecil dengan parameter setara, semua dievaluasi dengan protokol Fase 1 | Apa kontribusi LIF di atas mean pooling biasa? |
| E2 | **ANN twin:** arsitektur yang sama persis, tetapi aktivasi kontinu (tanpa spike) dan dilatih dengan backprop biasa | Biaya akurasi dari spiking |
| E3 | **Shuffle test:** acak urutan token kalimat test, lalu ukur penurunan Spearman | Apakah model benar-benar memakai urutan kata? |
| E4 | **Investigasi STS-B:** jelaskan kenapa STS-B jauh di atas subset lain (cek leakage, overlap domain dengan data training, dan pemilihan checkpoint) | Validitas angka utama |
| E5 | **Ablasi objektif ulang:** distill vs regresi label manusia vs SimCSE, semuanya dievaluasi terhadap **label manusia** (bukan label teacher) | Menghilangkan sirkularitas |
| E6 | **Kecepatan yang adil:** LSTM dan Delta BPTT di platform yang sama (CPU, implementasi setara), plus versi dense-matmul dari model yang sama untuk mengisolasi efek gating | Klaim speedup |
| E7 | **Ablasi attention yang lebih adil:** terapkan juga spike-gating di backward attention; kalau memungkinkan tambahkan satu attention spiking standar (misalnya ala Spikformer, ukuran kecil) | Klaim "attention redundan" |
| E8 | **Variasi β:** β tetap 1 − 2⁻ᵏ (shift-only) vs β float yang dilatih | Apakah klaim bebas-perkalian bisa dipertahankan tanpa kehilangan akurasi? |
| E9 | (Opsional) Panjang sekuens > 128 | Batasan yang disebut di Limitations |

Kalau waktu atau komputasi tidak cukup, prioritasnya: **E1, E3, E4, E5, E2**, baru sisanya.

## Fase 3: Tulis ulang paper
Terapkan semua perbaikan berikut, dengan isi yang disesuaikan pada hasil Fase 0–2:

1. **Framing kontribusi.** Delta BPTT diposisikan sebagai *implementasi event-driven yang eksak dari surrogate-gradient BPTT*, yang mengeksploitasi spike presinaptik biner untuk akumulasi gradien bobot yang sparse. Jangan menyebutnya aturan belajar baru. Nyatakan eksplisit bahwa gradiennya identik dengan BPTT standar.
2. **Klaim "0 MACs / 0 multiplications"** di judul, abstrak, Fig. 1, Tabel 1, dan kesimpulan dipersempit ke bagian yang memang add-only, sesuai hasil A2/A4/E8. Kalau perlu, ganti judul. Contoh: *"Sparse Event-Driven Gradient Accumulation for Training Tiny Spiking Sentence Embedders on Commodity CPUs"*.
3. **Hapus atau koreksi** "770× fewer parameters" dan "93.4% of SpikingBERT". Kalau dipertahankan, gunakan total parameter dan metrik yang sama, serta jelaskan perbedaan setting (bi-encoder zero-shot vs cross-encoder yang di-fine-tune di train set STS-B).
4. **Tabel 3 baru:** semua angka Spearman, kolom total parameter, kolom "setting" (zero-shot bi-encoder / fine-tuned cross-encoder), dan kolom "Hardware" dihapus. Setiap angka eksternal diberi sitasi tabel yang tepat.
5. **Tabel 2 baru:** Spearman per subset dengan mean ± std, ditambah baseline E1 di sebelahnya. Jelaskan anomali spike/waktu di D=128 vs D=256.
6. **Klaim tentang attention dan urutan kata** disesuaikan dengan hasil E3 dan E7. Kalau shuffle test menunjukkan model tidak sensitif urutan, hapus kalimat "*recurrent LIF dynamics inherently encode sufficient contextual order*" dan akui hal itu di Limitations.
7. **Klaim "outperforms GloVe/Word2Vec"** hanya boleh dipertahankan untuk subset tempat model memang unggul.
8. **Bagian metode diperjelas:** dinamika pooler (sesuai A3), gradien loss yang benar (A5), learning rate yang benar (A6), cara pembentukan data (A7), definisi sparsity (A9).
9. **Energi:** Tabel 1 dan teks memakai hitungan operasi lengkap dari Fase 1.3, dengan catatan eksplisit bahwa ini estimasi 45 nm yang tidak menghitung akses memori. Hapus perbandingan 1,298.8× vs Transformer, atau beri catatan yang tegas.
10. **Limitations diperluas:** setting zero-shot bi-encoder, ketergantungan pada teacher, estimasi energi yang hanya teoretis, sensitivitas urutan (kalau terbukti lemah), dan sekuens ≤ 128.
11. **Referensi:** perbarui Spikingformer ke versi terbaru (judul *"Spikingformer: A Key Foundation Model for Spiking Neural Networks"*, AAAI 2026). Tambahkan sitasi untuk SentEval/MTEB, protokol SBERT, dan sumber setiap angka baseline. Self-citation [17] boleh dipertahankan, tapi jangan dijadikan satu-satunya dasar baseline attention.

## Deliverables
1. `audit_report.md`: hasil Fase 0.
2. `eval_sts.py` beserta instruksi menjalankannya; skrip eksperimen E1–E9 (Rust dan/atau Python) dengan seed yang tetap.
3. `results/`: CSV mentah semua eksperimen dan log run.
4. `paper_revised.tex` (LNCS) beserta PDF hasil kompilasi.
5. `changelog.md`: tabel **masalah → perbaikan → bukti (eksperimen/file) → bagian paper yang berubah**, mencakup semua poin di bawah.
6. `response_to_reviewers.md`: draf jawaban untuk setiap poin, seolah-olah poin-poin ini adalah komentar reviewer.

## Checklist masalah yang harus tertutup
- [ ] M1 Delta BPTT diposisikan ulang sebagai implementasi eksak, bukan aturan baru
- [ ] M2 Klaim 0 MAC dipersempit (β, normalisasi, loss, AdamW)
- [ ] M3 Parameter dihitung total, termasuk embedding
- [ ] M4 Perbandingan dengan SpikingBERT/DistilBERT memakai metrik sama dan setting dijelaskan
- [ ] M5 Perbandingan dengan GloVe memakai angka baku dan per subset; anomali STS-B dijelaskan
- [ ] M6 Klaim attention redundan dan urutan kata diuji (E3, E7) dan disesuaikan
- [ ] M7 Ablasi objektif dievaluasi pada label manusia
- [ ] M8 Perbandingan kecepatan di platform yang setara
- [ ] m1 Learning rate dijelaskan atau dikoreksi
- [ ] m2 Gradien loss dan peran m dijelaskan
- [ ] m3 Pembentukan data 250K pasangan dijelaskan
- [ ] m4 Anomali Tabel 2 dan definisi sparsity dijelaskan
- [ ] m5 Skor teacher ditulis angka pasti
- [ ] m6 Referensi Spikingformer diperbarui dan angkanya diverifikasi
- [ ] m7 Multi-seed, mean ± std
- [ ] m8 Klaim 1,298.8× dihapus atau diberi catatan

## Cara kerja
Mulai dari Fase 0 dan kirim `audit_report.md` dulu. Kalau ada temuan yang mengubah arah paper secara besar (misalnya parameter ternyata 7,8M, atau STS-B bocor), hentikan pekerjaan dan minta konfirmasi saya sebelum lanjut ke Fase 2–3. Setelah itu kerjakan fase berikutnya secara berurutan. Di akhir setiap fase, beri ringkasan singkat: apa yang dikerjakan, angka kunci, dan klaim mana yang berubah.
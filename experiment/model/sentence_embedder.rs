use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::embedding::SpikingEmbedding;
use SpikingNetworkRust::layers::self_attention::SpikingSelfAttention;
use SpikingNetworkRust::layers::dense_bptt::SpikingDenseBPTT;
use SpikingNetworkRust::layers::base::Layer;

/// Orkestrator utama untuk Pipeline Spiking Neural Network
/// Alur: Teks -> BPETokenizer -> SpikingEmbedding -> SpikingSelfAttention -> SpikingDenseBPTT -> Embedding Kalimat (L2 Norm)
pub struct SpikingSentenceEmbedder {
    pub tokenizer: BPETokenizer,
    pub embedding: SpikingEmbedding,
    pub attention: SpikingSelfAttention,
    pub pooler: SpikingDenseBPTT,
    pub max_seq_length: usize,
    pub cached_actual_lengths: Option<Vec<usize>>,
}

impl SpikingSentenceEmbedder {
    pub fn new(
        tokenizer: BPETokenizer,
        vocab_size: usize,
        d_model: usize,
        max_seq_length: usize,
    ) -> Self {
        // 1. Spiking Embedding Layer
        let embedding = SpikingEmbedding::new(
            vocab_size, 
            d_model, 
            0.01, 
            -1.0, 
            1.0
        );

        // 2. Linear Self-Attention Layer (O(N*d^2) efficiency)
        let attention = SpikingSelfAttention::new(
            d_model,
            max_seq_length,
            0.01,
            -1.0,
            1.0,
            (0.8, 0.99),
            (0.1, 0.3)
        );

        // 3. Temporal Pooling Layer (BPTT)
        let pooler = SpikingDenseBPTT::new(
            d_model,
            d_model,
            true,
            -1.0,
            1.0,
            (0.8, 0.99),
            (0.1, 0.3)
        );

        Self {
            tokenizer,
            embedding,
            attention,
            pooler,
            max_seq_length,
            cached_actual_lengths: None,
        }
    }

    /// Forward pass mengonversi teks mentah menjadi representasi semantik ruang metrik (Metric Space)
    pub fn encode(&mut self, texts: &[&str]) -> Vec<Vec<f32>> {
        let batch_size = texts.len();
        self.embedding.reset_state();
        let mut tokenized_batch = Vec::with_capacity(batch_size * self.max_seq_length);

        // Tahap 1: Tokenisasi & Padding
        let mut actual_lengths = vec![0; batch_size];
        for (b, text) in texts.iter().enumerate() {
            let mut tokens = self.tokenizer.encode(text);
            actual_lengths[b] = tokens.len().min(self.max_seq_length);
            if tokens.len() > self.max_seq_length {
                tokens.truncate(self.max_seq_length);
            }
            let mut tokens_f32: Vec<f32> = tokens.into_iter().map(|t| t as f32).collect();
            while tokens_f32.len() < self.max_seq_length {
                tokens_f32.push(0.0); // Padding dengan 0
            }
            tokenized_batch.extend(tokens_f32);
        }

        self.cached_actual_lengths = Some(actual_lengths.clone());

        // Tahap 2: Spiking Embedding (Spatial ke Temporal Spikes)
        let emb_out = self.embedding.forward(&tokenized_batch);

        // Tahap 3: Spiking Self-Attention (Pencarian Konteks Global)
        let att_out = self.attention.forward(&emb_out, &actual_lengths);

        // Tahap 4: Temporal Pooling dengan BPTT (Integrasi Waktu)
        self.pooler.reset_sequence(batch_size, self.max_seq_length);
        let mut final_embeddings = vec![vec![0.0; self.pooler.units]; batch_size];

        for t in 0..self.max_seq_length {
            let mut step_input = vec![0.0; batch_size * self.pooler.in_features];
            for b in 0..batch_size {
                if t < actual_lengths[b] {
                    let base_idx = (b * self.max_seq_length + t) * self.pooler.in_features;
                    for i in 0..self.pooler.in_features {
                        step_input[b * self.pooler.in_features + i] = att_out[base_idx + i];
                    }
                }
            }

            let spikes = self.pooler.compute_step(&step_input, t);

            // Akumulasi potensi/spike untuk mean pooling HANYA untuk token non-padding
            for b in 0..batch_size {
                if t < actual_lengths[b] {
                    for i in 0..self.pooler.units {
                        final_embeddings[b][i] += spikes[b * self.pooler.units + i];
                    }
                }
            }
        }

        // Tahap 5: Mean Pooling dan L2 Normalization (Memproyeksikan ke Permukaan Bola Semantik)
        for b in 0..batch_size {
            let len_f32 = actual_lengths[b] as f32;
            let len_f32 = if len_f32 == 0.0 { 1.0 } else { len_f32 };
            for i in 0..self.pooler.units {
                final_embeddings[b][i] /= len_f32; 
            }
            
            let mut sum_sq = 0.0;
            for val in &final_embeddings[b] {
                sum_sq += val * val;
            }
            let norm = sum_sq.sqrt().max(1e-12);
            for i in 0..self.pooler.units {
                final_embeddings[b][i] /= norm;
            }
        }

        final_embeddings
    }

    /// Melatih jaringan menggunakan sinyal kesalahan gradien yang didapat dari Contrastive Hebbian Learning
    pub fn learn(&mut self, error_signals: &[Vec<f32>]) {
        let batch_size = error_signals.len();
        let actual_lengths = self.cached_actual_lengths.as_ref().expect("Panggil encode dulu!");
        
        // 1. Backpropagate DenseBPTT Pooler
        // BPTT butuh error [time_steps][batch_size * units]
        let mut error_seq = vec![vec![0.0; batch_size * self.pooler.units]; self.max_seq_length];
        
        for b in 0..batch_size {
            let len_f32 = if actual_lengths[b] == 0 { 1.0 } else { actual_lengths[b] as f32 };
            for t in 0..self.max_seq_length {
                if t < actual_lengths[b] {
                    for u in 0..self.pooler.units {
                        // Gradien dari mean pooling: error dibagi sequence length
                        error_seq[t][b * self.pooler.units + u] = error_signals[b][u] / len_f32;
                    }
                }
            }
        }
        
        // Ambil learning rate dari parameter dasar
        let lr = self.pooler.get_base_config().learning_rate;
        let bptt_gradients = self.pooler.learn_through_time(&error_seq, lr);

        // 2. Distribusi error ke Self-Attention
        // BPTT mengembalikan [time_steps][batch_size * in_features]
        // Kita perlu meratakannya menjadi flat array [batch_size * max_seq_length * in_features]
        let mut att_errors = vec![0.0; batch_size * self.max_seq_length * self.pooler.in_features];
        for b in 0..batch_size {
            for t in 0..self.max_seq_length {
                if t < actual_lengths[b] {
                    let base_idx = (b * self.max_seq_length + t) * self.pooler.in_features;
                    for i in 0..self.pooler.in_features {
                        att_errors[base_idx + i] = bptt_gradients[t][b * self.pooler.in_features + i];
                    }
                }
            }
        }
        
        self.attention.learn_attention(&att_errors, actual_lengths);

        // 3. Distribusi error ke Spiking Embedding
        self.embedding.backward(&att_errors);
    }

    /// Menampilkan keseluruhan topologi SNN
    pub fn summary(&self) {
        println!("=============================================");
        println!("           Spiking Sentence Embedder         ");
        println!("=============================================");
        println!(" Max Sequence Length: {}", self.max_seq_length);
        println!(" Vocabulary Size    : {}", self.embedding.input_dim);
        println!(" D_Model (Units)    : {}", self.attention.d_model);
        self.embedding.summary();
        self.attention.summary();
        self.pooler.summary();
        println!("=============================================");
    }
}

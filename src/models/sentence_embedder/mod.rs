use crate::core::bpe::BPETokenizer;
use crate::layers::embedding::SpikingEmbedding;
use crate::layers::self_attention::SpikingSelfAttention;
use crate::layers::dense_bptt::SpikingDenseBPTT;
use crate::layers::base::Layer;

pub mod config;
pub mod encode;
pub mod train;

pub use config::{SNNMetrics, SNNConfig};

pub struct SpikingSentenceEmbedder {
    pub tokenizer: BPETokenizer,
    pub embedding: SpikingEmbedding,
    pub attention: SpikingSelfAttention,
    pub pooler: SpikingDenseBPTT,
    pub max_seq_length: usize,
    pub cached_actual_lengths: Option<Vec<usize>>,
    pub metrics: SNNMetrics,
    pub use_attention: bool,
}

impl SpikingSentenceEmbedder {
    pub fn new(
        tokenizer: BPETokenizer,
        vocab_size: usize,
        config: SNNConfig,
    ) -> Self {
        let embedding = SpikingEmbedding::new(
            vocab_size, 
            config.d_model, 
            config.learning_rate, 
            config.clip_min, 
            config.clip_max
        );

        let attention = SpikingSelfAttention::new(
            config.d_model,
            config.max_seq_length,
            config.learning_rate,
            config.clip_min,
            config.clip_max,
            config.att_beta_range,
            config.att_threshold_range
        );

        let mut pooler = SpikingDenseBPTT::new(
            config.d_model,
            config.d_model,
            false, // Disable recurrent bias
            config.clip_min,
            config.clip_max,
            config.bptt_beta_range,
            config.bptt_threshold_range
        );
        
        for i in 0..config.d_model {
            for j in 0..config.d_model {
                pooler.kernel[i * config.d_model + j] = if i == j { 1.0 } else { 0.0 };
            }
            pooler.bias[i] = 0.0;
        }

        Self {
            tokenizer,
            embedding,
            attention,
            pooler,
            max_seq_length: config.max_seq_length,
            cached_actual_lengths: None,
            metrics: SNNMetrics::default(),
            use_attention: true,
        }
    }

    pub fn set_use_attention(&mut self, val: bool) {
        self.use_attention = val;
    }

    pub fn calculate_sops(&mut self) -> usize {
        // SOPs = Σ(S_emb × 3d) + Σ(S_pool × d)
        let embedding_contribution = self.metrics.embedding_spikes * 3 * self.embedding.output_dim;
        let pooling_contribution = self.metrics.pooler_active_inputs * self.pooler.units;
        
        let total = embedding_contribution + pooling_contribution;
        self.metrics.total_sops = total;
        
        println!("SOP Breakdown:");
        println!("  Embedding: {} × 3 × {} = {}", 
            self.metrics.embedding_spikes, self.embedding.output_dim, embedding_contribution);
        println!("  Pooling: {} × {} = {}", 
            self.metrics.pooler_active_inputs, self.pooler.units, pooling_contribution);
        println!("  Total SOPs: {}", total);
        
        total
    }

    fn zero_pad_token(&mut self) {
        for d in 0..self.embedding.output_dim {
            self.embedding.weights[d] = -1.0;
        }
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.embedding.base.learning_rate = lr;
        self.attention.base.learning_rate = lr;
        self.pooler.base.learning_rate = lr;
    }

    pub fn encode(&mut self, texts: &[&str]) -> Vec<Vec<f32>> {
        encode::encode_impl(self, texts)
    }

    pub fn train_step(&mut self, texts: &[&str], num_pairs: usize, margin: f32) -> (f32, f32, f32) {
        train::train_step_impl(self, texts, num_pairs, margin)
    }

    pub fn train_step_distill(&mut self, texts: &[&str], targets: &[f32], margin: f32) -> (f32, f32, f32) {
        train::train_step_distill_impl(self, texts, targets, margin)
    }

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

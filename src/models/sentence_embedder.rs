use crate::core::bpe::BPETokenizer;
use crate::layers::embedding::SpikingEmbedding;
use crate::layers::self_attention::SpikingSelfAttention;
use crate::layers::dense_bptt::SpikingDenseBPTT;
use crate::layers::base::Layer;

#[derive(Default, Clone, Copy)]
pub struct SNNMetrics {
    pub embedding_spikes: usize,
    pub attention_spikes: usize,
    pub pooler_spikes: usize,
    pub total_sops: usize,
    pub total_sentences: usize,
}

#[derive(Clone, Copy)]
pub struct SNNConfig {
    pub d_model: usize,
    pub max_seq_length: usize,
    pub learning_rate: f32,
    pub clip_min: f32,
    pub clip_max: f32,
    pub att_beta_range: (f32, f32),
    pub att_threshold_range: (f32, f32),
    pub bptt_beta_range: (f32, f32),
    pub bptt_threshold_range: (f32, f32),
}

pub struct SpikingSentenceEmbedder {
    pub tokenizer: BPETokenizer,
    pub embedding: SpikingEmbedding,
    pub attention: SpikingSelfAttention,
    pub pooler: SpikingDenseBPTT,
    pub max_seq_length: usize,
    pub cached_actual_lengths: Option<Vec<usize>>,
    pub metrics: SNNMetrics,
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
        }

        Self {
            tokenizer,
            embedding,
            attention,
            pooler,
            max_seq_length: config.max_seq_length,
            cached_actual_lengths: None,
            metrics: SNNMetrics::default(),
        }
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
        self.zero_pad_token();
        let batch_size = texts.len();
        self.embedding.reset_state();
        let mut tokenized_batch = Vec::with_capacity(batch_size * self.max_seq_length);

        let mut actual_lengths = vec![0; batch_size];
        for (b, text) in texts.iter().enumerate() {
            let mut tokens = self.tokenizer.encode(text);
            actual_lengths[b] = tokens.len().min(self.max_seq_length);
            if tokens.len() > self.max_seq_length {
                tokens.truncate(self.max_seq_length);
            }
            let mut tokens_f32: Vec<f32> = tokens.into_iter().map(|t| t as f32).collect();
            while tokens_f32.len() < self.max_seq_length {
                tokens_f32.push(0.0);
            }
            tokenized_batch.extend(tokens_f32);
        }

        self.cached_actual_lengths = Some(actual_lengths.clone());
        let batch_seq = batch_size * self.max_seq_length;
        let d_model = self.embedding.output_dim;

        let emb_out = self.embedding.forward(&tokenized_batch);
        let mut emb_spikes = 0;
        for &val in &emb_out {
            if val > 0.0 { emb_spikes += 1; }
        }
        self.metrics.embedding_spikes += emb_spikes;
        self.metrics.total_sops += emb_spikes * self.attention.d_model * 3;

        let att_out = self.attention.forward(&emb_out, &actual_lengths);
        let mut att_spikes = 0;
        for &val in &att_out {
            if val > 0.0 { att_spikes += 1; }
        }
        self.metrics.attention_spikes += att_spikes;

        let mut aggregated_features = vec![0.0; batch_seq * d_model];
        for i in 0..(batch_seq * d_model) {
            aggregated_features[i] = emb_out[i] + att_out[i];
        }

        self.pooler.reset_sequence(batch_size, self.max_seq_length);
        let mut final_embeddings = vec![vec![0.0; self.pooler.units]; batch_size];

        for t in 0..self.max_seq_length {
            let mut step_input = vec![0.0; batch_size * self.pooler.in_features];
            for b in 0..batch_size {
                if t < actual_lengths[b] {
                    let base_idx = (b * self.max_seq_length + t) * self.pooler.in_features;
                    for i in 0..self.pooler.in_features {
                        step_input[b * self.pooler.in_features + i] = aggregated_features[base_idx + i];
                    }
                }
            }

            let out_spikes = self.pooler.compute_step(&step_input, t);

            for b in 0..batch_size {
                if t < actual_lengths[b] {
                    let in_base = b * self.pooler.in_features;
                    for i in 0..self.pooler.in_features {
                        if step_input[in_base + i] > 0.0 {
                            self.metrics.total_sops += self.pooler.units;
                        }
                    }

                    let offset = b * self.pooler.units;
                    for i in 0..self.pooler.units {
                        if out_spikes[offset + i] > 0.0 {
                            self.metrics.pooler_spikes += 1;
                        }
                        final_embeddings[b][i] += self.pooler.history_potentials[t][offset + i];
                    }
                }
            }
        }

        for b in 0..batch_size {
            let mut sum_sq = 0.0;
            for val in &final_embeddings[b] {
                sum_sq += val * val;
            }
            let norm = sum_sq.sqrt().max(1e-8);
            for i in 0..self.pooler.units {
                final_embeddings[b][i] /= norm;
            }
        }

        self.metrics.total_sentences += batch_size;
        final_embeddings
    }

    pub fn train_step(&mut self, texts: &[&str], num_pairs: usize, margin: f32) -> (f32, f32, f32) {
        self.zero_pad_token();
        let batch_size = texts.len();
        self.embedding.reset_state();
        self.attention.reset_state(batch_size);
        self.pooler.reset_sequence(batch_size, self.max_seq_length);

        let mut tokenized_batch = Vec::with_capacity(batch_size * self.max_seq_length);
        let mut actual_lengths = vec![0; batch_size];
        for (b, text) in texts.iter().enumerate() {
            let mut tokens = self.tokenizer.encode(&text.to_lowercase());
            actual_lengths[b] = tokens.len().min(self.max_seq_length);
            if tokens.len() > self.max_seq_length {
                tokens.truncate(self.max_seq_length);
            }
            let mut tokens_f32: Vec<f32> = tokens.into_iter().map(|t| t as f32).collect();
            while tokens_f32.len() < self.max_seq_length {
                tokens_f32.push(0.0);
            }
            tokenized_batch.extend(tokens_f32);
        }

        self.cached_actual_lengths = Some(actual_lengths.clone());

        let batch_seq = batch_size * self.max_seq_length;
        let d_model = self.embedding.output_dim;

        let spikes1 = self.embedding.forward(&tokenized_batch);
        let mut err_emb_data = vec![0.0; batch_seq * d_model];
        let loss1 = crate::core::contrastiveHebbian::contrastiveHebbian(
            &spikes1, &mut err_emb_data, num_pairs, self.max_seq_length, d_model, margin, &actual_lengths
        );
        self.embedding.backward(&err_emb_data, None);

        let att_spikes = self.attention.forward(&spikes1, &actual_lengths);
        let mut spikes2 = vec![0.0; batch_seq * d_model];
        for i in 0..(batch_seq * d_model) {
            let att_val = if att_spikes[i] > 0.5 { 1.0 } else { 0.0 };
            let combined = spikes1[i] + att_val;
            spikes2[i] = if combined > 0.5 { 1.0 } else { 0.0 };
        }

        let mut err_att_data = vec![0.0; batch_seq * d_model];
        let loss2 = crate::core::contrastiveHebbian::contrastiveHebbian(
            &spikes2, &mut err_att_data, num_pairs, self.max_seq_length, d_model, margin, &actual_lengths
        );
        self.attention.learn_attention(&err_att_data, &actual_lengths);

        let mut final_out_data = vec![0.0; batch_size * self.pooler.units];

        for t in 0..self.max_seq_length {
            let mut step_input = vec![0.0; batch_size * self.pooler.in_features];
            for b in 0..batch_size {
                if t < actual_lengths[b] {
                    let base_idx = (b * self.max_seq_length + t) * self.pooler.in_features;
                    for i in 0..self.pooler.in_features {
                        step_input[b * self.pooler.in_features + i] = spikes2[base_idx + i];
                    }
                }
            }

            let out_spikes = self.pooler.compute_step(&step_input, t);

            for b in 0..batch_size {
                if t < actual_lengths[b] {
                    let offset = b * self.pooler.units;
                    for i in 0..self.pooler.units {
                        final_out_data[offset + i] += out_spikes[offset + i];
                    }
                }
            }
        }

        let mut normalized_out_data = vec![0.0; batch_size * self.pooler.units];
        for b in 0..batch_size {
            let mut sum_sq = 0.0;
            let offset = b * self.pooler.units;
            for i in 0..self.pooler.units {
                sum_sq += final_out_data[offset + i] * final_out_data[offset + i];
            }
            let norm = sum_sq.sqrt().max(1e-8);
            for i in 0..self.pooler.units {
                normalized_out_data[offset + i] = final_out_data[offset + i] / norm;
            }
        }

        let mut error_final_data = vec![0.0; batch_size * self.pooler.units];
        let dummy_lengths = vec![1; batch_size];
        let pooler_loss = crate::core::contrastiveHebbian::contrastiveHebbian(
            &normalized_out_data, &mut error_final_data, num_pairs, 1, self.pooler.units, margin, &dummy_lengths
        );

        let mut error_seq = vec![vec![0.0; batch_size * self.pooler.units]; self.max_seq_length];
        for s in 0..self.max_seq_length {
            for b in 0..batch_size {
                if s < actual_lengths[b] {
                    let offset = b * self.pooler.units;
                    for i in 0..self.pooler.units {
                        error_seq[s][offset + i] = error_final_data[offset + i];
                    }
                }
            }
        }
        
        let _lr = self.pooler.get_base_config().learning_rate;

        (loss1, loss2, pooler_loss)
    }

    pub fn train_step_distill(&mut self, texts: &[&str], targets: &[f32], margin: f32) -> (f32, f32, f32) {
        self.zero_pad_token();
        let batch_size = texts.len();
        let num_pairs = batch_size / 2;
        
        self.embedding.reset_state();
        self.attention.reset_state(batch_size);
        self.pooler.reset_sequence(batch_size, self.max_seq_length);

        let mut tokenized_batch = Vec::with_capacity(batch_size * self.max_seq_length);
        let mut actual_lengths = vec![0; batch_size];
        for (b, text) in texts.iter().enumerate() {
            let mut tokens = self.tokenizer.encode(&text.to_lowercase());
            actual_lengths[b] = tokens.len().min(self.max_seq_length);
            if tokens.len() > self.max_seq_length { tokens.truncate(self.max_seq_length); }
            let mut tokens_f32: Vec<f32> = tokens.into_iter().map(|t| t as f32).collect();
            while tokens_f32.len() < self.max_seq_length { tokens_f32.push(0.0); }
            tokenized_batch.extend(tokens_f32);
        }

        self.cached_actual_lengths = Some(actual_lengths.clone());

        let batch_seq = batch_size * self.max_seq_length;
        let d_model = self.embedding.output_dim;

        let spikes1 = self.embedding.forward(&tokenized_batch);
        let mut err_emb_data = vec![0.0; batch_seq * d_model];
        let loss1 = crate::core::contrastiveHebbian::distillationHebbian(
            &spikes1, &mut err_emb_data, num_pairs, self.max_seq_length, d_model, margin, &actual_lengths, targets
        );
        self.embedding.backward(&err_emb_data, None);

        let att_spikes = self.attention.forward(&spikes1, &actual_lengths);
        let mut spikes2 = vec![0.0; batch_seq * d_model];
        for i in 0..(batch_seq * d_model) {
            let att_val = if att_spikes[i] > 0.5 { 1.0 } else { 0.0 };
            let combined = spikes1[i] + att_val;
            spikes2[i] = if combined > 0.5 { 1.0 } else { 0.0 };
        }

        let mut err_att_data = vec![0.0; batch_seq * d_model];
        let loss2 = crate::core::contrastiveHebbian::distillationHebbian(
            &spikes2, &mut err_att_data, num_pairs, self.max_seq_length, d_model, margin, &actual_lengths, targets
        );
        self.attention.learn_attention(&err_att_data, &actual_lengths);

        let mut final_out_data = vec![0.0; batch_size * self.pooler.units];
        for t in 0..self.max_seq_length {
            let mut step_input = vec![0.0; batch_size * self.pooler.in_features];
            for b in 0..batch_size {
                if t < actual_lengths[b] {
                    let base_idx = (b * self.max_seq_length + t) * self.pooler.in_features;
                    for i in 0..self.pooler.in_features {
                        step_input[b * self.pooler.in_features + i] = spikes2[base_idx + i];
                    }
                }
            }
            let out_spikes = self.pooler.compute_step(&step_input, t);
            for b in 0..batch_size {
                if t < actual_lengths[b] {
                    let offset = b * self.pooler.units;
                    for i in 0..self.pooler.units { final_out_data[offset + i] += out_spikes[offset + i]; }
                }
            }
        }

        let mut normalized_out_data = vec![0.0; batch_size * self.pooler.units];
        for b in 0..batch_size {
            let mut sum_sq = 0.0;
            let offset = b * self.pooler.units;
            for i in 0..self.pooler.units { sum_sq += final_out_data[offset + i] * final_out_data[offset + i]; }
            let norm = sum_sq.sqrt().max(1e-8);
            for i in 0..self.pooler.units { normalized_out_data[offset + i] = final_out_data[offset + i] / norm; }
        }

        let mut error_final_data = vec![0.0; batch_size * self.pooler.units];
        let dummy_lengths = vec![1; batch_size];
        
        let pooler_loss = crate::core::contrastiveHebbian::distillationHebbian(
            &normalized_out_data, &mut error_final_data, num_pairs, 1, self.pooler.units, margin, &dummy_lengths, targets
        );

        (loss1, loss2, pooler_loss)
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

#[derive(Default, Clone, Copy)]
pub struct SNNMetrics {
    pub embedding_spikes: usize,
    pub attention_spikes: usize,
    pub pooler_spikes: usize,
    pub pooler_active_inputs: usize,
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

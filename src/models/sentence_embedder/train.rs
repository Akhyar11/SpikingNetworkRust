use super::SpikingSentenceEmbedder;

pub fn train_step_impl(embedder: &mut SpikingSentenceEmbedder, texts: &[&str], num_pairs: usize, margin: f32) -> (f32, f32, f32) {
    embedder.zero_pad_token();
    let batch_size = texts.len();
    embedder.embedding.reset_state();
    embedder.attention.reset_state(batch_size);
    embedder.pooler.reset_sequence(batch_size, embedder.max_seq_length);

    let mut tokenized_batch = Vec::with_capacity(batch_size * embedder.max_seq_length);
    let mut actual_lengths = vec![0; batch_size];
    for (b, text) in texts.iter().enumerate() {
        let mut tokens = embedder.tokenizer.encode(&text.to_lowercase());
        actual_lengths[b] = tokens.len().min(embedder.max_seq_length);
        if tokens.len() > embedder.max_seq_length {
            tokens.truncate(embedder.max_seq_length);
        }
        let mut tokens_f32: Vec<f32> = tokens.into_iter().map(|t| t as f32).collect();
        while tokens_f32.len() < embedder.max_seq_length {
            tokens_f32.push(0.0);
        }
        tokenized_batch.extend(tokens_f32);
    }

    embedder.cached_actual_lengths = Some(actual_lengths.clone());

    let batch_seq = batch_size * embedder.max_seq_length;
    let d_model = embedder.embedding.output_dim;

    let spikes1 = embedder.embedding.forward(&tokenized_batch);
    
    let mut spikes2 = vec![0.0; batch_seq * d_model];
    let att_spikes = if embedder.use_attention {
        embedder.attention.forward(&spikes1, &actual_lengths)
    } else {
        vec![0.0; batch_seq * d_model]
    };

    if embedder.use_attention {
        for i in 0..(batch_seq * d_model) {
            let spk1 = if spikes1[i] > 0.5 { 1.0 } else { 0.0 };
            let att_val = if att_spikes[i] > 0.5 { 1.0 } else { 0.0 };
            spikes2[i] = if spk1 != att_val { 1.0 } else { 0.0 };
        }
    } else {
        spikes2.copy_from_slice(&spikes1);
    }

    let mut final_out_data = vec![0.0; batch_size * embedder.pooler.units];

    let max_actual = actual_lengths.iter().copied().max().unwrap_or(0);
    for t in 0..max_actual {
        let mut step_input = vec![0.0; batch_size * embedder.pooler.in_features];
        for b in 0..batch_size {
            if t < actual_lengths[b] {
                let base_idx = (b * embedder.max_seq_length + t) * embedder.pooler.in_features;
                for i in 0..embedder.pooler.in_features {
                    step_input[b * embedder.pooler.in_features + i] = spikes2[base_idx + i];
                }
            }
        }

        let out_spikes = embedder.pooler.compute_step(&step_input, t);

        for b in 0..batch_size {
            if t < actual_lengths[b] {
                let offset = b * embedder.pooler.units;
                for i in 0..embedder.pooler.units {
                    final_out_data[offset + i] += out_spikes[offset + i];
                }
            }
        }
    }

    let mut normalized_out_data = vec![0.0; batch_size * embedder.pooler.units];
    for b in 0..batch_size {
        let offset = b * embedder.pooler.units;
        let mut sum = 0.0;
        for i in 0..embedder.pooler.units { sum += final_out_data[offset + i]; }
        let mean = sum / (embedder.pooler.units as f32);
        
        let mut sum_sq = 0.0;
        for i in 0..embedder.pooler.units {
            let centered = final_out_data[offset + i] - mean;
            sum_sq += centered * centered;
        }
        let norm = sum_sq.sqrt().max(1e-8);
        for i in 0..embedder.pooler.units { 
            normalized_out_data[offset + i] = (final_out_data[offset + i] - mean) / norm; 
        }
    }

    let mut error_final_data = vec![0.0; batch_size * embedder.pooler.units];
    let dummy_lengths = vec![1; batch_size];
    let pooler_loss = crate::core::contrastiveHebbian::contrastiveHebbian(
        &normalized_out_data, &mut error_final_data, num_pairs, 1, embedder.pooler.units, margin, &dummy_lengths
    );

    let mut emb_gradient_seq = vec![0.0; batch_seq * d_model];
    let mut att_gradient_seq = vec![0.0; batch_seq * d_model];

    for b in 0..batch_size {
        for s in 0..embedder.max_seq_length {
            if s < actual_lengths[b] {
                let offset_in = (b * embedder.max_seq_length + s) * d_model;
                let offset_out = b * d_model;
                for i in 0..d_model {
                    let err = error_final_data[offset_out + i];
                    if embedder.use_attention {
                        let spk1 = if spikes1[offset_in + i] > 0.5 { 1.0 } else { 0.0 };
                        let att_val = if att_spikes[offset_in + i] > 0.5 { 1.0 } else { 0.0 };
                        
                        if att_val > 0.5 {
                            emb_gradient_seq[offset_in + i] = -err;
                        } else {
                            emb_gradient_seq[offset_in + i] = err;
                        }
                        
                        if spk1 > 0.5 {
                            att_gradient_seq[offset_in + i] = -err;
                        } else {
                            att_gradient_seq[offset_in + i] = err;
                        }
                    } else {
                        emb_gradient_seq[offset_in + i] = err;
                    }
                }
            }
        }
    }

    embedder.embedding.backward(&emb_gradient_seq, None);
    if embedder.use_attention {
        embedder.attention.learn_attention(&att_gradient_seq, &actual_lengths);
    }

    (0.0, 0.0, pooler_loss)
}

pub fn train_step_distill_impl(embedder: &mut SpikingSentenceEmbedder, texts: &[&str], targets: &[f32], margin: f32) -> (f32, f32, f32) {
    embedder.zero_pad_token();
    let batch_size = texts.len();
    let num_pairs = batch_size / 2;
    
    embedder.embedding.reset_state();
    embedder.attention.reset_state(batch_size);
    embedder.pooler.reset_sequence(batch_size, embedder.max_seq_length);

    let mut tokenized_batch = Vec::with_capacity(batch_size * embedder.max_seq_length);
    let mut actual_lengths = vec![0; batch_size];
    for (b, text) in texts.iter().enumerate() {
        let mut tokens = embedder.tokenizer.encode(&text.to_lowercase());
        actual_lengths[b] = tokens.len().min(embedder.max_seq_length);
        if tokens.len() > embedder.max_seq_length { tokens.truncate(embedder.max_seq_length); }
        let mut tokens_f32: Vec<f32> = tokens.into_iter().map(|t| t as f32).collect();
        while tokens_f32.len() < embedder.max_seq_length { tokens_f32.push(0.0); }
        tokenized_batch.extend(tokens_f32);
    }

    embedder.cached_actual_lengths = Some(actual_lengths.clone());

    let batch_seq = batch_size * embedder.max_seq_length;
    let d_model = embedder.embedding.output_dim;

    let spikes1 = embedder.embedding.forward(&tokenized_batch);

    let mut spikes2 = vec![0.0; batch_seq * d_model];
    let att_spikes = if embedder.use_attention {
        embedder.attention.forward(&spikes1, &actual_lengths)
    } else {
        vec![0.0; batch_seq * d_model]
    };

    if embedder.use_attention {
        for i in 0..(batch_seq * d_model) {
            let spk1 = if spikes1[i] > 0.5 { 1.0 } else { 0.0 };
            let att_val = if att_spikes[i] > 0.5 { 1.0 } else { 0.0 };
            spikes2[i] = if spk1 != att_val { 1.0 } else { 0.0 };
        }
    } else {
        spikes2.copy_from_slice(&spikes1);
    }

    let mut final_out_data = vec![0.0; batch_size * embedder.pooler.units];
    let max_actual = actual_lengths.iter().copied().max().unwrap_or(0);
    for t in 0..max_actual {
        let mut step_input = vec![0.0; batch_size * embedder.pooler.in_features];
        for b in 0..batch_size {
            if t < actual_lengths[b] {
                let base_idx = (b * embedder.max_seq_length + t) * embedder.pooler.in_features;
                for i in 0..embedder.pooler.in_features {
                    step_input[b * embedder.pooler.in_features + i] = spikes2[base_idx + i];
                }
            }
        }
        let out_spikes = embedder.pooler.compute_step(&step_input, t);
        for b in 0..batch_size {
            if t < actual_lengths[b] {
                let offset = b * embedder.pooler.units;
                for i in 0..embedder.pooler.units { final_out_data[offset + i] += embedder.pooler.history_potentials[t][offset + i]; }
            }
        }
    }

    let mut normalized_out_data = vec![0.0; batch_size * embedder.pooler.units];
    for b in 0..batch_size {
        let offset = b * embedder.pooler.units;
        let mut sum = 0.0;
        for i in 0..embedder.pooler.units { sum += final_out_data[offset + i]; }
        let mean = sum / (embedder.pooler.units as f32);
        
        let mut sum_sq = 0.0;
        for i in 0..embedder.pooler.units {
            let centered = final_out_data[offset + i] - mean;
            sum_sq += centered * centered;
        }
        let norm = sum_sq.sqrt().max(1e-8);
        for i in 0..embedder.pooler.units { 
            normalized_out_data[offset + i] = (final_out_data[offset + i] - mean) / norm; 
        }
    }

    let mut error_final_data = vec![0.0; batch_size * embedder.pooler.units];
    let pooler_loss = crate::core::contrastiveHebbian::poolerDistillation(
        &normalized_out_data, &mut error_final_data, num_pairs, embedder.pooler.units, margin, targets
    );

    let mut emb_gradient_seq = vec![0.0; batch_seq * d_model];
    let mut att_gradient_seq = vec![0.0; batch_seq * d_model];

    for b in 0..batch_size {
        for s in 0..embedder.max_seq_length {
            if s < actual_lengths[b] {
                let offset_in = (b * embedder.max_seq_length + s) * d_model;
                let offset_out = b * d_model;
                for i in 0..d_model {
                    let err = error_final_data[offset_out + i];
                    if embedder.use_attention {
                        let spk1 = if spikes1[offset_in + i] > 0.5 { 1.0 } else { 0.0 };
                        let att_val = if att_spikes[offset_in + i] > 0.5 { 1.0 } else { 0.0 };
                        
                        if att_val > 0.5 {
                            emb_gradient_seq[offset_in + i] = -err;
                        } else {
                            emb_gradient_seq[offset_in + i] = err;
                        }
                        
                        if spk1 > 0.5 {
                            att_gradient_seq[offset_in + i] = -err;
                        } else {
                            att_gradient_seq[offset_in + i] = err;
                        }
                    } else {
                        emb_gradient_seq[offset_in + i] = err;
                    }
                }
            }
        }
    }

    embedder.embedding.backward(&emb_gradient_seq, None);
    if embedder.use_attention {
        embedder.attention.learn_attention(&att_gradient_seq, &actual_lengths);
    }

    (0.0, 0.0, pooler_loss)
}

use super::SpikingSentenceEmbedder;

pub fn encode_impl(embedder: &mut SpikingSentenceEmbedder, texts: &[&str]) -> Vec<Vec<f32>> {
    embedder.zero_pad_token();
    let batch_size = texts.len();
    embedder.embedding.reset_state();
    let mut tokenized_batch = Vec::with_capacity(batch_size * embedder.max_seq_length);

    let mut actual_lengths = vec![0; batch_size];
    for (b, text) in texts.iter().enumerate() {
        let mut tokens = embedder.tokenizer.encode(text);
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
    
    let emb_out = embedder.embedding.forward(&tokenized_batch);
    
    let mut emb_spikes = 0;
    for &val in &emb_out {
        if val > 0.0 { emb_spikes += 1; }
    }
    embedder.metrics.embedding_spikes += emb_spikes;
    embedder.metrics.total_sops += emb_spikes * embedder.attention.d_model * 3;

    let aggregated_features = if embedder.use_attention {
        let att_out = embedder.attention.forward(&emb_out, &actual_lengths);
        let mut att_spikes = 0;
        for &val in &att_out { if val > 0.0 { att_spikes += 1; } }
        embedder.metrics.attention_spikes += att_spikes;
        
        let mut agg = vec![0.0; batch_seq * d_model];
        for i in 0..(batch_seq * d_model) { 
            let spk1 = if emb_out[i] > 0.5 { 1.0 } else { 0.0 };
            let att_val = if att_out[i] > 0.5 { 1.0 } else { 0.0 };
            agg[i] = if spk1 != att_val { 1.0 } else { 0.0 };
        }
        agg
    } else {
        emb_out.clone()
    };

    embedder.pooler.reset_sequence(batch_size, embedder.max_seq_length);
    let mut final_embeddings = vec![vec![0.0; embedder.pooler.units]; batch_size];

    let max_actual = actual_lengths.iter().copied().max().unwrap_or(0);

    for t in 0..max_actual {
        let mut step_input = vec![0.0; batch_size * embedder.pooler.in_features];
        for b in 0..batch_size {
            if t < actual_lengths[b] {
                let base_idx = (b * embedder.max_seq_length + t) * embedder.pooler.in_features;
                for i in 0..embedder.pooler.in_features {
                    step_input[b * embedder.pooler.in_features + i] = aggregated_features[base_idx + i];
                }
            }
        }

        let out_spikes = embedder.pooler.compute_step(&step_input, t);

        for b in 0..batch_size {
            if t < actual_lengths[b] {
                let in_base = b * embedder.pooler.in_features;
                for i in 0..embedder.pooler.in_features {
                    if step_input[in_base + i] > 0.0 {
                        embedder.metrics.pooler_active_inputs += 1;
                        embedder.metrics.total_sops += embedder.pooler.units;
                    }
                }

                let offset = b * embedder.pooler.units;
                for i in 0..embedder.pooler.units {
                    if out_spikes[offset + i] > 0.0 {
                        embedder.metrics.pooler_spikes += 1;
                    }
                    final_embeddings[b][i] += embedder.pooler.history_potentials[t][offset + i];
                }
            }
        }
    }

    for b in 0..batch_size {
        let mut sum = 0.0;
        for val in &final_embeddings[b] {
            sum += val;
        }
        let mean = sum / (embedder.pooler.units as f32);
        
        let mut sum_sq = 0.0;
        for val in &final_embeddings[b] {
            let centered = val - mean;
            sum_sq += centered * centered;
        }
        let norm = sum_sq.sqrt().max(1e-8);
        for i in 0..embedder.pooler.units {
            final_embeddings[b][i] = (final_embeddings[b][i] - mean) / norm;
        }
    }

    embedder.metrics.total_sentences += batch_size;
    final_embeddings
}

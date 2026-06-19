use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::models::sentence_embedder::{SpikingSentenceEmbedder, SNNConfig};
use std::time::Instant;

use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;

#[derive(Deserialize)]
struct DataPair {
    s1: String,
    s2: String,
    score: f32,
}

fn load_dataset(path: &str, limit: usize) -> (Vec<String>, Vec<f32>) {
    let file = File::open(path).expect("Failed to open dataset");
    let reader = BufReader::new(file);
    let data: Vec<DataPair> = serde_json::from_reader(reader).expect("Failed to parse JSON");
    
    let mut texts = Vec::new();
    let mut targets = Vec::new();
    
    for pair in data.into_iter().take(limit) {
        texts.push(pair.s1);
        texts.push(pair.s2);
        targets.push(pair.score);
    }
    
    (texts, targets)
}

fn main() {
    let d_model = 128;
    let num_train = 5000;
    let num_test = 1000;
    let total_samples = num_train + num_test;
    
    println!("============================================================");
    println!(" REAL SNN PIPELINE - TRAIN {} / TEST {} (20 EPOCH)", num_train, num_test);
    println!("============================================================\n");

    let (texts, targets) = load_dataset("experiment/file_model/teacher_distillation_dataset_scored.json", total_samples);

    // Load Tokenizer
    let tokenizer = BPETokenizer::load("experiment/file_model/vocab.json");
    
    // Tentukan max_seq_length dengan mengecek dataset
    let max_seq_length = texts.iter().map(|text| tokenizer.encode(&text.to_lowercase()).len()).max().unwrap_or(16);
    println!("Max sequence length detected from dataset: {}", max_seq_length);

    let config = SNNConfig {
        d_model, max_seq_length, learning_rate: 1.5,
        clip_min: -1.0, clip_max: 1.0,
        att_beta_range: (0.8, 0.99), att_threshold_range: (-1.0, -0.5),
        bptt_beta_range: (0.8, 0.99), bptt_threshold_range: (0.5, 1.0),
    };

    let mut embedder = SpikingSentenceEmbedder::new(tokenizer, 50000, config.clone());
    
    let mut config_att = config.clone();
    config_att.learning_rate = 2.0; // Boost LR for Attention to help it converge faster
    let tokenizer2 = BPETokenizer::load("experiment/file_model/vocab.json");
    let mut embedder_att = SpikingSentenceEmbedder::new(tokenizer2, 50000, config_att);

    // FIX: Set Pooler to be a TRUE Identity Integrator so exact continuous gradients are mathematically correct
    for i in 0..d_model {
        for j in 0..d_model {
            embedder.pooler.kernel[i * d_model + j] = if i == j { 1.0 } else { 0.0 };
            embedder_att.pooler.kernel[i * d_model + j] = if i == j { 1.0 } else { 0.0 };
        }
        embedder.pooler.bias[i] = 0.0;
        embedder_att.pooler.bias[i] = 0.0;
    }

    // FIX: Copy exact random Embedding weights so both models start exactly equal!
    embedder_att.embedding.weights = embedder.embedding.weights.clone();
    embedder_att.embedding.potentials = embedder.embedding.potentials.clone();
    embedder_att.embedding.beta = embedder.embedding.beta.clone();
    embedder_att.embedding.threshold = embedder.embedding.threshold.clone();
    
    let margin = 1.0;

    println!("\n\n==============================================================");
    println!(" RUN 1: NO ATTENTION (Word2Vec Style SNN)");
    println!("==============================================================");
    for epoch in 1..=20 {
        let epoch_start = Instant::now();
        let use_attention = false;
        
        let print_log = epoch % 5 == 0 || epoch == 1 || epoch == 20;
        if print_log {
            println!("\n==============================================================");
            println!(" EPOCH {}", epoch);
            println!("==============================================================");
        }

        let mut train_loss = 0.0;
        let mut train_correct = 0;
        let mut test_loss = 0.0;
        let mut test_correct = 0;

        let mini_batch_pairs = 10;

        for mb_idx in (0..total_samples).step_by(mini_batch_pairs) {
            let start_pair = mb_idx;
            let end_pair = (mb_idx + mini_batch_pairs).min(total_samples);
            let num_pairs_mb = end_pair - start_pair;
            
            let is_train = start_pair < num_train;

            let start_text = start_pair * 2;
            let end_text = end_pair * 2;
            let texts_mb = &texts[start_text..end_text];
            let targets_mb = &targets[start_pair..end_pair];

            let batch_size = texts_mb.len();
            let batch_seq = batch_size * max_seq_length;
            
            embedder.embedding.reset_state();
            embedder.attention.reset_state(batch_size);
            embedder.pooler.reset_sequence(batch_size, max_seq_length);

            // 1. TOKENIZATION
            let mut tokenized_batch = Vec::with_capacity(batch_size * max_seq_length);
            let mut actual_lengths = vec![0; batch_size];
            for (b, text) in texts_mb.iter().enumerate() {
                let mut tokens = embedder.tokenizer.encode(&text.to_lowercase());
                actual_lengths[b] = tokens.len().min(max_seq_length);
                if tokens.len() > max_seq_length { tokens.truncate(max_seq_length); }
                let mut tokens_f32: Vec<f32> = tokens.into_iter().map(|t| t as f32).collect();
                while tokens_f32.len() < max_seq_length { tokens_f32.push(0.0); } // padding
                tokenized_batch.extend(&tokens_f32);
            }

            // 2. EMBEDDING LAYER
            let spikes1 = embedder.embedding.forward(&tokenized_batch);

            // 3. ATTENTION LAYER (Transfer Fitur / Spikes2)
            let mut spikes2 = vec![0.0; batch_seq * d_model];
            spikes2.copy_from_slice(&spikes1);

            // 5. POOLER BPTT
            let mut final_out_data = vec![0.0; batch_size * embedder.pooler.units];
            for t in 0..max_seq_length {
                let mut step_input = vec![0.0; batch_size * embedder.pooler.in_features];
                for b in 0..batch_size {
                    if t < actual_lengths[b] {
                        let base_idx = (b * max_seq_length + t) * embedder.pooler.in_features;
                        for i in 0..embedder.pooler.in_features {
                            step_input[b * embedder.pooler.in_features + i] = spikes2[base_idx + i];
                        }
                    }
                }
                let _out_spikes = embedder.pooler.compute_step(&step_input, t);
                let pot_at_t = &embedder.pooler.history_potentials[t];
                for b in 0..batch_size {
                    if t < actual_lengths[b] {
                        let offset = b * embedder.pooler.units;
                        for i in 0..embedder.pooler.units { 
                            final_out_data[offset + i] += pot_at_t[offset + i]; 
                        }
                    }
                }
            }

            // 6. MEAN-CENTERING & NORMALISASI & HASIL AKHIR
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

            for p in 0..num_pairs_mb {
                let b1 = p * 2;
                let b2 = p * 2 + 1;
                let off1 = b1 * embedder.pooler.units;
                let off2 = b2 * embedder.pooler.units;
                let sim = cosine_sim(&normalized_out_data[off1..off1+embedder.pooler.units], &normalized_out_data[off2..off2+embedder.pooler.units]);
                if (sim - targets_mb[p]).abs() < 0.05 {
                    if is_train { train_correct += 1; } else { test_correct += 1; }
                }
            }

            let mut error_final_data = vec![0.0; batch_size * embedder.pooler.units];
            let pooler_loss = SpikingNetworkRust::core::contrastiveHebbian::poolerDistillation(
                &normalized_out_data, &mut error_final_data, num_pairs_mb, embedder.pooler.units, margin, targets_mb
            );
            if is_train { train_loss += pooler_loss; } else { test_loss += pooler_loss; }

            // 7. BACKWARD PASS (LEARNING)
            let mut exact_gradient_seq = vec![0.0; batch_seq * d_model];
            for b in 0..batch_size {
                for s in 0..max_seq_length {
                    if s < actual_lengths[b] {
                        let offset_in = (b * max_seq_length + s) * d_model;
                        let offset_out = b * d_model;
                        for i in 0..d_model {
                            exact_gradient_seq[offset_in + i] = error_final_data[offset_out + i];
                        }
                    }
                }
            }
            
            if is_train {
                embedder.embedding.backward(&exact_gradient_seq, None);
            }
        }

        if print_log {
            let train_acc = (train_correct as f32 / num_train as f32) * 100.0;
            let test_acc = (test_correct as f32 / num_test as f32) * 100.0;
            println!("   [TRAIN] Loss: {:.4} | Acc: {:.2}% ({} dari {})", train_loss, train_acc, train_correct, num_train);
            println!("   [TEST]  Loss: {:.4} | Acc: {:.2}% ({} dari {})", test_loss, test_acc, test_correct, num_test);
            let epoch_duration = epoch_start.elapsed();
            println!("   [TIME] Epoch {} selesai dalam: {} ms", epoch, epoch_duration.as_millis());
        }
    }

    println!("\n\n==============================================================");
    println!(" RUN 2: WITH SPARSE COINCIDENCE ATTENTION");
    println!("==============================================================");
    for epoch in 1..=20 {
        let epoch_start = Instant::now();
        let use_attention = true;
        
        let print_log = epoch % 5 == 0 || epoch == 1 || epoch == 20;
        if print_log {
            println!("\n==============================================================");
            println!(" EPOCH {}", epoch);
            println!("==============================================================");
        }

        let mut train_loss = 0.0;
        let mut train_correct = 0;
        let mut test_loss = 0.0;
        let mut test_correct = 0;

        let mini_batch_pairs = 10;

        for mb_idx in (0..total_samples).step_by(mini_batch_pairs) {
            let start_pair = mb_idx;
            let end_pair = (mb_idx + mini_batch_pairs).min(total_samples);
            let num_pairs_mb = end_pair - start_pair;
            
            let is_train = start_pair < num_train;
            
            let start_text = start_pair * 2;
            let end_text = end_pair * 2;
            let texts_mb = &texts[start_text..end_text];
            let targets_mb = &targets[start_pair..end_pair];

            let batch_size = texts_mb.len();
            let batch_seq = batch_size * max_seq_length;
            
            embedder_att.embedding.reset_state();
            embedder_att.attention.reset_state(batch_size);
            embedder_att.pooler.reset_sequence(batch_size, max_seq_length);

            // 1. TOKENIZATION
            let mut tokenized_batch = Vec::with_capacity(batch_size * max_seq_length);
            let mut actual_lengths = vec![0; batch_size];
            for (b, text) in texts_mb.iter().enumerate() {
                let mut tokens = embedder_att.tokenizer.encode(&text.to_lowercase());
                actual_lengths[b] = tokens.len().min(max_seq_length);
                if tokens.len() > max_seq_length { tokens.truncate(max_seq_length); }
                let mut tokens_f32: Vec<f32> = tokens.into_iter().map(|t| t as f32).collect();
                while tokens_f32.len() < max_seq_length { tokens_f32.push(0.0); } // padding
                tokenized_batch.extend(&tokens_f32);
            }

            // 2. EMBEDDING LAYER
            let spikes1 = embedder_att.embedding.forward(&tokenized_batch);

            // 3. ATTENTION LAYER (XOR Residual Correction)
            let mut spikes2 = vec![0.0; batch_seq * d_model];
            let att_spikes = embedder_att.attention.forward(&spikes1, &actual_lengths);
            for i in 0..(batch_seq * d_model) {
                let att_val = if att_spikes[i] > 0.5 { 1.0 } else { 0.0 };
                // Logical XOR: jika berbeda maka 1, jika sama maka 0
                spikes2[i] = if spikes1[i] != att_val { 1.0 } else { 0.0 };
            }

            // 5. POOLER BPTT
            let mut final_out_data = vec![0.0; batch_size * embedder_att.pooler.units];
            for t in 0..max_seq_length {
                let mut step_input = vec![0.0; batch_size * embedder_att.pooler.in_features];
                for b in 0..batch_size {
                    if t < actual_lengths[b] {
                        let base_idx = (b * max_seq_length + t) * embedder_att.pooler.in_features;
                        for i in 0..embedder_att.pooler.in_features {
                            step_input[b * embedder_att.pooler.in_features + i] = spikes2[base_idx + i];
                        }
                    }
                }
                let _out_spikes = embedder_att.pooler.compute_step(&step_input, t);
                let pot_at_t = &embedder_att.pooler.history_potentials[t];
                for b in 0..batch_size {
                    if t < actual_lengths[b] {
                        let offset = b * embedder_att.pooler.units;
                        for i in 0..embedder_att.pooler.units { 
                            final_out_data[offset + i] += pot_at_t[offset + i]; 
                        }
                    }
                }
            }

            // 6. MEAN-CENTERING & NORMALISASI & HASIL AKHIR
            let mut normalized_out_data = vec![0.0; batch_size * embedder_att.pooler.units];
            for b in 0..batch_size {
                let offset = b * embedder_att.pooler.units;
                let mut sum = 0.0;
                for i in 0..embedder_att.pooler.units { sum += final_out_data[offset + i]; }
                let mean = sum / (embedder_att.pooler.units as f32);
                let mut sum_sq = 0.0;
                for i in 0..embedder_att.pooler.units { 
                    let centered = final_out_data[offset + i] - mean;
                    sum_sq += centered * centered; 
                }
                let norm = sum_sq.sqrt().max(1e-8);
                for i in 0..embedder_att.pooler.units { 
                    normalized_out_data[offset + i] = (final_out_data[offset + i] - mean) / norm; 
                }
            }

            for p in 0..num_pairs_mb {
                let b1 = p * 2;
                let b2 = p * 2 + 1;
                let off1 = b1 * embedder_att.pooler.units;
                let off2 = b2 * embedder_att.pooler.units;
                let sim = cosine_sim(&normalized_out_data[off1..off1+embedder_att.pooler.units], &normalized_out_data[off2..off2+embedder_att.pooler.units]);
                if (sim - targets_mb[p]).abs() < 0.05 {
                    if is_train { train_correct += 1; } else { test_correct += 1; }
                }
            }

            let mut error_final_data = vec![0.0; batch_size * embedder_att.pooler.units];
            let pooler_loss = SpikingNetworkRust::core::contrastiveHebbian::poolerDistillation(
                &normalized_out_data, &mut error_final_data, num_pairs_mb, embedder_att.pooler.units, margin, targets_mb
            );
            if is_train { train_loss += pooler_loss; } else { test_loss += pooler_loss; }

            // BPTT
            let mut exact_gradient_seq = vec![0.0; batch_seq * d_model];
            for b in 0..batch_size {
                for s in 0..max_seq_length {
                    if s < actual_lengths[b] {
                        let offset_in = (b * max_seq_length + s) * d_model;
                        let offset_out = b * d_model;
                        for i in 0..d_model {
                            exact_gradient_seq[offset_in + i] = error_final_data[offset_out + i];
                        }
                    }
                }
            }
            // BPTT untuk Embedding: 
            let mut emb_gradient_seq = vec![0.0; batch_seq * d_model];
            for i in 0..(batch_seq * d_model) {
                let att_val = if att_spikes[i] > 0.5 { 1.0 } else { 0.0 };
                if att_val > 0.5 {
                    emb_gradient_seq[i] = -exact_gradient_seq[i]; // Flip gradient for XOR
                } else {
                    emb_gradient_seq[i] = exact_gradient_seq[i];
                }
            }
            if is_train {
                embedder_att.embedding.backward(&emb_gradient_seq, None);
                
                let mut att_gradient_seq = vec![0.0; batch_seq * d_model];
                for i in 0..(batch_seq * d_model) {
                    if spikes1[i] > 0.5 {
                        att_gradient_seq[i] = -exact_gradient_seq[i];
                    } else {
                        att_gradient_seq[i] = exact_gradient_seq[i];
                    }
                }
                embedder_att.attention.learn_attention(&att_gradient_seq, &actual_lengths);
            }
        }

        if print_log {
            let train_acc = (train_correct as f32 / num_train as f32) * 100.0;
            let test_acc = (test_correct as f32 / num_test as f32) * 100.0;
            println!("   [TRAIN] Loss: {:.4} | Acc: {:.2}% ({} dari {})", train_loss, train_acc, train_correct, num_train);
            println!("   [TEST]  Loss: {:.4} | Acc: {:.2}% ({} dari {})", test_loss, test_acc, test_correct, num_test);
            let epoch_duration = epoch_start.elapsed();
            println!("   [TIME] Epoch {} selesai dalam: {} ms", epoch, epoch_duration.as_millis());
        }
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0_f32, 0.0_f32, 0.0_f32);
    for i in 0..a.len() {
        dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
    }
    if na == 0.0 || nb == 0.0 { return 0.0; }
    dot / (na.sqrt() * nb.sqrt())
}

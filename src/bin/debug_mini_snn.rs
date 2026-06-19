use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::models::sentence_embedder::{SpikingSentenceEmbedder, SNNConfig};
use std::time::Instant;

fn main() {
    let d_model = 16;
    let max_seq_length = 6;
    
    println!("============================================================");
    println!(" REAL SNN PIPELINE - DETAILED DEBUG (5 EPOCH)");
    println!("============================================================");

    // Load Tokenizer
    let tokenizer = BPETokenizer::load("experiment/file_model/vocab.json");
    let config = SNNConfig {
        d_model, max_seq_length, learning_rate: 0.05,
        clip_min: -1.0, clip_max: 1.0,
        att_beta_range: (0.8, 0.99), att_threshold_range: (0.1, 0.3),
        bptt_beta_range: (0.8, 0.99), bptt_threshold_range: (0.5, 1.0),
    };

    let mut embedder = SpikingSentenceEmbedder::new(tokenizer, 50000, config);
    let use_attention = false; // <-- DISABLE ATTENTION FOR POLYSEMY TEST


    // Mini Corpus dengan Kasus Polisemi (Kata sama, makna beda)
    let texts = [
        "saya makan nasi putih hangat", 
        "dia makan nasi putih enak", 
        "kucing tidur di kasur besar", 
        "anjing lari di jalan kecil",
        "bunga di bank sangat tinggi",
        "bunga di taman sangat cantik",
        "bunga di bank sangat tinggi",
        "suku bunga pinjaman sangat naik"
    ];
    let targets = [0.9, 0.1, 0.1, 0.9]; 
    let margin = 0.5;

    for epoch in 1..=500 {
        let epoch_start = Instant::now();
        
        let print_log = epoch % 100 == 0 || epoch == 1 || epoch == 500;
        if print_log {
            println!("\n==============================================================");
            println!(" EPOCH {}", epoch);
            println!("==============================================================");
        }

        let batch_size = texts.len();
        let num_pairs = batch_size / 2;
        let batch_seq = batch_size * max_seq_length;
        
        embedder.embedding.reset_state();
        embedder.attention.reset_state(batch_size);
        embedder.pooler.reset_sequence(batch_size, max_seq_length);

        // 1. TOKENIZATION
        let mut tokenized_batch = Vec::with_capacity(batch_size * max_seq_length);
        let mut actual_lengths = vec![0; batch_size];
        if print_log {
            println!(">> 1. TOKENIZATION & INPUT");
        }
        for (b, text) in texts.iter().enumerate() {
            let mut tokens = embedder.tokenizer.encode(&text.to_lowercase());
            actual_lengths[b] = tokens.len().min(max_seq_length);
            if tokens.len() > max_seq_length { tokens.truncate(max_seq_length); }
            let mut tokens_f32: Vec<f32> = tokens.into_iter().map(|t| t as f32).collect();
            while tokens_f32.len() < max_seq_length { tokens_f32.push(0.0); } // padding
            tokenized_batch.extend(&tokens_f32);
            if print_log {
                println!("   Batch {}: '{}' -> {:?}", b, text, tokens_f32);
            }
        }

        // 2. EMBEDDING LAYER
        let spikes1 = embedder.embedding.forward(&tokenized_batch);
        if print_log {
            println!("\n>> 2. EMBEDDING LAYER (Spikes1 Keluar)");
        }
        for b in 0..batch_size {
            let start = b * max_seq_length * d_model;
            let end = start + max_seq_length * d_model;
            let count = spikes1[start..end].iter().filter(|&&x| x > 0.0).count();
            if print_log {
                println!("   Batch {} menghasilkan {} letupan spike (Sparsitas: {:.1}%)", b, count, (count as f32 / (max_seq_length * d_model) as f32) * 100.0);
            }
        }

        // 3. ATTENTION LAYER (Transfer Fitur / Spikes2)
        let mut spikes2 = vec![0.0; batch_seq * d_model];
        if use_attention {
            let att_spikes = embedder.attention.forward(&spikes1, &actual_lengths);
            if print_log {
                println!("\n>> 3. ATTENTION LAYER (Transfer Fitur / Spikes2)");
            }
            for i in 0..(batch_seq * d_model) {
                let att_val = if att_spikes[i] > 0.5 { 1.0 } else { 0.0 };
                spikes2[i] = if spikes1[i] + att_val > 0.5 { 1.0 } else { 0.0 };
            }
            for b in 0..batch_size {
                let start = b * max_seq_length * d_model;
                let end = start + max_seq_length * d_model;
                let count = spikes2[start..end].iter().filter(|&&x| x > 0.0).count();
                if print_log {
                    println!("   Batch {} Spikes2 memiliki {} letupan spike setelah ditambah Attention", b, count);
                }
            }
        } else {
            if print_log {
                println!("\n>> 3. ATTENTION LAYER (DISABLED - Bypass ke Pooler)");
            }
            spikes2.copy_from_slice(&spikes1);
        }

        // 4. DISTILLATION ERROR DARI ATTENTION
        let mut err_att_data = vec![0.0; batch_seq * d_model];
        let loss2 = SpikingNetworkRust::core::contrastiveHebbian::distillationHebbian(
            &spikes2, &mut err_att_data, num_pairs, max_seq_length, d_model, margin, &actual_lengths, &targets
        );
        if print_log {
            println!("   Loss Distilasi dari Layer Attention: {:.4}", loss2);
            println!("   Sample Error Signal (Batch 0, T=0): {:?}", &err_att_data[0..d_model]);
        }

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
            let out_spikes = embedder.pooler.compute_step(&step_input, t);
            for b in 0..batch_size {
                if t < actual_lengths[b] {
                    let offset = b * embedder.pooler.units;
                    for i in 0..embedder.pooler.units { final_out_data[offset + i] += out_spikes[offset + i]; }
                }
            }
        }

        // 6. NORMALISASI & HASIL AKHIR
        let mut normalized_out_data = vec![0.0; batch_size * embedder.pooler.units];
        for b in 0..batch_size {
            let mut sum_sq = 0.0;
            let offset = b * embedder.pooler.units;
            for i in 0..embedder.pooler.units { sum_sq += final_out_data[offset + i] * final_out_data[offset + i]; }
            let norm = sum_sq.sqrt().max(1e-8);
            for i in 0..embedder.pooler.units { normalized_out_data[offset + i] = final_out_data[offset + i] / norm; }
        }

        if print_log {
            println!("\n>> 4. POOLER & COSINE SIMILARITY");
        }
        for p in 0..num_pairs {
            let b1 = p * 2;
            let b2 = p * 2 + 1;
            let off1 = b1 * embedder.pooler.units;
            let off2 = b2 * embedder.pooler.units;
            let sim = cosine_sim(&normalized_out_data[off1..off1+embedder.pooler.units], &normalized_out_data[off2..off2+embedder.pooler.units]);
            if print_log {
                println!("   Pair {} -> Sim: {:.4} (Target: {:.4})", p, sim, targets[p]);
            }
        }

        let mut error_final_data = vec![0.0; batch_size * embedder.pooler.units];
        let dummy_lengths = vec![1; batch_size];
        let pooler_loss = SpikingNetworkRust::core::contrastiveHebbian::poolerDistillation(
            &normalized_out_data, &mut error_final_data, num_pairs, embedder.pooler.units, margin, &targets
        );
        if print_log {
            println!("   Contrastive Loss Pooler: {:.4}", pooler_loss);
        }

        // 7. BACKWARD PASS (LEARNING)
        if print_log {
            println!("\n>> 5. LEARNING PASS (BACKWARD)");
        }
        if use_attention {
            embedder.attention.learn_attention(&err_att_data, &actual_lengths);
            if print_log {
                println!("   ✓ Attention Weights Updated (Hebbian + Oja's Decay)");
            }
        }
        
        let mut err_emb_data = vec![0.0; batch_seq * d_model];
        let _loss1 = SpikingNetworkRust::core::contrastiveHebbian::distillationHebbian(
            &spikes1, &mut err_emb_data, num_pairs, max_seq_length, d_model, margin, &actual_lengths, &targets
        );
        embedder.embedding.backward(&err_emb_data, None);
        if print_log {
            println!("   ✓ Embedding Weights Updated");
        }
        
        let mut error_seq = vec![vec![0.0; batch_size * embedder.pooler.units]; max_seq_length];
        for s in 0..max_seq_length {
            for b in 0..batch_size {
                if s < actual_lengths[b] {
                    let offset = b * embedder.pooler.units;
                    for i in 0..embedder.pooler.units {
                        error_seq[s][offset + i] = error_final_data[offset + i];
                    }
                }
            }
        }
        use SpikingNetworkRust::layers::base::Layer;
        let lr = embedder.pooler.get_base_config().learning_rate;
        let _error_wrt_inputs = embedder.pooler.learn_through_time(&error_seq, lr);
        if print_log {
            println!("   ✓ Pooler Weights Updated");

            let mut q_sum = 0.0;
            for i in 0..(d_model * d_model) { q_sum += embedder.attention.kernel_q[i]; }
            println!("   [DEBUG] Total Sum of Q Kernels: {:.4}", q_sum);
        }

        // UKUR WAKTU SATU EPOCH
        let epoch_duration = epoch_start.elapsed();
        if print_log {
            println!("\n   [TIME] Epoch {} selesai dalam: {} ms ({} mikrodetik)", epoch, epoch_duration.as_millis(), epoch_duration.as_micros());
        }
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0_f32, 0.0_f32, 0.0_f32);
    for i in 0..a.len() {
        dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
    }
    if na == 0.0 || nb == 0.0 { return 0.0; }
    (dot / (na.sqrt() * nb.sqrt())).max(0.0)
}

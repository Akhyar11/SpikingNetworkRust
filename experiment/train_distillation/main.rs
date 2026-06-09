#[path = "../model/sentence_embedder.rs"]
pub mod sentence_embedder;

use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;

use sentence_embedder::SpikingSentenceEmbedder;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Write};
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
struct DistillationPair {
    s1: String,
    s2: String,
    score: f32,
}

fn main() {
    let vocab_path = "experiment/file_model/vocab.json";
    let dataset_path = "/home/akhyar/Dokumen/Code/NODE_JS/penelitian_model_bahasa_dengan_spiking/dataset/teacher_distillation_dataset.json";
    let model_save_path = "experiment/file_model/saved_model.json";

    println!("Memuat tokenizer dari {}...", vocab_path);
    let tokenizer = BPETokenizer::load(vocab_path);
    let vocab_size = tokenizer.vocab_size();

    // Hyperparameters
    let d_model = 128;
    let max_seq_length = 32;
    let num_pairs = 32; 
    let num_epochs = 1;

    let snn_config = sentence_embedder::SNNConfig {
        d_model,
        max_seq_length,
        learning_rate: 0.01,
        clip_min: -1.0,
        clip_max: 1.0,
        att_beta_range: (0.8, 0.99),
        att_threshold_range: (0.1, 0.3),
        bptt_beta_range: (0.8, 0.99),
        bptt_threshold_range: (0.5, 1.0),
    };

    println!("Inisialisasi SpikingSentenceEmbedder (Vocab: {}, D_Model: {})...", vocab_size, d_model);
    let mut embedder = SpikingSentenceEmbedder::new(
        tokenizer,
        vocab_size,
        snn_config,
    );
    embedder.summary();

    println!("Memuat dataset distilasi dari {}...", dataset_path);
    let file = File::open(dataset_path).expect("Gagal membuka dataset. Pastikan Anda sudah menjalankan generate_teacher_dataset.js");
    let reader = BufReader::new(file);
    let mut dataset: Vec<DistillationPair> = serde_json::from_reader(reader).expect("Format JSON tidak valid");
    
    let valid_pairs_count = dataset.len();
    let max_steps_per_epoch = if valid_pairs_count > 0 { valid_pairs_count / num_pairs } else { 1 };
    println!("Total pasangan kalimat: {}, Estimasi {} step per epoch", valid_pairs_count, max_steps_per_epoch);

    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();

    let mut best_loss = f32::MAX;
    let total_global_steps = max_steps_per_epoch * num_epochs;
    let mut global_step = 0;

    for epoch in 1..=num_epochs {
        dataset.shuffle(&mut rng);
        let mut step = 0;
        let start_time = Instant::now();
        let mut last_log_time = Instant::now();
        let mut epoch_loss_emb = 0.0;
        let mut epoch_loss_att = 0.0;

        let mut batch_texts = Vec::new();
        let mut batch_targets = Vec::new();

        for pair in &dataset {
            batch_texts.push(pair.s1.clone());
            batch_texts.push(pair.s2.clone());
            batch_targets.push(pair.score);

            if batch_targets.len() == num_pairs {
                let current_lr = 0.01 * f32::max(0.01, 1.0 - (global_step as f32 / total_global_steps as f32));
                embedder.set_learning_rate(current_lr);

                let texts_str: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();

                let (loss1, loss2, _) = embedder.train_step_distill(&texts_str, &batch_targets, 0.2);
                
                epoch_loss_emb += loss1;
                epoch_loss_att += loss2;
                step += 1;
                global_step += 1;

                batch_texts.clear();
                batch_targets.clear();

                if last_log_time.elapsed().as_millis() > 200 {
                    print!("\r[Epoch {}/{}] Progress: {:.1}% ({}/{}) | Loss Emb: {:.4} | Loss Att: {:.4}", 
                        epoch, num_epochs, 
                        (step as f32 / max_steps_per_epoch as f32) * 100.0,
                        step, max_steps_per_epoch,
                        loss1, loss2
                    );
                    std::io::stdout().flush().unwrap();
                    last_log_time = Instant::now();
                }
            }
        }

        let avg_loss = (epoch_loss_emb + epoch_loss_att) / (2.0 * step as f32);
        let elapsed = start_time.elapsed();
        println!("\n[HASIL] Epoch {}/{} | Rata-rata Hebbian Loss: {:.4} | Waktu Total: {:.2} s", 
            epoch, num_epochs, avg_loss, elapsed.as_secs_f32()
        );

        if avg_loss < best_loss {
            println!(">> Hebbian Loss membaik dari {:.4} ke {:.4}. Menyimpan model...", best_loss, avg_loss);
            best_loss = avg_loss;

            let mut final_model_data = serde_json::Map::new();
            final_model_data.insert("d_model".to_string(), serde_json::json!(d_model));
            final_model_data.insert("max_seq_length".to_string(), serde_json::json!(max_seq_length));
            
            let mut emb_weights = serde_json::Map::new();
            for (k, v) in embedder.embedding.get_parameters() {
                emb_weights.insert(k.to_string(), serde_json::json!(v));
            }
            final_model_data.insert("embedding".to_string(), serde_json::Value::Object(emb_weights));

            let mut att_weights = serde_json::Map::new();
            for (k, v) in embedder.attention.get_parameters() {
                att_weights.insert(k.to_string(), serde_json::json!(v));
            }
            final_model_data.insert("attention".to_string(), serde_json::Value::Object(att_weights));

            let mut pooler_weights = serde_json::Map::new();
            for (k, v) in embedder.pooler.get_parameters() {
                pooler_weights.insert(k.to_string(), serde_json::json!(v));
            }
            final_model_data.insert("pooler".to_string(), serde_json::Value::Object(pooler_weights));

            let json_str = serde_json::to_string_pretty(&final_model_data).unwrap();
            let mut file = File::create(model_save_path).unwrap();
            file.write_all(json_str.as_bytes()).unwrap();
        }
    }
    println!("\nTraining Distilasi selesai! Model tersimpan di {}", model_save_path);
}

#[path = "../model/sentence_embedder_simcse.rs"]
pub mod sentence_embedder;

use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;

use sentence_embedder::SpikingSentenceEmbedder;
use rand::Rng;
use serde_json::json;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

// Fungsi Trik Unsupervised SimCSE: P adalah teks yang identik dengan Q!
// Perbedaan/variasi diberikan dari Dropout di dalam network, bukan di teks.
fn corrupt_sentence(sentence: &str) -> String {
    sentence.to_string()
}

fn main() {
    let vocab_path = "experiment/file_model/vocab.json";
    let corpus_path = "/home/akhyar/Dokumen/Code/NODE_JS/penelitian_model_bahasa_dengan_spiking/dataset/mini_corpus.txt";
    let model_save_path = "experiment/file_model/saved_model.json";

    println!("Memuat tokenizer dari {}...", vocab_path);
    let tokenizer = BPETokenizer::load(vocab_path);

    let vocab_size = tokenizer.vocab_size();
    
    // Hyperparameters (Metadata Pelatihan)
    let d_model = 256;
    let max_seq_length = 128; // Diubah sesuai permintaan
    let num_pairs = 32;       // Mengikuti referensi train_wiki_unsupervised.ts
    let _batch_size = num_pairs * 2; // Total 64 kalimat per batch
    let num_epochs = 10;
    let min_words = 10;

    // SNN Hyperparameters diletakkan di sini sesuai permintaan
    let snn_config = sentence_embedder::SNNConfig {
        d_model,
        max_seq_length,
        learning_rate: 0.01,
        clip_min: -1.0,
        clip_max: 1.0,
        att_beta_range: (0.8, 0.99),
        att_threshold_range: (0.1, 0.3),
        bptt_beta_range: (0.5, 0.99),
        bptt_threshold_range: (0.5, 1.0),
    };
    
    let margin = 0.2; // Margin untuk Contrastive Hebbian Loss

    println!("Inisialisasi SpikingSentenceEmbedder (Vocab: {}, D_Model: {})...", vocab_size, d_model);
    let mut embedder = SpikingSentenceEmbedder::new(tokenizer, vocab_size, snn_config);
    embedder.summary();
    
    println!("Menganalisa corpus untuk menghitung total step...");
    
    let mut all_lines = Vec::new();
    if let Ok(file) = File::open(corpus_path) {
        let reader = BufReader::new(file);
        for line in reader.lines() {
            if let Ok(l) = line {
                let trim = l.trim();
                if !trim.is_empty() && trim.split_whitespace().count() >= min_words {
                    all_lines.push(trim.to_string());
                }
            }
        }
    } else {
        panic!("Gagal membuka corpus di path: {}", corpus_path);
    }
    
    let valid_lines_count = all_lines.len();
    let max_steps_per_epoch = if valid_lines_count > 0 { valid_lines_count / num_pairs } else { 1 };
    println!("Total kalimat valid: {}, Estimasi {} step per epoch", valid_lines_count, max_steps_per_epoch);

    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();

    let mut best_loss = f32::MAX;
    let mut patience_counter = 0;
    let patience_limit = 2; // Berhenti jika loss tidak membaik selama 2 epoch berturut-turut

    let mut global_step = 0;
    let total_global_steps = max_steps_per_epoch * num_epochs;
    let dropout_rate = 0.1; // 10% neuron spikes will be dropped randomly

    for epoch in 1..=num_epochs {
        let mut step = 0;
        let start_time = Instant::now();
        let mut last_log_time = Instant::now();
        
        let mut epoch_loss_l1 = 0.0;
        let mut epoch_loss_l2 = 0.0;
        let mut epoch_loss_pooler = 0.0;
        
        all_lines.shuffle(&mut rng);
        
        let mut q_texts = Vec::new();
        let mut p_texts = Vec::new();
        
        for q_line in &all_lines {
            // P adalah Q yang persis sama, perbedaan akan didapat dari Dropout!
            let p_line = corrupt_sentence(q_line); 
            
            q_texts.push(q_line.clone());
            p_texts.push(p_line);

            if q_texts.len() == num_pairs {
                let mut batch_texts = Vec::new();
                for q in &q_texts { batch_texts.push(q.as_str()); }
                for p in &p_texts { batch_texts.push(p.as_str()); }

                // Hitung decay berdasarkan GLOBAL STEP, bukan step per epoch!
                let current_lr = 0.01 * f32::max(0.01, 1.0 - (global_step as f32 / total_global_steps as f32));
                embedder.set_learning_rate(current_lr);

                let (loss1, loss2, loss3) = embedder.train_step(&batch_texts, num_pairs, margin, dropout_rate);
                
                epoch_loss_l1 += loss1;
                epoch_loss_l2 += loss2;
                epoch_loss_pooler += loss3;

                q_texts.clear();
                p_texts.clear();
                step += 1;
                global_step += 1;

                if step % 10 == 0 {
                    let elapsed_interval = last_log_time.elapsed();
                    let ms_per_batch = elapsed_interval.as_millis() as f64 / 10.0;

                    let avg_l1 = epoch_loss_l1 / step as f32;
                    let avg_l2 = epoch_loss_l2 / step as f32;
                    let avg_l3 = epoch_loss_pooler / step as f32;

                    print!(
                        "\r[Epoch {}/{}] Progress: {:.1}% ({}/{}) | L1: {:.2} | L2: {:.2} | BPTT/L3: {:.2} | {:.1} ms/step ",
                        epoch,
                        num_epochs,
                        (step as f64 / max_steps_per_epoch as f64) * 100.0,
                        step,
                        max_steps_per_epoch,
                        avg_l1,
                        avg_l2,
                        avg_l3,
                        ms_per_batch
                    );
                    std::io::stdout().flush().unwrap();
                    
                    last_log_time = Instant::now();
                }
            }
        }
        
        let total_epoch_time = start_time.elapsed().as_secs_f64();
        let final_avg_l1 = epoch_loss_l1 / step as f32;
        let final_avg_l2 = epoch_loss_l2 / step as f32;
        let final_avg_l3 = epoch_loss_pooler / step as f32;
        let current_total_loss = final_avg_l1 + final_avg_l2 + final_avg_l3;
        
        println!("\n\n[HASIL] Epoch {}/{} | Rata-rata L1 Loss: {:.4} | Rata-rata L2 Loss: {:.4} | Rata-rata BPTT Loss: {:.4} | Total Loss: {:.4} | Waktu Total: {:.2} s", 
                 epoch, num_epochs, final_avg_l1, final_avg_l2, final_avg_l3, current_total_loss, total_epoch_time);

        // Early Stopping & Model Checkpointing
        if current_total_loss < best_loss {
            println!(">> Loss membaik dari {:.4} ke {:.4}. Menyimpan model terbaik sementara...", best_loss, current_total_loss);
            best_loss = current_total_loss;
            patience_counter = 0;
            save_model(&embedder, model_save_path); // Simpan model di tiap titik terbaik
        } else {
            patience_counter += 1;
            println!(">> Loss tidak membaik (Patience: {}/{})", patience_counter, patience_limit);
            if patience_counter >= patience_limit {
                println!("!! Early Stopping terpicu pada epoch {}. Proses training dihentikan.", epoch);
                break;
            }
        }
    }

    println!("\nTraining eksperimen selesai! Model terbaik telah tersimpan di {}", model_save_path);
}

fn save_model(embedder: &SpikingSentenceEmbedder, path: &str) {
    let mut model_data = serde_json::Map::new();
    
    // 0. Simpan Hyperparameter
    model_data.insert("d_model".to_string(), json!(embedder.attention.d_model));
    model_data.insert("max_seq_length".to_string(), json!(embedder.max_seq_length));

    // 1. Simpan parameter Embedding
    let mut emb_params = serde_json::Map::new();
    for (name, data) in embedder.embedding.get_parameters() {
        emb_params.insert(name.to_string(), json!(data));
    }
    model_data.insert("embedding".to_string(), json!(emb_params));

    // 2. Simpan parameter Attention
    let mut att_params = serde_json::Map::new();
    for (name, data) in embedder.attention.get_parameters() {
        att_params.insert(name.to_string(), json!(data));
    }
    model_data.insert("attention".to_string(), json!(att_params));

    // 3. Simpan parameter Temporal Pooler (BPTT)
    let mut pool_params = serde_json::Map::new();
    for (name, data) in embedder.pooler.get_parameters() {
        pool_params.insert(name.to_string(), json!(data));
    }
    model_data.insert("pooler".to_string(), json!(pool_params));

    let json_string = serde_json::to_string_pretty(&model_data).unwrap();
    let mut file = File::create(path).expect("Gagal membuat file model JSON");
    file.write_all(json_string.as_bytes()).expect("Gagal menulis file JSON");

    println!("Model berhasil diekspor di {}", path);
}

#[path = "../model/sentence_embedder.rs"]
pub mod sentence_embedder;

use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use SpikingNetworkRust::core::contrastiveHebbian::contrastiveHebbian;
use sentence_embedder::SpikingSentenceEmbedder;
use rand::Rng;
use serde_json::json;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

// Fungsi Trik Unsupervised: Membuat Positive Sample (P+) dengan Noise/Masking (Aggressive)
fn corrupt_sentence(sentence: &str) -> String {
    let mut words: Vec<String> = sentence.split_whitespace().map(|s| s.to_string()).collect();
    let mut rng = rand::thread_rng();

    if words.len() > 3 {
        let drop_count = 2.max((words.len() as f32 * 0.25) as usize);
        for _ in 0..drop_count {
            if words.len() <= 2 { break; }

            let drop_type: f32 = rng.gen_range(0.0..1.0);
            let target_idx = rng.gen_range(0..words.len());

            if drop_type < 0.4 {
                // 40% Peluang: Hapus 1 kata penuh
                words.remove(target_idx);
            } else if drop_type < 0.8 {
                // 40% Peluang: Hapus 1 huruf di dalam kata (Typo)
                let target_word = &words[target_idx];
                if target_word.len() > 3 {
                    let char_idx = rng.gen_range(0..target_word.len());
                    let mut new_word = String::new();
                    for (i, c) in target_word.chars().enumerate() {
                        if i != char_idx {
                            new_word.push(c);
                        }
                    }
                    words[target_idx] = new_word;
                } else {
                    words.remove(target_idx);
                }
            } else {
                // 20% Peluang: Duplikasi/Pengulangan kata (Gagap)
                let dup = words[target_idx].clone();
                words.insert(target_idx, dup);
            }
        }
    }
    words.join(" ")
}

fn main() {
    let vocab_path = "experiment/file_model/vocab.json";
    let corpus_path = "experiment/file_model/mini_corpus20mb.txt";
    let model_save_path = "experiment/file_model/saved_model.json";

    println!("Memuat tokenizer dari {}...", vocab_path);
    let tokenizer = BPETokenizer::load(vocab_path);

    let vocab_size = tokenizer.vocab_size();
    
    // Hyperparameters (Metadata Pelatihan)
    let d_model = 64;
    let max_seq_length = 32; // Diubah sesuai permintaan
    let num_pairs = 32;       // Mengikuti referensi train_wiki_unsupervised.ts
    let batch_size = num_pairs * 2; // Total 64 kalimat per batch
    let num_epochs = 1;

    // SNN Hyperparameters diletakkan di sini sesuai permintaan
    let snn_config = sentence_embedder::SNNConfig {
        d_model,
        max_seq_length,
        learning_rate: 0.01,
        clip_min: -1.0,
        clip_max: 1.0,
        att_beta_range: (0.8, 0.9),
        att_threshold_range: (0.1, 0.3),
        bptt_beta_range: (0.8, 0.9),
        bptt_threshold_range: (0.1, 0.5),
    };
    
    let margin = 0.2; // Margin untuk Contrastive Hebbian Loss

    println!("Inisialisasi SpikingSentenceEmbedder (Vocab: {}, D_Model: {})...", vocab_size, d_model);
    let mut embedder = SpikingSentenceEmbedder::new(tokenizer, vocab_size, snn_config);
    embedder.summary();
    
    println!("Menganalisa corpus untuk menghitung total step...");
    let mut valid_lines_count = 0;
    if let Ok(file) = File::open(corpus_path) {
        let reader = BufReader::new(file);
        for line in reader.lines() {
            if let Ok(l) = line {
                let q_line = l.trim();
                if !q_line.is_empty() && q_line.len() >= 31 {
                    valid_lines_count += 1;
                }
            }
        }
    }
    let max_steps_per_epoch = if valid_lines_count > 0 { valid_lines_count / num_pairs } else { 1 };
    println!("Total kalimat valid: {}, Estimasi {} step per epoch", valid_lines_count, max_steps_per_epoch);

    for epoch in 1..=num_epochs {
        let mut step = 0; // Reset step counter per epoch
        let mut start_time = Instant::now();
        let mut last_log_time = Instant::now();
        let file = File::open(corpus_path).expect("Gagal membuka corpus.");
        let reader = BufReader::new(file);
        let mut lines_iter = reader.lines();
        
        let mut q_texts = Vec::new();
        let mut p_texts = Vec::new();
        
        while let Some(Ok(line)) = lines_iter.next() {
        let q_line = line.trim();
        // Lewati kalimat kosong atau terlalu pendek seperti pada referensi
        if q_line.is_empty() || q_line.len() < 31 { continue; }
        
        let p_line = corrupt_sentence(q_line);
        
        q_texts.push(q_line.to_string());
        p_texts.push(p_line);

        if q_texts.len() == num_pairs {
            let mut batch_texts = Vec::new();
            for q in &q_texts { batch_texts.push(q.as_str()); }
            for p in &p_texts { batch_texts.push(p.as_str()); }

            let loss = embedder.train_step(&batch_texts, num_pairs, margin);

            q_texts.clear();
            p_texts.clear();
            step += 1;

            if step % 10 == 0 {
                let elapsed_total = start_time.elapsed();
                let elapsed_interval = last_log_time.elapsed();
                let ms_per_batch = elapsed_interval.as_millis() as f64 / 10.0;
                let pct = (step as f64 / max_steps_per_epoch as f64).min(1.0);
                let bar_len: usize = 20;
                let filled = (pct * bar_len as f64) as usize;
                let empty = bar_len.saturating_sub(filled);
                let bar = format!("{}{}{}", "=".repeat(filled), ">", " ".repeat(empty));

                print!(
                    "\r[{:02}:{:02}:{:02}] Epoch: {}/{} | [{}] Step: {:5}/{} | Loss: {:>8.4} | {:>6.2} ms/batch  ",
                    elapsed_total.as_secs() / 3600,
                    (elapsed_total.as_secs() % 3600) / 60,
                    elapsed_total.as_secs() % 60,
                    epoch,
                    num_epochs,
                    bar,
                    step,
                    max_steps_per_epoch,
                    loss,
                    ms_per_batch
                );
                std::io::stdout().flush().unwrap();
                
                last_log_time = Instant::now();
            }
        }
        } // Tutup while let loop
        println!("\nEpoch {} selesai!", epoch);
    }

    println!("Training eksperimen selesai! Menyimpan model...");
    save_model(&embedder, model_save_path);
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

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
    let max_seq_length = 128; // Diubah sesuai permintaan
    let num_pairs = 32;       // Mengikuti referensi train_wiki_unsupervised.ts
    let batch_size = num_pairs * 2; // Total 64 kalimat per batch

    println!("Inisialisasi SpikingSentenceEmbedder (Vocab: {}, D_Model: {})...", vocab_size, d_model);
    let mut embedder = SpikingSentenceEmbedder::new(tokenizer, vocab_size, d_model, max_seq_length);
    embedder.summary();

    println!("Membaca corpus dari {}...", corpus_path);
    let file = File::open(corpus_path).expect("Gagal membuka corpus. Pastikan file mini_corpus20mb.txt ada.");
    let reader = BufReader::new(file);

    let mut lines_iter = reader.lines();
    
    let mut step = 0;
    let start_time = Instant::now();
    let mut last_log_time = Instant::now();

    let mut q_texts = Vec::new();
    let mut p_texts = Vec::new();
    
    while let Some(Ok(line)) = lines_iter.next() {
        let q_line = line.trim();
        // Lewati kalimat kosong atau terlalu pendek seperti pada referensi
        if q_line.is_empty() || q_line.len() < 50 { continue; }
        
        let p_line = corrupt_sentence(q_line);
        
        q_texts.push(q_line.to_string());
        p_texts.push(p_line);

        if q_texts.len() == num_pairs {
            let mut batch_texts = Vec::new();
            for q in &q_texts { batch_texts.push(q.as_str()); }
            for p in &p_texts { batch_texts.push(p.as_str()); }

            // Encode (Forward pass)
            let embeddings = embedder.encode(&batch_texts);

            // Ratakan output embeddings untuk fungsi native contrastiveHebbian
            let mut flat_embeddings = Vec::with_capacity(batch_size * d_model);
            for emb in &embeddings {
                flat_embeddings.extend_from_slice(emb);
            }

            // Siapkan array error/gradient (Delta)
            let mut err_data = vec![0.0; batch_size * d_model];
            
            // Hitung Contrastive Hebbian Loss dan gradient
            let loss = contrastiveHebbian(&flat_embeddings, &mut err_data, num_pairs, 1, d_model);

            // Susun kembali sinyal error ke bentuk matriks 2D untuk backward pass SNN
            let mut error_signals = vec![vec![0.0; d_model]; batch_size];
            for b in 0..batch_size {
                for i in 0..d_model {
                    error_signals[b][i] = err_data[b * d_model + i];
                }
            }

            // Propagasi mundur
            embedder.learn(&error_signals);

            q_texts.clear();
            p_texts.clear();
            step += 1;

            if step % 10 == 0 {
                let elapsed_total = start_time.elapsed();
                let elapsed_interval = last_log_time.elapsed();
                let ms_per_batch = elapsed_interval.as_millis() as f64 / 10.0;
                
                let max_steps = 1000;
                let pct = step as f64 / max_steps as f64;
                let bar_len = 20;
                let filled = (pct * bar_len as f64) as usize;
                let empty = bar_len - filled;
                let bar = format!("{}{}{}", "=".repeat(filled), ">", " ".repeat(empty));

                print!(
                    "\r[{:02}:{:02}:{:02}] [{}] Step: {:4}/{} | Loss: {:>8.4} | {:>6.2} ms/batch  ",
                    elapsed_total.as_secs() / 3600,
                    (elapsed_total.as_secs() % 3600) / 60,
                    elapsed_total.as_secs() % 60,
                    bar,
                    step,
                    max_steps,
                    loss,
                    ms_per_batch
                );
                std::io::stdout().flush().unwrap();
                
                last_log_time = Instant::now();
            }

            // Lakukan iterasi untuk pemanasan jaringan
            if step >= 1000 {
                println!(); // Pindah ke baris baru setelah selesai
                break;
            }
        }
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

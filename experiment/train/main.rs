#[path = "../model/sentence_embedder.rs"]
pub mod sentence_embedder;

use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use SpikingNetworkRust::core::contrastiveHebbian::contrastiveHebbian;
use sentence_embedder::SpikingSentenceEmbedder;
use serde_json::json;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

fn main() {
    let vocab_path = "experiment/file_model/vocab.json";
    let corpus_path = "experiment/file_model/mini_corpus20mb.txt";
    let model_save_path = "experiment/file_model/saved_model.json";

    println!("Memuat tokenizer dari {}...", vocab_path);
    let tokenizer = BPETokenizer::load(vocab_path);

    let vocab_size = tokenizer.vocab_size();
    let d_model = 64; // Bisa disesuaikan
    let max_seq_length = 32;
    let batch_size = 8;

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

    let num_pairs = batch_size / 2;
    let mut q_texts = Vec::new();
    let mut p_texts = Vec::new();
    
    while let Some(Ok(line)) = lines_iter.next() {
        if line.trim().is_empty() { continue; }
        
        let mut next_line = String::new();
        let mut found_pair = false;
        while let Some(Ok(l2)) = lines_iter.next() {
            if !l2.trim().is_empty() {
                next_line = l2;
                found_pair = true;
                break;
            }
        }
        
        if found_pair {
            q_texts.push(line);
            p_texts.push(next_line);
        }

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
                
                println!(
                    "[{:02}:{:02}:{:02}] Step: {:4} | Loss: {:>8.4} | Waktu: {:>6.2} ms/batch",
                    elapsed_total.as_secs() / 3600,
                    (elapsed_total.as_secs() % 3600) / 60,
                    elapsed_total.as_secs() % 60,
                    step,
                    loss,
                    ms_per_batch
                );
                
                last_log_time = Instant::now();
            }

            // Lakukan 1000 iterasi untuk pemanasan jaringan
            if step >= 1000 {
                break;
            }
        }
    }

    println!("Training eksperimen selesai! Menyimpan model...");
    save_model(&embedder, model_save_path);
}

fn save_model(embedder: &SpikingSentenceEmbedder, path: &str) {
    let mut model_data = serde_json::Map::new();
    
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

//! Revision experiments: train (attention-free Delta pooler recipe) and export embeddings.
//! Usage: train_export --d 256 --seed 1 --epochs 10 --data data_eval/train_decontam.json \
//!        --obj distill --out results/run_x [--eval-only-sentences data_eval/all_eval_sentences.json] [--no-train]
//! Seed is applied through env SNN_SEED (embedding/pooler init) and the shuffle RNG.
use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::models::sentence_embedder::{SpikingSentenceEmbedder, SNNConfig};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, Write};
use std::time::Instant;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Deserialize, Clone)]
struct Pair { s1: String, s2: String, score: f32 }

fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1).cloned()).unwrap_or(default.to_string())
}

fn main() {
    let d: usize = arg("--d", "256").parse().unwrap();
    let seed: u64 = arg("--seed", "1").parse().unwrap();
    let epochs: usize = arg("--epochs", "10").parse().unwrap();
    let data_path = arg("--data", "data_eval/train_decontam.json");
    let out = arg("--out", "results/run");
    let sent_path = arg("--sentences", "data_eval/all_eval_sentences.json");
    let base_lr: f32 = arg("--lr", "2.0").parse().unwrap();
    let max_len = 128usize;
    unsafe { std::env::set_var("SNN_SEED", seed.to_string()); }

    let tok = BPETokenizer::load("experiment/file_model/vocab_multilingual.json");
    let vocab = tok.vocab_size();
    let mut emb = SpikingSentenceEmbedder::new(tok, vocab, SNNConfig {
        d_model: d, max_seq_length: max_len, learning_rate: 0.01, clip_min: -1.0, clip_max: 1.0,
        att_beta_range: (0.8, 0.99), att_threshold_range: (-1.0, -0.5),
        bptt_beta_range: (0.8, 0.99), bptt_threshold_range: (0.5, 1.0),
    });
    for i in 0..d { for j in 0..d { emb.pooler.kernel[i * d + j] = if i == j { 1.0 } else { 0.0 }; } }
    emb.set_use_attention(false);

    let mut train_secs = 0.0;
    let mut steps = 0usize;
    if epochs > 0 {
        let data: Vec<Pair> = serde_json::from_reader(BufReader::new(File::open(&data_path).unwrap())).unwrap();
        let np = 32;
        let spe = data.len() / np;
        let total = spe * epochs;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let t0 = Instant::now();
        for ep in 0..epochs {
            let mut dd = data.clone();
            dd.shuffle(&mut rng);
            for chunk in dd.chunks_exact(np) {
                let progress = steps as f32 / total as f32;
                let lr = base_lr * (0.01 + 0.99 * (0.5 * (1.0 + (progress * std::f32::consts::PI).cos())));
                emb.set_learning_rate(lr);
                let mut texts: Vec<String> = Vec::new();
                let mut tg = Vec::new();
                for p in chunk { texts.push(p.s1.clone()); texts.push(p.s2.clone()); tg.push(p.score); }
                let tr: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                emb.train_step_distill(&tr, &tg, 0.5);
                steps += 1;
                if steps % 500 == 0 { eprint!("\r ep {ep} step {steps}/{total} {:.0}s", t0.elapsed().as_secs_f64()); }
            }
        }
        train_secs = t0.elapsed().as_secs_f64();
        eprintln!();
    }

    // Export embeddings of all eval sentences
    let sents: Vec<String> = serde_json::from_reader(File::open(&sent_path).unwrap()).unwrap();
    let t0 = Instant::now();
    let mut buf: Vec<u8> = Vec::with_capacity(sents.len() * d * 4);
    let (mut spikes, mut slots) = (0usize, 0usize);
    for chunk in sents.chunks(64) {
        let low: Vec<String> = chunk.iter().map(|s| s.to_lowercase()).collect();
        let tr: Vec<&str> = low.iter().map(|s| s.as_str()).collect();
        let before = emb.metrics.embedding_spikes;
        let e = emb.encode(&tr);
        spikes += emb.metrics.embedding_spikes - before;
        slots += emb.cached_actual_lengths.as_ref().unwrap().iter().sum::<usize>() * d;
        for v in e { for x in v { buf.extend_from_slice(&x.to_le_bytes()); } }
    }
    let enc_secs = t0.elapsed().as_secs_f64();
    File::create(format!("{out}.emb.bin")).unwrap().write_all(&buf).unwrap();
    // E3 shuffle test: same model, word-order-shuffled eval sentences (row i = shuffled version of sentence i)
    let sents2: Vec<String> = serde_json::from_reader(File::open(arg("--sentences-shuf", "data_eval/all_eval_sentences_shuffled.json")).unwrap()).unwrap();
    let mut buf2: Vec<u8> = Vec::new();
    for chunk in sents2.chunks(64) {
        let low: Vec<String> = chunk.iter().map(|s| s.to_lowercase()).collect();
        let tr: Vec<&str> = low.iter().map(|s| s.as_str()).collect();
        for v in emb.encode(&tr) { for x in v { buf2.extend_from_slice(&x.to_le_bytes()); } }
    }
    File::create(format!("{out}.shuf.emb.bin")).unwrap().write_all(&buf2).unwrap();
    let meta = serde_json::json!({
        "d": d, "seed": seed, "epochs": epochs, "steps": steps, "train_seconds": train_secs,
        "encode_ms_per_sentence": enc_secs * 1000.0 / sents.len() as f64,
        "n_sentences": sents.len(), "emb_spikes_total": spikes, "token_slots_total": slots,
        "sparsity_excl_pad": 1.0 - spikes as f64 / slots as f64,
        "spikes_per_sentence": spikes as f64 / sents.len() as f64,
        "vocab_size": vocab, "max_len": max_len, "lr": base_lr, "data": data_path
    });
    std::fs::write(format!("{out}.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    println!("{}", meta);
}

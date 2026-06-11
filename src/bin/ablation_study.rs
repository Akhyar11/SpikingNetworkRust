use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use SpikingNetworkRust::models::sentence_embedder::SpikingSentenceEmbedder;
use SpikingNetworkRust::models::sentence_embedder;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::File;
use std::io::{BufReader, Write};
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct DistillationPair {
    s1: String,
    s2: String,
    score: f32,
}

#[derive(Deserialize)]
struct STSPair {
    sentence1: String,
    sentence2: String,
    score: f32,
}

fn apply_init_weights(embedder: &mut SpikingSentenceEmbedder, init_data: &serde_json::Value) {
    for group in &["embedding", "attention", "pooler"] {
        let layer: &mut dyn Layer = match *group {
            "embedding" => &mut embedder.embedding,
            "attention" => &mut embedder.attention,
            _           => &mut embedder.pooler,
        };
        if let Some(obj) = init_data.get(group).and_then(|v| v.as_object()) {
            for (k, v) in obj {
                if let Ok(data) = serde_json::from_value::<Vec<f32>>(v.clone()) {
                    let _ = layer.set_parameter(k, &data);
                }
            }
        }
    }
}

fn train_model(
    tokenizer: BPETokenizer,
    vocab_size: usize,
    dataset: &[DistillationPair],
    init_data: &serde_json::Value,
    d_model: usize,
    max_seq_length: usize,
    use_attention: bool,
) -> SpikingSentenceEmbedder {
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

    let mut embedder = SpikingSentenceEmbedder::new(tokenizer, vocab_size, snn_config);
    // Terapkan bobot inisialisasi bersama — eliminasi bias inisialisasi
    apply_init_weights(&mut embedder, init_data);
    embedder.set_use_attention(use_attention);

    let num_pairs = 32;
    let num_epochs = 1;
    let total_steps = (dataset.len() / num_pairs) * num_epochs;
    let mut global_step = 0;

    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut data = dataset.to_vec();

    for _ in 1..=num_epochs {
        data.shuffle(&mut rng);
        let mut batch_texts = Vec::new();
        let mut batch_targets = Vec::new();

        for pair in &data {
            batch_texts.push(pair.s1.clone());
            batch_texts.push(pair.s2.clone());
            batch_targets.push(pair.score);

            if batch_targets.len() == num_pairs {
                let lr = 0.01 * f32::max(0.01, 1.0 - (global_step as f32 / total_steps as f32));
                embedder.set_learning_rate(lr);
                let texts_str: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();
                embedder.train_step_distill(&texts_str, &batch_targets, 0.2);
                batch_texts.clear();
                batch_targets.clear();
                global_step += 1;
            }
        }
    }

    embedder
}

fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let mut mean1 = 0.0_f32;
    let mut mean2 = 0.0_f32;
    for i in 0..vec1.len() { mean1 += vec1[i]; mean2 += vec2[i]; }
    mean1 /= vec1.len() as f32;
    mean2 /= vec2.len() as f32;

    let (mut dot, mut n1, mut n2) = (0.0_f32, 0.0_f32, 0.0_f32);
    for i in 0..vec1.len() {
        let a = vec1[i] - mean1;
        let b = vec2[i] - mean2;
        dot += a * b; n1 += a * a; n2 += b * b;
    }
    if n1 == 0.0 || n2 == 0.0 { return 0.0; }
    (dot / (n1.sqrt() * n2.sqrt())).max(0.0)
}

fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    if n == 0.0 { return 0.0; }
    let (sx, sy): (f32, f32) = (x.iter().sum(), y.iter().sum());
    let (sxx, syy, sxy): (f32, f32, f32) = (
        x.iter().map(|&v| v * v).sum(),
        y.iter().map(|&v| v * v).sum(),
        x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum(),
    );
    let num = n * sxy - sx * sy;
    let den = ((n * sxx - sx * sx).max(0.0) * (n * syy - sy * sy).max(0.0)).sqrt();
    if den == 0.0 { 0.0 } else { num / den }
}

fn evaluate(embedder: &mut SpikingSentenceEmbedder, eval_data: &[STSPair]) -> (f32, f64) {
    let mut preds = Vec::new();
    let mut targets = Vec::new();
    let t0 = Instant::now();

    for pair in eval_data {
        let s1 = pair.sentence1.to_lowercase();
        let s2 = pair.sentence2.to_lowercase();
        let embs = embedder.encode(&[s1.as_str(), s2.as_str()]);
        preds.push(cosine_similarity(&embs[0], &embs[1]));
        targets.push(pair.score);
    }

    let pearson = pearson_correlation(&preds, &targets);
    let ms_per_pair = t0.elapsed().as_secs_f64() * 1000.0 / eval_data.len() as f64;
    (pearson, ms_per_pair)
}

fn main() {
    let vocab_path = "experiment/file_model/vocab.json";
    // Gunakan Knowledge Distillation (AI) — terbukti terbaik dari full_eval_controlled
    let dataset_path = "experiment/file_model/teacher_distillation_dataset.json";
    let eval_path = "experiment/file_model/sts-b_valid.json";
    let output_path = "experiment/file_model/ablation_results_distil.json";

    println!("Memuat tokenizer...");
    let tokenizer = BPETokenizer::load(vocab_path);
    let vocab_size = tokenizer.vocab_size();

    println!("Memuat dataset training...");
    let train_file = File::open(dataset_path).expect("Dataset tidak ditemukan");
    let dataset: Vec<DistillationPair> = serde_json::from_reader(BufReader::new(train_file)).unwrap();

    println!("Memuat dataset evaluasi STS-B...");
    let eval_file = File::open(eval_path).expect("sts-b_valid.json tidak ditemukan");
    let eval_data: Vec<STSPair> = serde_json::from_reader(BufReader::new(eval_file)).unwrap();

    println!("Total training pairs: {}, Eval pairs: {}", dataset.len(), eval_data.len());

    // Muat bobot inisialisasi bersama
    let init_path = "experiment/file_model/init_weights.json";
    println!("Memuat bobot inisialisasi dari {}...", init_path);
    let init_file = File::open(init_path)
        .expect("init_weights.json tidak ditemukan! Jalankan: cargo run --release --bin generate_init_weights");
    let init_data: serde_json::Value = serde_json::from_reader(BufReader::new(init_file)).unwrap();
    println!("✓ Semua varian akan mulai dari bobot inisialisasi yang IDENTIK.\n");

    // Definisi semua variasi ablasi
    let ablation_configs: Vec<(&str, usize, bool)> = vec![
        ("T=16 (with-attention)",  16, true),
        ("T=32 (with-attention)",  32, true),   // Baseline
        ("T=64 (with-attention)",  64, true),
        ("T=32 (no-attention)",    32, false),   // No-Attention ablation
    ];

    let mut all_results = serde_json::Map::new();

    println!("\n=======================================================");
    println!(" ABLASI ARSITEKTUR — Dataset: Knowledge Distillation AI ");
    println!("=======================================================");
    println!(" {:<26} | {:>8} | {:>10} | {:>12} | {:>10}", "Konfigurasi", "Pearson", "ms/pair", "Training(s)", "SOPs/kalimat");
    println!("-------------------------------------------------------------------");

    for (label, max_seq_length, use_attention) in &ablation_configs {
        print!(" {:<26} | training...", label);
        std::io::stdout().flush().unwrap();

        let t_train = Instant::now();
        let mut embedder = train_model(
            tokenizer.clone_with_same_vocab(),
            vocab_size,
            &dataset,
            &init_data,
            64,
            *max_seq_length,
            *use_attention,
        );
        let train_secs = t_train.elapsed().as_secs_f64();

        let (pearson, ms_per_pair) = evaluate(&mut embedder, &eval_data);

        let avg_sops = embedder.metrics.total_sops as f64
            / embedder.metrics.total_sentences.max(1) as f64;

        println!("\r {:<26} | {:>8.4} | {:>10.2} | {:>12.2} | {:>10.0}",
            label, pearson, ms_per_pair, train_secs, avg_sops);

        all_results.insert(label.to_string(), json!({
            "training_dataset": "Knowledge Distillation (AI)",
            "max_seq_length": max_seq_length,
            "use_attention": use_attention,
            "pearson_correlation": pearson,
            "ms_per_pair": ms_per_pair,
            "train_time_seconds": train_secs,
            "average_sops": avg_sops,
        }));
    }

    println!("=======================================================");

    // Simpan ke JSON
    let summary = json!({ "ablation_study": all_results });
    let mut out = File::create(output_path).expect("Gagal membuat file output");
    out.write_all(serde_json::to_string_pretty(&summary).unwrap().as_bytes()).unwrap();
    println!("\nHasil ablasi disimpan ke: {}", output_path);
}

/// Evaluasi Sensitivitas Parameter Biologis (Beta & Threshold)
/// Membandingkan inisialisasi heterogen (beragam per neuron) vs homogen (seragam)
/// pada representasi semantik dan efisiensi energi SNN.
use std::fs::File;
use std::io::{BufReader, Write};
use std::time::Instant;
use serde_json::json;

use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use SpikingNetworkRust::models::sentence_embedder::{SpikingSentenceEmbedder, SNNConfig};

#[derive(serde::Deserialize, Clone)]
struct PairScored {
    s1: String,
    s2: String,
    score: f32,
}

#[derive(serde::Deserialize, Clone)]
struct STSPair {
    sentence1: String,
    sentence2: String,
    score: f32,
}

// ─── Setup Model helper ───
fn new_embedder(tokenizer: BPETokenizer, vocab_size: usize) -> SpikingSentenceEmbedder {
    let snn_config = SNNConfig {
        d_model: 64,
        max_seq_length: 32,
        learning_rate: 0.01,
        clip_min: -1.0,
        clip_max: 1.0,
        att_beta_range: (0.8, 0.99),
        att_threshold_range: (0.1, 0.3),
        bptt_beta_range: (0.8, 0.99),
        bptt_threshold_range: (0.5, 1.0),
    };
    SpikingSentenceEmbedder::new(tokenizer, vocab_size, snn_config)
}

fn load_model_with_weights(
    tokenizer: BPETokenizer,
    vocab_size: usize,
    init_data: &serde_json::Value,
    force_homogeneous: bool
) -> SpikingSentenceEmbedder {
    let mut embedder = new_embedder(tokenizer, vocab_size);

    // 1. Muat parameter embedding
    if let Some(emb_obj) = init_data.get("embedding") {
        for (name, val) in emb_obj.as_object().unwrap() {
            let vec: Vec<f32> = serde_json::from_value(val.clone()).unwrap();
            embedder.embedding.set_parameter(name, &vec).unwrap();
        }
    }

    // 2. Muat parameter attention
    if let Some(att_obj) = init_data.get("attention") {
        for (name, val) in att_obj.as_object().unwrap() {
            let mut vec: Vec<f32> = serde_json::from_value(val.clone()).unwrap();
            if force_homogeneous {
                if name.starts_with("beta") {
                    // Paksa beta seragam = 0.90
                    for x in vec.iter_mut() { *x = 0.90; }
                } else if name.starts_with("threshold") {
                    // Paksa threshold seragam = 0.20 (att)
                    for x in vec.iter_mut() { *x = 0.20; }
                }
            }
            embedder.attention.set_parameter(name, &vec).unwrap();
        }
    }

    // 3. Muat parameter pooler
    if let Some(pool_obj) = init_data.get("pooler") {
        for (name, val) in pool_obj.as_object().unwrap() {
            let mut vec: Vec<f32> = serde_json::from_value(val.clone()).unwrap();
            if force_homogeneous {
                if name.starts_with("beta") {
                    // Paksa beta seragam = 0.90
                    for x in vec.iter_mut() { *x = 0.90; }
                } else if name.starts_with("threshold") {
                    // Paksa threshold seragam = 0.75 (pooler)
                    for x in vec.iter_mut() { *x = 0.75; }
                }
            }
            embedder.pooler.set_parameter(name, &vec).unwrap();
        }
    }

    embedder
}

// ─── Training helper ───
fn train_model(
    mut embedder: SpikingSentenceEmbedder,
    dataset: &[PairScored],
    subset_size: usize
) -> SpikingSentenceEmbedder {
    let num_pairs = 32;
    let limit_data = &dataset[0..subset_size.min(dataset.len())];
    let total_steps = limit_data.len() / num_pairs;
    let mut step = 0;

    let mut batch_texts = Vec::new();
    let mut batch_targets = Vec::new();

    for pair in limit_data {
        batch_texts.push(pair.s1.clone());
        batch_texts.push(pair.s2.clone());
        batch_targets.push(pair.score);

        if batch_targets.len() == num_pairs {
            let lr = 0.01 * f32::max(0.01, 1.0 - (step as f32 / total_steps as f32));
            embedder.set_learning_rate(lr);
            let texts: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();
            embedder.train_step_distill(&texts, &batch_targets, 0.2);
            batch_texts.clear();
            batch_targets.clear();
            step += 1;
        }
    }
    embedder
}

// ─── Evaluation helper ───
fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let mut mean1 = 0.0;
    let mut mean2 = 0.0;
    for i in 0..vec1.len() {
        mean1 += vec1[i];
        mean2 += vec2[i];
    }
    mean1 /= vec1.len() as f32;
    mean2 /= vec2.len() as f32;

    let (mut dot, mut n1, mut n2) = (0.0, 0.0, 0.0);
    for i in 0..vec1.len() {
        let a = vec1[i] - mean1;
        let b = vec2[i] - mean2;
        dot += a * b;
        n1 += a * a;
        n2 += b * b;
    }
    if n1 == 0.0 || n2 == 0.0 { return 0.0; }
    (dot / (n1.sqrt() * n2.sqrt())).max(0.0)
}

fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    if n == 0.0 { return 0.0; }
    let sx: f32 = x.iter().sum();
    let sy: f32 = y.iter().sum();
    let sxx: f32 = x.iter().map(|&v| v * v).sum();
    let syy: f32 = y.iter().map(|&v| v * v).sum();
    let sxy: f32 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();

    let num = n * sxy - sx * sy;
    let den = ((n * sxx - sx * sx).max(0.0) * (n * syy - sy * sy).max(0.0)).sqrt();
    if den == 0.0 { 0.0 } else { num / den }
}

fn evaluate(embedder: &mut SpikingSentenceEmbedder, eval_data: &[STSPair]) -> (f32, f64, f64) {
    embedder.metrics = SpikingNetworkRust::models::sentence_embedder::SNNMetrics::default();
    let t0 = Instant::now();
    let mut preds = Vec::new();
    let mut targets = Vec::new();

    for pair in eval_data {
        let v1 = embedder.encode(&[&pair.sentence1])[0].clone();
        let v2 = embedder.encode(&[&pair.sentence2])[0].clone();
        preds.push(cosine_similarity(&v1, &v2));
        targets.push(pair.score);
    }

    let pearson = pearson_correlation(&preds, &targets);
    let ms_per_pair = t0.elapsed().as_secs_f64() * 1000.0 / eval_data.len() as f64;
    
    let total_sops = embedder.metrics.total_sops as f64;
    let total_sentences = embedder.metrics.total_sentences.max(1) as f64;
    let avg_sops = total_sops / total_sentences;

    (pearson, ms_per_pair, avg_sops)
}

fn main() {
    let vocab_path = "experiment/file_model/vocab.json";
    let dataset_path = "experiment/file_model/teacher_distillation_dataset.json";
    let eval_path = "experiment/file_model/sts-b_valid.json";
    let init_path = "experiment/file_model/init_weights.json";
    let output_path = "experiment/file_model/biological_sensitivity.json";

    println!("Memuat tokenizer...");
    let tokenizer = BPETokenizer::load(vocab_path);
    let vocab_size = tokenizer.vocab_size();

    println!("Memuat dataset training (Distil AI)...");
    let train_file = File::open(dataset_path).expect("Dataset tidak ditemukan");
    let dataset: Vec<PairScored> = serde_json::from_reader(BufReader::new(train_file)).unwrap();

    println!("Memuat dataset evaluasi STS-B...");
    let eval_file = File::open(eval_path).expect("sts-b_valid.json tidak ditemukan");
    let eval_data: Vec<STSPair> = serde_json::from_reader(BufReader::new(eval_file)).unwrap();

    println!("Memuat inisialisasi bobot...");
    let init_file = File::open(init_path).expect("init_weights.json tidak ditemukan");
    let init_data: serde_json::Value = serde_json::from_reader(BufReader::new(init_file)).unwrap();

    // Jalankan 15.000 subset pairs saja agar cepat dan data efisien
    let subset_size = 15000;
    println!("Menggunakan {} training pairs dari dataset distilasi...", subset_size);

    println!("\n=====================================================================");
    println!("       UJI SENSITIVITAS PARAMETER BIOLOGIS (HETEROGEN VS HOMOGEN)    ");
    println!("=====================================================================");

    // ─── Evaluasi 1: Model Heterogen (Biologis Asli) ───
    println!("1. Melatih Model HETEROGEN (Biologis Asli)...");
    let mut model_het = load_model_with_weights(tokenizer.clone_with_same_vocab(), vocab_size, &init_data, false);
    let model_het = train_model(model_het, &dataset, subset_size);
    let mut embedder_het = model_het;
    let (pearson_het, ms_het, sops_het) = evaluate(&mut embedder_het, &eval_data);
    println!("   ✓ Pearson: {:.4} | SOPs/kalimat: {:.0} | ms/pasang: {:.2}", pearson_het, sops_het, ms_het);

    // ─── Evaluasi 2: Model Homogen (Seragam) ───
    println!("2. Melatih Model HOMOGEN (Seragam: beta=0.90, threshold=0.75/0.20)...");
    let mut model_hom = load_model_with_weights(tokenizer.clone_with_same_vocab(), vocab_size, &init_data, true);
    let model_hom = train_model(model_hom, &dataset, subset_size);
    let mut embedder_hom = model_hom;
    let (pearson_hom, ms_hom, sops_hom) = evaluate(&mut embedder_hom, &eval_data);
    println!("   ✓ Pearson: {:.4} | SOPs/kalimat: {:.0} | ms/pasang: {:.2}", pearson_hom, sops_hom, ms_hom);

    println!("=====================================================================");

    // Simpan ke JSON
    let result = json!({
        "heterogeneous": {
            "pearson": pearson_het,
            "average_sops": sops_het,
            "ms_per_pair": ms_het
        },
        "homogeneous": {
            "pearson": pearson_hom,
            "average_sops": sops_hom,
            "ms_per_pair": ms_hom
        }
    });

    let mut out = File::create(output_path).unwrap();
    out.write_all(serde_json::to_string_pretty(&result).unwrap().as_bytes()).unwrap();
    println!("Hasil uji sensitivitas disimpan ke: {}", output_path);
}

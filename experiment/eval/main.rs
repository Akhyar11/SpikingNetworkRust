#[path = "../model/sentence_embedder.rs"]
pub mod sentence_embedder;

use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use sentence_embedder::SpikingSentenceEmbedder;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use serde::Deserialize;

#[derive(Deserialize)]
struct STSPair {
    sentence1: String,
    sentence2: String,
    score: f32,
}

fn load_model(path: &str) -> (serde_json::Value, usize, usize) {
    let file = File::open(path).unwrap_or_else(|_| panic!("Gagal buka file model di {}. Anda mungkin harus menjalankan proses train minimal 1 iterasi terlebih dahulu.", path));
    let reader = BufReader::new(file);
    let model_data: serde_json::Value = serde_json::from_reader(reader).expect("Gagal memecah data model JSON");

    // Jika model_data tidak menyimpan d_model dan max_seq_length, kita bisa ambil default (64 dan 128)
    let d_model = model_data.get("d_model").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let max_seq_length = model_data.get("max_seq_length").and_then(|v| v.as_u64()).unwrap_or(128) as usize;

    (model_data, d_model, max_seq_length)
}

fn apply_weights(embedder: &mut SpikingSentenceEmbedder, model_data: &serde_json::Value) {
    let load_layer = |layer: &mut dyn Layer, group: &str| {
        if let Some(obj) = model_data.get(group).and_then(|v| v.as_object()) {
            for (k, v) in obj {
                let data: Vec<f32> = serde_json::from_value(v.clone()).unwrap();
                layer.set_parameter(k, &data).unwrap();
            }
        }
    };

    load_layer(&mut embedder.embedding, "embedding");
    load_layer(&mut embedder.attention, "attention");
    load_layer(&mut embedder.pooler, "pooler");
}

fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let mut mean1 = 0.0;
    let mut mean2 = 0.0;
    for i in 0..vec1.len() {
        mean1 += vec1[i];
        mean2 += vec2[i];
    }
    mean1 /= vec1.len() as f32;
    mean2 /= vec2.len() as f32;

    let mut dot = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    for i in 0..vec1.len() {
        let val1 = vec1[i] - mean1;
        let val2 = vec2[i] - mean2;
        dot += val1 * val2;
        norm1 += val1 * val1;
        norm2 += val2 * val2;
    }
    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }
    let sim = dot / (norm1.sqrt() * norm2.sqrt());
    sim.max(0.0) // Hindari persentase negatif
}

fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let sum_x: f32 = x.iter().sum();
    let sum_y: f32 = y.iter().sum();
    let sum_x_sq: f32 = x.iter().map(|&v| v * v).sum();
    let sum_y_sq: f32 = y.iter().map(|&v| v * v).sum();
    let sum_xy: f32 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let var_x = (n * sum_x_sq - sum_x * sum_x).max(0.0);
    let var_y = (n * sum_y_sq - sum_y * sum_y).max(0.0);
    let denominator = (var_x * var_y).sqrt();
    
    if denominator == 0.0 { 0.0 } else { numerator / denominator }
}

fn main() {
    let vocab_path = "experiment/file_model/vocab.json";
    let model_save_path = "experiment/file_model/saved_model.json";
    let eval_dataset_path = "experiment/file_model/sts-b_valid.json";

    println!("Memuat tokenizer...");
    let tokenizer = BPETokenizer::load(vocab_path);
    let vocab_size = tokenizer.vocab_size();
    println!("Memuat bobot parameter dari disk ({})...", model_save_path);
    let (model_data, d_model, max_seq_length) = load_model(model_save_path);

    println!("Inisialisasi arsitektur jaringan SpikingSentenceEmbedder (D_Model: {}, Max_Seq: {})...", d_model, max_seq_length);
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
    
    apply_weights(&mut embedder, &model_data);

    println!("Membuka dataset evaluasi dari {}...", eval_dataset_path);
    let eval_file = File::open(eval_dataset_path).expect("File dataset sts-b_valid.json tidak ditemukan!");
    let reader = BufReader::new(eval_file);
    let dataset: Vec<STSPair> = serde_json::from_reader(reader).expect("Format JSON invalid pada dataset STS-B");
    let total = dataset.len();

    let mut predictions = Vec::with_capacity(total);
    let mut targets = Vec::with_capacity(total);

    let start_time = Instant::now();
    let mut step = 0;

    println!("Mulai Evaluasi SNN (Total: {} pasang kalimat)...", total);

    for pair in dataset {
        let texts = [pair.sentence1.as_str(), pair.sentence2.as_str()];
        // Panggil Forward Pass (Inference)
        let embeddings = embedder.encode(&texts);
        
        let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
        
        predictions.push(sim);
        targets.push(pair.score);

        step += 1;
        if step % 200 == 0 {
            println!("  Progres: {:>4}/{} pasang dievaluasi.", step, total);
        }
    }

    let duration = start_time.elapsed().as_secs_f64();
    let pearson = pearson_correlation(&predictions, &targets);
    
    println!("\n=============================================");
    println!("             HASIL EVALUASI SNN              ");
    println!("=============================================");
    println!(" Total Pasangan   : {}", total);
    println!(" Waktu Inferensi  : {:.2} detik", duration);
    println!(" Kecepatan        : {:.2} ms / pasang", (duration * 1000.0) / total as f64);
    println!(" Pearson (STS-B)  : {:.4}", pearson);
    println!("=============================================");
}

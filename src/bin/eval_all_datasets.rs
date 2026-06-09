use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use SpikingNetworkRust::models::sentence_embedder::SpikingSentenceEmbedder;
use SpikingNetworkRust::models::sentence_embedder;
use std::fs::File;
use std::io::{BufReader, Write};
use std::time::Instant;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct STSPair {
    sentence1: String,
    sentence2: String,
    score: f32,
}

fn load_model(path: &str) -> (serde_json::Value, usize, usize) {
    let file = File::open(path).expect("Gagal buka file model");
    let reader = BufReader::new(file);
    let model_data: serde_json::Value = serde_json::from_reader(reader).expect("Gagal memecah data model JSON");
    let d_model = model_data.get("d_model").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let max_seq_length = model_data.get("max_seq_length").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
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
    if norm1 == 0.0 || norm2 == 0.0 { return 0.0; }
    (dot / (norm1.sqrt() * norm2.sqrt())).max(0.0)
}

fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    if n == 0.0 { return 0.0; }
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
    let model_save_path = std::env::args().nth(1).unwrap_or_else(|| "experiment/file_model/saved_model_human.json".to_string());
    
    let datasets = vec![
        ("STS-12", "experiment/file_model/mteb_sts12-sts.json"),
        ("STS-13", "experiment/file_model/mteb_sts13-sts.json"),
        ("STS-14", "experiment/file_model/mteb_sts14-sts.json"),
        ("STS-15", "experiment/file_model/mteb_sts15-sts.json"),
        ("STS-16", "experiment/file_model/mteb_sts16-sts.json"),
        ("SICK-R", "experiment/file_model/mteb_sickr-sts.json"),
        ("STS-B", "experiment/file_model/sts-b_valid.json"),
    ];

    println!("Memuat tokenizer...");
    let tokenizer = BPETokenizer::load(vocab_path);
    let vocab_size = tokenizer.vocab_size();
    println!("Memuat model dari disk ({})...", model_save_path);
    let (model_data, d_model, max_seq_length) = load_model(&model_save_path);

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

    let mut all_results = serde_json::Map::new();

    println!("=============================================");
    println!("     EVALUASI GENERALISASI OUT-OF-DOMAIN     ");
    println!("=============================================");

    let mut total_inference_time = 0.0;
    let mut total_pairs_all = 0;

    let mut embedder = SpikingSentenceEmbedder::new(tokenizer, vocab_size, snn_config);
    apply_weights(&mut embedder, &model_data);

    for (ds_name, ds_path) in datasets {
        let eval_file = File::open(ds_path);
        if eval_file.is_err() {
            println!("Dataset {} tidak ditemukan di {}", ds_name, ds_path);
            continue;
        }
        let reader = BufReader::new(eval_file.unwrap());
        let dataset: Vec<STSPair> = serde_json::from_reader(reader).expect("Format JSON invalid");
        
        let total = dataset.len();
        if total == 0 { continue; }

        let mut predictions = Vec::with_capacity(total);
        let mut targets = Vec::with_capacity(total);

        let start_time = Instant::now();

        for pair in dataset {
            let s1 = pair.sentence1.to_lowercase();
            let s2 = pair.sentence2.to_lowercase();
            let texts = [s1.as_str(), s2.as_str()];
            let embeddings = embedder.encode(&texts);
            let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
            predictions.push(sim);
            targets.push(pair.score);
        }

        let duration = start_time.elapsed().as_secs_f64();
        total_inference_time += duration;
        total_pairs_all += total;

        let pearson = pearson_correlation(&predictions, &targets);
        
        let avg_sops = embedder.metrics.total_sops as f64 / embedder.metrics.total_sentences as f64;
        let avg_spikes = (embedder.metrics.embedding_spikes + embedder.metrics.attention_spikes + embedder.metrics.pooler_spikes) as f64 / embedder.metrics.total_sentences as f64;

        println!(" {:<8} | Pairs: {:<5} | Pearson: {:.4} | Time: {:.2}s", ds_name, total, pearson, duration);

        let result_json = json!({
            "total_pairs": total,
            "pearson_correlation": pearson,
            "inference_time_seconds": duration,
            "average_spikes": avg_spikes,
            "average_sops": avg_sops
        });
        all_results.insert(ds_name.to_string(), result_json);
    }
    
    let summary = json!({
        "total_datasets_evaluated": all_results.len(),
        "total_pairs_evaluated": total_pairs_all,
        "total_inference_time_seconds": total_inference_time,
        "results": all_results
    });

    let metrics_path = "experiment/file_model/eval_all_datasets_metrics.json";
    let mut file = File::create(metrics_path).expect("Gagal membuat file output");
    file.write_all(serde_json::to_string_pretty(&summary).unwrap().as_bytes()).expect("Gagal menulis file output");

    println!("=============================================");
    println!("Hasil lengkap disimpan di: {}", metrics_path);
}

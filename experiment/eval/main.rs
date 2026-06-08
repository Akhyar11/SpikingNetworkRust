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
    let mut load_layer = |layer: &mut dyn Layer, group: &str| {
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

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let sum_x: f32 = x.iter().sum();
    let sum_y: f32 = y.iter().sum();
    let sum_x_sq: f32 = x.iter().map(|&v| v * v).sum();
    let sum_y_sq: f32 = y.iter().map(|&v| v * v).sum();
    let sum_xy: f32 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();
    
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
    let mut embedder = SpikingSentenceEmbedder::new(tokenizer, vocab_size, d_model, max_seq_length);
    
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
        
        if step < 3 {
            println!("Debug Pair {}: sim={}", step, sim);
            println!("  Text 1 (len {}): {}", texts[0].len(), texts[0]);
            let tok1 = embedder.tokenizer.encode(texts[0]);
            println!("  Tok  1: {:?}", tok1);
            println!("  Text 2 (len {}): {}", texts[1].len(), texts[1]);
            let tok2 = embedder.tokenizer.encode(texts[1]);
            println!("  Tok  2: {:?}", tok2);
            println!("  Emb 1 sample: {:?}", &embeddings[0][..5]);
            println!("  Emb 2 sample: {:?}", &embeddings[1][..5]);
        }
        
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

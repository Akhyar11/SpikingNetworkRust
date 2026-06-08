#[path = "../model/sentence_embedder.rs"]
pub mod sentence_embedder;

use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use sentence_embedder::SpikingSentenceEmbedder;
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::time::Instant;

fn load_model(path: &str) -> (serde_json::Value, usize, usize) {
    let file = File::open(path).unwrap_or_else(|_| panic!("Gagal buka file model di {}. Anda mungkin harus menjalankan proses train minimal 1 iterasi terlebih dahulu.", path));
    let reader = BufReader::new(file);
    let model_data: serde_json::Value = serde_json::from_reader(reader).expect("Gagal memecah data model JSON");

    let d_model = model_data.get("d_model").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let max_seq_length = model_data.get("max_seq_length").and_then(|v| v.as_u64()).unwrap_or(128) as usize;

    (model_data, d_model, max_seq_length)
}

fn apply_weights(embedder: &mut SpikingSentenceEmbedder, model_data: &serde_json::Value) {
    let load_layer = |layer: &mut dyn Layer, group: &str| {
        if let Some(obj) = model_data.get(group).and_then(|v| v.as_object()) {
            for (k, v) in obj {
                let data: Vec<f32> = serde_json::from_value(v.clone()).unwrap();
                // Ignored result, just like eval/main.rs
                let _ = layer.set_parameter(k, &data);
            }
        }
    };

    load_layer(&mut embedder.embedding, "embedding");
    // load_layer(&mut embedder.attention, "attention"); // Telah dihapus
    load_layer(&mut embedder.pooler, "pooler");
}

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

fn main() {
    println!("Memuat tokenizer...");
    let tokenizer = BPETokenizer::load("experiment/file_model/vocab.json");
    let vocab_size = 32000;

    println!("Memuat bobot parameter dari disk (experiment/file_model/saved_model.json)...");
    let (model_data, d_model, max_seq_length) = load_model("experiment/file_model/saved_model.json");

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
    let mut embedder = SpikingSentenceEmbedder::new(
        tokenizer,
        vocab_size,
        snn_config,
    );

    apply_weights(&mut embedder, &model_data);

    println!("Model berhasil dimuat! Ketik 'exit' untuk keluar.\n");

    let stdin = io::stdin();
    
    loop {
        let mut text_a = String::new();
        let mut text_b = String::new();

        print!("Kalimat A: ");
        io::stdout().flush().unwrap();
        stdin.read_line(&mut text_a).unwrap();
        let text_a = text_a.trim();
        
        if text_a.eq_ignore_ascii_case("exit") { break; }

        print!("Kalimat B: ");
        io::stdout().flush().unwrap();
        stdin.read_line(&mut text_b).unwrap();
        let text_b = text_b.trim();
        
        if text_b.eq_ignore_ascii_case("exit") { break; }

        if text_a.is_empty() || text_b.is_empty() {
            println!("Kalimat tidak boleh kosong!");
            continue;
        }

        let start = Instant::now();
        
        let texts = vec![text_a, text_b];
        let embeddings = embedder.encode(&texts);
        
        let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
        let duration = start.elapsed();

        println!("\n=============================================");
        println!("  Text A: {}", text_a);
        println!("  Text B: {}", text_b);
        println!("---------------------------------------------");
        println!("  Cosine Similarity : {:.4}", sim);
        println!("  Waktu Komputasi   : {:?}", duration);
        
        println!("---------------------------------------------");
        println!("  Sample Embedding A: {:?}", &embeddings[0][..10.min(embeddings[0].len())]);
        println!("  Sample Embedding B: {:?}", &embeddings[1][..10.min(embeddings[1].len())]);
        println!("=============================================\n");
    }
}

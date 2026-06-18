use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use SpikingNetworkRust::models::sentence_embedder::SpikingSentenceEmbedder;
use SpikingNetworkRust::models::sentence_embedder;
use std::fs::{self, File};
use std::io::{BufReader, Write};

fn load_init(path: &str) -> serde_json::Value {
    let f = File::open(path).expect("Trained weights (init_weights.json) not found!");
    serde_json::from_reader(BufReader::new(f)).unwrap()
}

fn main() {
    println!("Mempersiapkan export untuk Hugging Face Hub...");
    
    let out_dir = "huggingface_export";
    fs::create_dir_all(out_dir).unwrap();

    let vocab_path = "experiment/file_model/vocab.json";
    
    // Copy tokenizer
    fs::copy(vocab_path, format!("{}/tokenizer.json", out_dir)).unwrap_or_else(|_| 0);
    
    let d_model = 384;
    let max_seq_length = 128;

    let mut weights_path = "experiment/file_model/trained_weights.json";
    if !std::path::Path::new(weights_path).exists() {
        println!("trained_weights.json tidak ditemukan, menggunakan init_weights.json sebagai fallback.");
        weights_path = "experiment/file_model/init_weights.json";
    }
    
    let init_val = load_init(weights_path);

    // Menyimpan config.json
    let config = serde_json::json!({
        "architectures": ["SpikingSentenceEmbedder"],
        "model_type": "spiking_snn",
        "d_model": d_model,
        "max_position_embeddings": max_seq_length,
        "neuron_type": "homogeneous_lif",
        "beta": 0.90,
        "margin": 0.05,
        "auto_map": {
            "AutoModel": "modeling_spiking.SpikingSentenceEmbedder"
        }
    });

    let mut config_file = File::create(format!("{}/config.json", out_dir)).unwrap();
    config_file.write_all(serde_json::to_string_pretty(&config).unwrap().as_bytes()).unwrap();

    // Copy bobot JSON langsung
    fs::copy(weights_path, format!("{}/model_weights.json", out_dir)).unwrap();
    
    println!("Export selesai! Anda dapat mengunggah isi folder '{}' ke Hugging Face.", out_dir);
}

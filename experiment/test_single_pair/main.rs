#[path = "../model/sentence_embedder.rs"]
pub mod sentence_embedder;

use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;

use sentence_embedder::SpikingSentenceEmbedder;
use serde_json::Value;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: test_single_pair \"Sentence A\" \"Sentence B\"");
        std::process::exit(1);
    }

    let text_a = &args[1];
    let text_b = &args[2];

    let vocab_path = "experiment/file_model/vocab.json";
    let model_save_path = "experiment/file_model/saved_model_human.json";

    let tokenizer = BPETokenizer::load(vocab_path);
    let vocab_size = tokenizer.vocab_size();

    // Load Weights
    let model_data_str = fs::read_to_string(model_save_path).expect("Gagal memuat model");
    let model_data: Value = serde_json::from_str(&model_data_str).unwrap();

    let d_model = model_data["d_model"].as_u64().unwrap_or(128) as usize;
    let max_seq_length = model_data["max_seq_length"].as_u64().unwrap_or(32) as usize;

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

    if let Value::Object(emb_weights) = &model_data["embedding"] {
        for (k, v) in emb_weights {
            let _ = embedder.embedding.set_parameter(k, &v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>());
        }
    }
    if let Value::Object(att_weights) = &model_data["attention"] {
        for (k, v) in att_weights {
            let _ = embedder.attention.set_parameter(k, &v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>());
        }
    }
    if let Value::Object(pooler_weights) = &model_data["pooler"] {
        for (k, v) in pooler_weights {
            let _ = embedder.pooler.set_parameter(k, &v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>());
        }
    }

    let texts = vec![text_a.as_str(), text_b.as_str()];
    let embeddings = embedder.encode(&texts);

    let emb_a = &embeddings[0];
    let emb_b = &embeddings[1];

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..d_model {
        dot += emb_a[i] * emb_b[i];
        norm_a += emb_a[i] * emb_a[i];
        norm_b += emb_b[i] * emb_b[i];
    }

    let sim = if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    };

    // Print hanya angka agar mudah dibaca JS
    println!("{:.4}", sim);
}

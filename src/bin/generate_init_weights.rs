/// Script ini menghasilkan SATU set bobot inisialisasi acak yang akan
/// digunakan bersama oleh semua varian ablasi. Jalankan SEKALI sebelum ablation_study.
use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use SpikingNetworkRust::models::sentence_embedder::SpikingSentenceEmbedder;
use SpikingNetworkRust::models::sentence_embedder;
use std::fs::File;
use std::io::Write;

fn main() {
    let vocab_path = "experiment/file_model/vocab.json";
    let output_path = "experiment/file_model/init_weights.json";

    // Gunakan d_model=64 yang dipakai di semua eksperimen.
    // max_seq_length di sini tidak mempengaruhi bentuk bobot,
    // hanya mempengaruhi ukuran buffer temporal (tidak disimpan).
    let d_model = 64;
    let max_seq_length = 32;

    println!("Memuat tokenizer...");
    let tokenizer = BPETokenizer::load(vocab_path);
    let vocab_size = tokenizer.vocab_size();

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

    println!("Inisialisasi model acak (sekali)...");
    let embedder = SpikingSentenceEmbedder::new(tokenizer, vocab_size, snn_config);

    // Kumpulkan semua parameter dari ketiga layer
    let mut init_data = serde_json::Map::new();
    init_data.insert("d_model".to_string(), serde_json::json!(d_model));

    let mut emb_params = serde_json::Map::new();
    for (name, data) in embedder.embedding.get_parameters() {
        emb_params.insert(name.to_string(), serde_json::json!(data));
    }
    init_data.insert("embedding".to_string(), serde_json::Value::Object(emb_params));

    let mut att_params = serde_json::Map::new();
    for (name, data) in embedder.attention.get_parameters() {
        att_params.insert(name.to_string(), serde_json::json!(data));
    }
    init_data.insert("attention".to_string(), serde_json::Value::Object(att_params));

    let mut pooler_params = serde_json::Map::new();
    for (name, data) in embedder.pooler.get_parameters() {
        pooler_params.insert(name.to_string(), serde_json::json!(data));
    }
    init_data.insert("pooler".to_string(), serde_json::Value::Object(pooler_params));

    let json_str = serde_json::to_string_pretty(&init_data).unwrap();
    let mut file = File::create(output_path).expect("Gagal membuat file output");
    file.write_all(json_str.as_bytes()).expect("Gagal menulis file");

    println!("✓ Bobot inisialisasi disimpan ke: {}", output_path);
    println!("  Jalankan ablation_study sekarang — semua varian akan mulai dari bobot yang sama.");
}

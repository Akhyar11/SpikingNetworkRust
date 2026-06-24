use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use SpikingNetworkRust::models::sentence_embedder::{SpikingSentenceEmbedder, SNNConfig};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

#[derive(Deserialize)]
struct STSPair {
    #[serde(alias = "s1")]
    sentence1: String,
    #[serde(alias = "s2")]
    sentence2: String,
    score: f32
}

fn load_weights(path: &str) -> serde_json::Value {
    let f = File::open(path)
        .unwrap_or_else(|_| panic!("{} tidak ditemukan!", path));
    serde_json::from_reader(BufReader::new(f)).unwrap()
}

fn apply_weights(embedder: &mut SpikingSentenceEmbedder, init: &serde_json::Value) {
    for group in &["embedding", "attention", "pooler"] {
        let layer: &mut dyn Layer = match *group {
            "embedding" => &mut embedder.embedding,
            "attention" => &mut embedder.attention,
            _           => &mut embedder.pooler,
        };
        if let Some(obj) = init.get(group).and_then(|v| v.as_object()) {
            for (k, v) in obj {
                if let Ok(data) = serde_json::from_value::<Vec<f32>>(v.clone()) {
                    let _ = layer.set_parameter(k, &data);
                }
            }
        }
    }
}

fn new_embedder(tokenizer: BPETokenizer, vocab_size: usize, d_model: usize, max_seq_length: usize) -> SpikingSentenceEmbedder {
    SpikingSentenceEmbedder::new(tokenizer, vocab_size, SNNConfig {
        d_model, max_seq_length, learning_rate: 0.01,
        clip_min: -1.0, clip_max: 1.0,
        att_beta_range: (0.8, 0.99), att_threshold_range: (-1.0, -0.5),
        bptt_beta_range: (0.8, 0.99), bptt_threshold_range: (0.5, 1.0),
    })
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let (mut ma, mut mb) = (0.0_f32, 0.0_f32);
    for i in 0..a.len() { ma += a[i]; mb += b[i]; }
    ma /= a.len() as f32; mb /= b.len() as f32;
    let (mut dot, mut na, mut nb) = (0.0_f32, 0.0_f32, 0.0_f32);
    for i in 0..a.len() {
        let (x, y) = (a[i]-ma, b[i]-mb);
        dot += x*y; na += x*x; nb += y*y;
    }
    if na == 0.0 || nb == 0.0 { return 0.0; }
    (dot / (na.sqrt() * nb.sqrt())).max(0.0)
}

fn pearson(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    if n == 0.0 { return 0.0; }
    let (sx, sy): (f32, f32) = (x.iter().sum(), y.iter().sum());
    let (sxx, syy, sxy): (f32, f32, f32) = (
        x.iter().map(|&v| v*v).sum(), y.iter().map(|&v| v*v).sum(),
        x.iter().zip(y.iter()).map(|(&a,&b)| a*b).sum(),
    );
    let num = n*sxy - sx*sy;
    let den = ((n*sxx - sx*sx).max(0.0) * (n*syy - sy*sy).max(0.0)).sqrt();
    if den == 0.0 { 0.0 } else { num/den }
}

fn evaluate_stsb(embedder: &mut SpikingSentenceEmbedder, eval_data: &[STSPair]) -> (f32, f64, f64) {
    let mut preds = Vec::new();
    let mut targets = Vec::new();
    let t0 = Instant::now();
    
    println!("    --- Contoh 5 Prediksi SNN vs Target Guru ---");
    for (i, pair) in eval_data.iter().enumerate() {
        let s1 = pair.sentence1.to_lowercase();
        let s2 = pair.sentence2.to_lowercase();
        let embs = embedder.encode(&[s1.as_str(), s2.as_str()]);
        let sim = cosine_sim(&embs[0], &embs[1]);
        preds.push(sim);
        targets.push(pair.score);
        
        if i < 5 {
            println!("      S1: {}", pair.sentence1);
            println!("      S2: {}", pair.sentence2);
            println!("      SNN Pred: {:.4} | Target Guru: {:.4}\n", sim, pair.score);
        }
    }
    let dur = t0.elapsed().as_secs_f64();
    let ms_per_pair = dur * 1000.0 / eval_data.len() as f64;
    (pearson(&preds, &targets), ms_per_pair, dur)
}

fn main() {
    let vocab_path = "experiment/file_model/vocab_multilingual.json";
    let weights_path = "experiment/file_model/trained_weights.json";
    let dataset_path = "experiment/file_model/all_sts_teacher_scored.json";

    println!("=============================================================");
    println!(" EVALUASI MODEL TERSIMPAN");
    println!("=============================================================\n");

    println!("Memuat tokenizer dari {}...", vocab_path);
    let tokenizer = BPETokenizer::load(vocab_path);
    let vocab_size = tokenizer.vocab_size();

    println!("Memuat bobot dari {}...", weights_path);
    let weights = load_weights(weights_path);
    let d_model = weights.get("d_model").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
    println!("  -> d_model: {}", d_model);
    
    let mut embedder = new_embedder(tokenizer, vocab_size, d_model, 128);
    apply_weights(&mut embedder, &weights);
    embedder.set_use_attention(false);

    println!("Memuat dataset evaluasi dari {}...", dataset_path);
    let f = File::open(dataset_path).expect("File dataset tidak ditemukan!");
    let eval_data: Vec<STSPair> = serde_json::from_reader(BufReader::new(f)).unwrap();

    println!("Mengevaluasi {} pasang kalimat...", eval_data.len());
    let (p, ms, dur) = evaluate_stsb(&mut embedder, &eval_data);

    println!("\n=============================================================");
    println!(" HASIL EVALUASI");
    println!(" Dataset     : {}", dataset_path);
    println!(" Total Data  : {}", eval_data.len());
    println!(" Pearson     : {:.4}", p);
    println!(" Kecepatan   : {:.2} ms / pair", ms);
    println!(" Total Waktu : {:.2} detik", dur);
    println!("=============================================================");
}

use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use SpikingNetworkRust::models::sentence_embedder::SpikingSentenceEmbedder;
use SpikingNetworkRust::models::sentence_embedder;
use serde::Deserialize;
use serde_json::json;
use std::fs::File;
use std::io::{BufReader, Write};
use std::time::Instant;

#[derive(Deserialize, Clone)]
struct PairScored { s1: String, s2: String, score: f32 }

#[derive(Deserialize)]
struct STSPair { sentence1: String, sentence2: String, score: f32 }

fn load_init(path: &str) -> serde_json::Value {
    let f = File::open(path)
        .unwrap_or_else(|_| panic!("init_weights.json tidak ditemukan!\nJalankan: cargo run --release --bin generate_init_weights"));
    serde_json::from_reader(BufReader::new(f)).unwrap()
}

fn apply_init(embedder: &mut SpikingSentenceEmbedder, init: &serde_json::Value) {
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
    SpikingSentenceEmbedder::new(tokenizer, vocab_size, sentence_embedder::SNNConfig {
        d_model, max_seq_length, learning_rate: 0.01,
        clip_min: -1.0, clip_max: 1.0,
        att_beta_range: (0.8, 0.9), att_threshold_range: (0.1, 0.3),
        bptt_beta_range: (0.8, 0.9), bptt_threshold_range: (0.5, 1.0),
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
    for pair in eval_data {
        let s1 = pair.sentence1.to_lowercase();
        let s2 = pair.sentence2.to_lowercase();
        let embs = embedder.encode(&[s1.as_str(), s2.as_str()]);
        preds.push(cosine_sim(&embs[0], &embs[1]));
        targets.push(pair.score);
    }
    let dur = t0.elapsed().as_secs_f64();
    let ms_per_pair = dur * 1000.0 / eval_data.len() as f64;
    (pearson(&preds, &targets), ms_per_pair, dur)
}

fn train_distil(tokenizer: BPETokenizer, vocab_size: usize, init: &serde_json::Value, d_model: usize, use_init: bool, init_d_model: usize, max_seq_length: usize) -> SpikingSentenceEmbedder {
    let dataset_path = "experiment/file_model/teacher_distillation_dataset.json";
    println!("Memuat dataset dari {}...", dataset_path);
    let f = File::open(dataset_path).expect("teacher_distillation_dataset.json tidak ditemukan");
    let dataset: Vec<PairScored> = serde_json::from_reader(BufReader::new(f)).unwrap();
    let mut embedder = new_embedder(tokenizer, vocab_size, d_model, max_seq_length);
    if use_init {
        println!("  -> Menggunakan inisialisasi terkontrol (d_model={}) dari init_weights.", d_model);
        apply_init(&mut embedder, init);
    } else {
        println!("  -> Melewati inisialisasi terkontrol karena d_model script ({}) berbeda dengan init_weights ({}).", d_model, init_d_model);
    }
    embedder.set_use_attention(true);

    let num_pairs = 32;
    let steps_per_epoch = dataset.len() / num_pairs;
    let num_epochs = 1; // Kembali ke 1 epoch untuk mencegah Catastrophic Forgetting pada SNN
    let total_steps = steps_per_epoch * num_epochs;
    let mut step = 0;
    
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    println!("Mulai pelatihan Distilasi ({} Epoch, total batch: {})...", num_epochs, total_steps);
    
    for epoch in 0..num_epochs {
        let mut data = dataset.clone();
        data.shuffle(&mut rng);
        let mut batch_texts = Vec::new();
        let mut batch_targets = Vec::new();
        
        for pair in &data {
            batch_texts.push(pair.s1.clone()); batch_texts.push(pair.s2.clone());
            batch_targets.push(pair.score);
            if batch_targets.len() == num_pairs {
                // Menggunakan base LR yang sangat kecil (0.001) untuk menjaga kestabilan init_weights
                let base_lr = if use_init { 0.001 } else { 0.01 };
                let lr = base_lr * f32::max(0.01, 1.0 - (step as f32 / total_steps as f32));
                embedder.set_learning_rate(lr);
                
                let texts: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();
                embedder.train_step_distill(&texts, &batch_targets, 0.2);
                
                batch_texts.clear(); batch_targets.clear(); step += 1;
                
                if step % 500 == 0 || step == total_steps {
                    println!("  Epoch {}/{} | Step {}/{} | LR: {:.5}", epoch + 1, num_epochs, step, total_steps, lr);
                }
            }
        }
    }
    embedder
}

fn eval_all_datasets(embedder: &mut SpikingSentenceEmbedder) -> serde_json::Value {
    let datasets = vec![
        ("All-STS-Teacher", "experiment/file_model/all_sts_teacher_scored.json"),
    ];
    let mut results = serde_json::Map::new();
    println!("\n  [Evaluasi Knowledge Distillation]");
    for (name, path) in &datasets {
        let f = match File::open(path) {
            Ok(f) => f,
            Err(_) => { println!("    {name}: file tidak ditemukan, lewati"); continue; }
        };
        let data: Vec<STSPair> = serde_json::from_reader(BufReader::new(f)).unwrap();
        let (r, ms, _) = evaluate_stsb(embedder, &data);
        println!("    {name:<8} | Pearson: {r:.4} | {ms:.2}ms/pair");
        results.insert(name.to_string(), json!({ "pearson": r, "ms_per_pair": ms, "n_pairs": data.len() }));
    }
    let avg_sops = embedder.metrics.total_sops as f64 / embedder.metrics.total_sentences.max(1) as f64;
    let avg_spikes = (embedder.metrics.embedding_spikes + embedder.metrics.attention_spikes + embedder.metrics.pooler_spikes) as f64
        / embedder.metrics.total_sentences.max(1) as f64;
    let transformer_macs = 10_223_616.0_f64;
    results.insert("_energy".to_string(), json!({
        "average_spikes_per_sentence": avg_spikes,
        "snn_sops_per_sentence": avg_sops,
        "transformer_macs_estimated": transformer_macs,
        "energy_savings_ratio": transformer_macs / avg_sops.max(1.0),
    }));
    serde_json::Value::Object(results)
}

fn eval_train_subset(embedder: &mut SpikingSentenceEmbedder) {
    let dataset_path = "experiment/file_model/teacher_distillation_dataset.json";
    if let Ok(f) = File::open(dataset_path) {
        if let Ok(dataset) = serde_json::from_reader::<_, Vec<PairScored>>(BufReader::new(f)) {
            let subset_len = dataset.len() / 4; // Ambil 1/4 saja
            println!("\n  [Evaluasi Overfitting pada 1/4 Data Train]");
            
            let mut preds = Vec::new();
            let mut targets = Vec::new();
            
            for i in 0..subset_len {
                let pair = &dataset[i];
                let s1 = pair.s1.to_lowercase();
                let s2 = pair.s2.to_lowercase();
                let embs = embedder.encode(&[&s1, &s2]);
                preds.push(cosine_sim(&embs[0], &embs[1]));
                targets.push(pair.score);
            }
            
            let r = pearson(&preds, &targets);
            println!("    Train Dataset (1/4) | Pearson: {:.4}", r);
        }
    }
}

fn main() {
    let vocab_path = "experiment/file_model/vocab.json";
    let init_path  = "experiment/file_model/init_weights.json";
    let output_path = "experiment/file_model/train_distil_only_results.json";

    println!("=============================================================");
    println!(" PELATIHAN KNOWLEDGE DISTILLATION SAJA");
    println!("=============================================================\n");

    println!("Memuat tokenizer...");
    let tokenizer = BPETokenizer::load(vocab_path);
    let vocab_size = tokenizer.vocab_size();

    // Ubah nilai d_model dan max_seq_length di sini
    let d_model = 64; 
    let max_seq_length = 64;

    println!("Memuat bobot inisialisasi terkontrol dari {}...", init_path);
    let init = load_init(init_path);
    
    let t_train = Instant::now();
    let init_d_model = init.get("d_model").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let use_init = d_model == init_d_model;
    let mut embedder = train_distil(tokenizer, vocab_size, &init, d_model, use_init, init_d_model, max_seq_length);
    let train_secs = t_train.elapsed().as_secs_f64();
    println!("Pelatihan selesai dalam {train_secs:.1}s\n");

    let mut all_results = serde_json::Map::new();
    let result = eval_all_datasets(&mut embedder);
    
    // Panggil evaluasi data train untuk mengecek overfitting
    eval_train_subset(&mut embedder);
    
    all_results.insert("Knowledge Distillation".to_string(), json!({
        "train_time_seconds": train_secs,
        "results": result
    }));

    let output = json!({
        "controlled_initialization": true,
        "init_weights_file": init_path,
        "evaluation": all_results,
    });
    
    let mut f = File::create(output_path).unwrap();
    f.write_all(serde_json::to_string_pretty(&output).unwrap().as_bytes()).unwrap();
    println!("\n✓ Hasil evaluasi disimpan ke: {}", output_path);
}

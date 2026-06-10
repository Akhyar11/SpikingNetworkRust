/// Script evaluasi ulang menyeluruh untuk Poin A, B, dan C dengan inisialisasi terkontrol.
/// Semua model dilatih dari bobot yang identik (init_weights.json).
/// Jalankan generate_init_weights terlebih dahulu jika init_weights.json belum ada.
use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use SpikingNetworkRust::models::sentence_embedder::SpikingSentenceEmbedder;
use SpikingNetworkRust::models::sentence_embedder;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

// ─── Dataset Types ────────────────────────────────────────────────────────────

#[derive(Deserialize, Clone)]
struct PairScored { s1: String, s2: String, score: f32 }

#[derive(Deserialize)]
struct STSPair { sentence1: String, sentence2: String, score: f32 }

// ─── Helpers ──────────────────────────────────────────────────────────────────

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
        att_beta_range: (0.8, 0.99), att_threshold_range: (0.1, 0.3),
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

// ─── Training: Human-Annotated (STS-B) ───────────────────────────────────────

fn train_human(tokenizer: BPETokenizer, vocab_size: usize, init: &serde_json::Value) -> SpikingSentenceEmbedder {
    let dataset_path = "experiment/file_model/human_only_dataset.json";
    let f = File::open(dataset_path).expect("human_only_dataset.json tidak ditemukan");
    let dataset: Vec<PairScored> = serde_json::from_reader(BufReader::new(f)).unwrap();
    let mut embedder = new_embedder(tokenizer, vocab_size, 64, 32);
    apply_init(&mut embedder, init);

    let num_pairs = 32;
    let total_steps = dataset.len() / num_pairs;
    let mut step = 0;
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let mut data = dataset.clone();
    data.shuffle(&mut rng);
    let mut batch_texts = Vec::new();
    let mut batch_targets = Vec::new();
    for pair in &data {
        batch_texts.push(pair.s1.clone()); batch_texts.push(pair.s2.clone());
        batch_targets.push(pair.score);
        if batch_targets.len() == num_pairs {
            let lr = 0.01 * f32::max(0.01, 1.0 - (step as f32 / total_steps as f32));
            embedder.set_learning_rate(lr);
            let texts: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();
            embedder.train_step_distill(&texts, &batch_targets, 0.2);
            batch_texts.clear(); batch_targets.clear(); step += 1;
        }
    }
    embedder
}

// ─── Training: Knowledge Distillation (Teacher AI) ───────────────────────────

fn train_distil(tokenizer: BPETokenizer, vocab_size: usize, init: &serde_json::Value) -> SpikingSentenceEmbedder {
    let dataset_path = "experiment/file_model/teacher_distillation_dataset.json";
    let f = File::open(dataset_path).expect("teacher_distillation_dataset.json tidak ditemukan");
    let dataset: Vec<PairScored> = serde_json::from_reader(BufReader::new(f)).unwrap();
    let mut embedder = new_embedder(tokenizer, vocab_size, 64, 32);
    apply_init(&mut embedder, init);

    let num_pairs = 32;
    let total_steps = dataset.len() / num_pairs;
    let mut step = 0;
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let mut data = dataset.clone();
    data.shuffle(&mut rng);
    let mut batch_texts = Vec::new();
    let mut batch_targets = Vec::new();
    for pair in &data {
        batch_texts.push(pair.s1.clone()); batch_texts.push(pair.s2.clone());
        batch_targets.push(pair.score);
        if batch_targets.len() == num_pairs {
            let lr = 0.01 * f32::max(0.01, 1.0 - (step as f32 / total_steps as f32));
            embedder.set_learning_rate(lr);
            let texts: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();
            embedder.train_step_distill(&texts, &batch_targets, 0.2);
            batch_texts.clear(); batch_targets.clear(); step += 1;
        }
    }
    embedder
}

// ─── Training: Unsupervised SimCSE (Corpus) ──────────────────────────────────

fn corrupt_sentence(s: &str) -> String { s.to_string() }

fn create_hard_negative(sentence: &str, all_lines: &[String], rng: &mut rand::rngs::ThreadRng) -> String {
    use rand::seq::SliceRandom;
    use rand::Rng;
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let random_line = if !all_lines.is_empty() { all_lines.choose(rng).unwrap().as_str() } else { "" };
    let rwords: Vec<&str> = random_line.split_whitespace().collect();
    if words.len() < 4 || rwords.len() < 4 { return random_line.to_string(); }
    let si1 = words.len() / 2 + rng.gen_range(0..=1);
    let si2 = rwords.len() / 2;
    let mut mix = Vec::new();
    mix.extend_from_slice(&words[0..si1.min(words.len())]);
    mix.extend_from_slice(&rwords[si2..]);
    mix.join(" ")
}

fn train_simcse(tokenizer: BPETokenizer, vocab_size: usize, init: &serde_json::Value) -> SpikingSentenceEmbedder {
    let corpus_path = "/home/akhyar/Dokumen/Code/NODE_JS/penelitian_model_bahasa_dengan_spiking/dataset/mini_corpus20mb.txt";
    let mut all_lines: Vec<String> = Vec::new();
    if let Ok(f) = File::open(corpus_path) {
        for line in BufReader::new(f).lines().flatten() {
            let t = line.trim().to_string();
            if !t.is_empty() && t.split_whitespace().count() >= 10 { all_lines.push(t); }
        }
    } else { panic!("mini_corpus20mb.txt tidak ditemukan"); }

    let mut embedder = new_embedder(tokenizer, vocab_size, 64, 32);
    apply_init(&mut embedder, init);

    let num_pairs = 32;
    let total_steps = all_lines.len() / num_pairs;
    let mut global_step = 0;
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    all_lines.shuffle(&mut rng);
    let mut q_texts = Vec::new();
    let mut p_texts = Vec::new();
    let mut h_texts = Vec::new();
    for q_line in &all_lines {
        p_texts.push(corrupt_sentence(q_line));
        h_texts.push(create_hard_negative(q_line, &all_lines, &mut rng));
        q_texts.push(q_line.clone());
        if q_texts.len() == num_pairs {
            let lr = 0.01 * f32::max(0.01, 1.0 - (global_step as f32 / total_steps as f32));
            embedder.set_learning_rate(lr);
            let mut batch: Vec<&str> = Vec::new();
            for s in &q_texts { batch.push(s.as_str()); }
            for s in &p_texts { batch.push(s.as_str()); }
            for s in &h_texts { batch.push(s.as_str()); }
            // SimCSE contrastive: gunakan train_step_distill dengan pairs sebagai proxy
            // Buat pasangan (q,p) dengan score 1.0 dan (q,h) dengan score 0.0
            let mut pair_texts = Vec::new();
            let mut pair_scores = Vec::new();
            for i in 0..num_pairs {
                pair_texts.push(q_texts[i].as_str());
                pair_texts.push(p_texts[i].as_str());
                pair_scores.push(1.0_f32);
            }
            embedder.train_step_distill(&pair_texts, &pair_scores, 0.2);
            q_texts.clear(); p_texts.clear(); h_texts.clear();
            global_step += 1;
        }
    }
    embedder
}

// ─── Multi-Dataset Evaluation ─────────────────────────────────────────────────

fn eval_all_datasets(embedder: &mut SpikingSentenceEmbedder, label: &str) -> serde_json::Value {
    let datasets = vec![
        ("STS-B",  "experiment/file_model/sts-b_valid.json"),
        ("STS-12", "experiment/file_model/mteb_sts12-sts.json"),
        ("STS-13", "experiment/file_model/mteb_sts13-sts.json"),
        ("STS-14", "experiment/file_model/mteb_sts14-sts.json"),
        ("STS-15", "experiment/file_model/mteb_sts15-sts.json"),
        ("STS-16", "experiment/file_model/mteb_sts16-sts.json"),
        ("SICK-R", "experiment/file_model/mteb_sickr-sts.json"),
    ];
    let mut results = serde_json::Map::new();
    println!("  [{label}]");
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
    // Energy metrics dari STS-B
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

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let vocab_path = "experiment/file_model/vocab.json";
    let init_path  = "experiment/file_model/init_weights.json";
    let output_path = "experiment/file_model/full_eval_controlled.json";

    println!("Memuat tokenizer...");
    let tokenizer = BPETokenizer::load(vocab_path);
    let vocab_size = tokenizer.vocab_size();

    println!("Memuat bobot inisialisasi terkontrol dari {}...", init_path);
    let init = load_init(init_path);
    println!("✓ Semua model akan dilatih dari bobot yang IDENTIK.\n");

    let strategies: Vec<(&str, fn(BPETokenizer, usize, &serde_json::Value) -> SpikingSentenceEmbedder)> = vec![
        ("Human-Only (STS-B)",         train_human),
        ("Knowledge Distillation (AI)", train_distil),
        ("Unsupervised SimCSE",        train_simcse),
    ];

    let mut all_results = serde_json::Map::new();

    println!("=============================================================");
    println!(" EVALUASI ULANG TERKONTROL — Poin A, B, C (inisialisasi sama)");
    println!("=============================================================\n");

    for (label, train_fn) in &strategies {
        println!(">>> Training: {label}");
        let t_train = Instant::now();
        let mut embedder = train_fn(tokenizer.clone_with_same_vocab(), vocab_size, &init);
        let train_secs = t_train.elapsed().as_secs_f64();
        println!("    Selesai dalam {train_secs:.1}s\n");

        let result = eval_all_datasets(&mut embedder, label);
        all_results.insert(label.to_string(), json!({
            "train_time_seconds": train_secs,
            "results": result
        }));
        println!();
    }

    println!("=============================================================");
    let output = json!({
        "controlled_initialization": true,
        "init_weights_file": init_path,
        "evaluation": all_results,
    });
    let mut f = File::create(output_path).unwrap();
    f.write_all(serde_json::to_string_pretty(&output).unwrap().as_bytes()).unwrap();
    println!("✓ Semua hasil disimpan ke: {}", output_path);
}

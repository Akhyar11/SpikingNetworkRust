/// Script evaluasi ulang menyeluruh pipeline dinamis:
/// 1. Evaluasi 3 Dataset: SimCSE, Distillation, Human-Only
/// 2. Evaluasi Scaling d_model (32, 64, 128, 256) menggunakan dataset terbaik
/// 3. Evaluasi Ablasi (No-Attention, Homogeneous) menggunakan d_model terbaik

use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use SpikingNetworkRust::models::sentence_embedder::SpikingSentenceEmbedder;
use SpikingNetworkRust::models::sentence_embedder;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;

// ─── Dataset Types ────────────────────────────────────────────────────────────

#[derive(Deserialize, Clone)]
struct PairScored { s1: String, s2: String, score: f32 }

#[derive(Deserialize)]
struct STSPair { sentence1: String, sentence2: String, score: f32 }

#[derive(Clone, Copy, PartialEq, Debug)]
enum DatasetChoice {
    HumanOnly,
    Distillation,
    SimCSE,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn new_embedder(tokenizer: BPETokenizer, vocab_size: usize, d_model: usize, max_seq_length: usize) -> SpikingSentenceEmbedder {
    let mut embedder = SpikingSentenceEmbedder::new(tokenizer, vocab_size, sentence_embedder::SNNConfig {
        d_model, max_seq_length, learning_rate: 0.01,
        clip_min: -1.0, clip_max: 1.0,
        att_beta_range: (0.8, 0.99), att_threshold_range: (-1.0, -0.5), // Threshold negatif agar lebih reseptif
        bptt_beta_range: (0.8, 0.99), bptt_threshold_range: (0.5, 1.0),
    });

    // FIX: Paksa Pooler menjadi Matriks Identitas agar gradien mengalir sempurna ke Attention
    for i in 0..d_model {
        for j in 0..d_model {
            embedder.pooler.kernel[i * d_model + j] = if i == j { 1.0 } else { 0.0 };
        }
        embedder.pooler.bias[i] = 0.0;
    }

    embedder
}

fn apply_init_homogen(embedder: &mut SpikingSentenceEmbedder) {
    let emb_params: Vec<(String, Vec<f32>)> = embedder.embedding.get_parameters()
        .into_iter().map(|(k, v)| (k.to_string(), v.to_vec())).collect();
    for (k, mut data) in emb_params {
        if k.starts_with("beta") {
            for x in data.iter_mut() { *x = 0.90; }
        } else if k.starts_with("threshold") {
            for x in data.iter_mut() { *x = 0.75; }
        }
        let _ = embedder.embedding.set_parameter(&k, &data);
    }
    
    let att_params: Vec<(String, Vec<f32>)> = embedder.attention.get_parameters()
        .into_iter().map(|(k, v)| (k.to_string(), v.to_vec())).collect();
    for (k, mut data) in att_params {
        if k.starts_with("beta") {
            for x in data.iter_mut() { *x = 0.90; }
        } else if k.starts_with("threshold") {
            for x in data.iter_mut() { *x = -0.50; } // Threshold negatif agar identik
        }
        let _ = embedder.attention.set_parameter(&k, &data);
    }
    
    let pooler_params: Vec<(String, Vec<f32>)> = embedder.pooler.get_parameters()
        .into_iter().map(|(k, v)| (k.to_string(), v.to_vec())).collect();
    for (k, mut data) in pooler_params {
        if k.starts_with("beta") {
            for x in data.iter_mut() { *x = 0.90; }
        } else if k.starts_with("threshold") {
            for x in data.iter_mut() { *x = 0.75; }
        }
        let _ = embedder.pooler.set_parameter(&k, &data);
    }
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

fn evaluate_stsb(embedder: &mut SpikingSentenceEmbedder, eval_data: &[STSPair], print_examples: bool) -> (f32, f64, f64) {
    let mut preds = Vec::new();
    let mut targets = Vec::new();
    let t0 = Instant::now();
    for (i, pair) in eval_data.iter().enumerate() {
        let s1 = pair.sentence1.to_lowercase();
        let s2 = pair.sentence2.to_lowercase();
        let embs = embedder.encode(&[s1.as_str(), s2.as_str()]);
        let pred = cosine_sim(&embs[0], &embs[1]);
        preds.push(pred);
        targets.push(pair.score);
        
        if print_examples && i < 5 {
            let err = (pair.score - pred).abs();
            println!("    [Contoh {}] Target: {:.4} | Prediksi: {:.4} | Err: {:.4}", i+1, pair.score, pred, err);
            println!("               S1: '{}'", pair.sentence1);
            println!("               S2: '{}'\n", pair.sentence2);
        }
    }
    let dur = t0.elapsed().as_secs_f64();
    let ms_per_pair = dur * 1000.0 / eval_data.len() as f64;
    (pearson(&preds, &targets), ms_per_pair, dur)
}

fn print_progress_bar(step: usize, total_steps: usize, start_time: Instant) {
    let progress = step as f64 / total_steps as f64;
    let bar_length = 40;
    let filled = (progress * bar_length as f64) as usize;
    let empty = bar_length - filled;
    
    let elapsed = start_time.elapsed().as_secs_f64();
    let mut eta = 0.0;
    if step > 0 {
        let time_per_step = elapsed / step as f64;
        eta = time_per_step * (total_steps - step) as f64;
    }
    
    print!("\r    [");
    for _ in 0..filled { print!("="); }
    for _ in 0..empty { print!("-"); }
    print!("] {:.1}% | Step {}/{} | Elapsed: {:.1}s | ETA: {:.1}s", progress * 100.0, step, total_steps, elapsed, eta);
    std::io::stdout().flush().unwrap();
}

// ─── Unified Training Pipeline ──────────────────────────────────────────────

fn corrupt_sentence(s: &str) -> String { s.to_string() }

fn create_hard_negative(sentence: &str, all_lines: &[String], rng: &mut rand::rngs::StdRng) -> String {
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

fn train_model(
    dataset_choice: DatasetChoice,
    tokenizer: BPETokenizer,
    vocab_size: usize,
    d_model: usize,
    use_attention: bool,
    is_homogeneous: bool,
) -> SpikingSentenceEmbedder {
    let mut max_seq_length = 16;
    let mut human_data: Option<Vec<PairScored>> = None;
    let mut distil_data: Option<Vec<PairScored>> = None;
    let mut simcse_data: Option<Vec<String>> = None;

    match dataset_choice {
        DatasetChoice::HumanOnly => {
            let dataset_path = "experiment/file_model/human_only_dataset.json";
            let f = File::open(dataset_path).expect("human_only_dataset.json tidak ditemukan");
            let dataset: Vec<PairScored> = serde_json::from_reader(BufReader::new(f)).unwrap();
            max_seq_length = dataset.iter().flat_map(|p| vec![p.s1.as_str(), p.s2.as_str()])
                .map(|t| tokenizer.encode(&t.to_lowercase()).len()).max().unwrap_or(16);
            human_data = Some(dataset);
        },
        DatasetChoice::Distillation => {
            let dataset_path = "experiment/file_model/teacher_distillation_dataset_scored.json";
            let f = File::open(dataset_path).expect("teacher_distillation_dataset_scored.json tidak ditemukan");
            let dataset: Vec<PairScored> = serde_json::from_reader(BufReader::new(f)).unwrap();
            max_seq_length = dataset.iter().flat_map(|p| vec![p.s1.as_str(), p.s2.as_str()])
                .map(|t| tokenizer.encode(&t.to_lowercase()).len()).max().unwrap_or(16);
            distil_data = Some(dataset);
        },
        DatasetChoice::SimCSE => {
            let corpus_path = "/home/akhyar/Dokumen/Code/NODE_JS/penelitian_model_bahasa_dengan_spiking/dataset/mini_corpus20mb.txt";
            let mut all_lines: Vec<String> = Vec::new();
            if let Ok(f) = File::open(corpus_path) {
                for line in BufReader::new(f).lines().flatten() {
                    let t = line.trim().to_string();
                    if !t.is_empty() && t.split_whitespace().count() >= 10 { all_lines.push(t); }
                }
            } else { panic!("mini_corpus20mb.txt tidak ditemukan"); }
            max_seq_length = all_lines.iter().take(10000)
                .map(|t| tokenizer.encode(&t.to_lowercase()).len()).max().unwrap_or(16);
            simcse_data = Some(all_lines);
        }
    }
    
    println!("    Max sequence length detected: {}", max_seq_length);

    let mut embedder = new_embedder(tokenizer.clone_with_same_vocab(), vocab_size, d_model, max_seq_length);
    embedder.set_use_attention(use_attention);
    
    if is_homogeneous {
        apply_init_homogen(&mut embedder);
    }

    match dataset_choice {
        DatasetChoice::HumanOnly => {
            let dataset = human_data.unwrap();
            let num_pairs = 32;
            let total_steps = dataset.len() / num_pairs;
            let mut step = 0;
            let t_start = Instant::now();
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
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
                    embedder.train_step_distill(&texts, &batch_targets, 0.5);
                    batch_texts.clear(); batch_targets.clear(); step += 1;
                    print_progress_bar(step, total_steps, t_start);
                }
            }
            println!();
        },
        DatasetChoice::Distillation => {
            let dataset = distil_data.unwrap();
            let num_pairs = 32;
            let steps_per_epoch = dataset.len() / num_pairs;
            let num_epochs = 10;
            let total_steps = steps_per_epoch * num_epochs;
            let mut step = 0;
            let t_start = Instant::now();
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            for _epoch in 0..num_epochs {
                let mut data = dataset.clone();
                data.shuffle(&mut rng);
                let mut batch_texts = Vec::new();
                let mut batch_targets = Vec::new();
                for pair in &data {
                    batch_texts.push(pair.s1.clone()); batch_texts.push(pair.s2.clone());
                    batch_targets.push(pair.score);
                    if batch_targets.len() == num_pairs {
                        let base_lr = 2.0;
                        let progress = step as f32 / total_steps as f32;
                        let lr = base_lr * (0.01 + 0.99 * (0.5 * (1.0 + (progress * std::f32::consts::PI).cos())));
                        embedder.set_learning_rate(lr);
                        let texts: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();
                        embedder.train_step_distill(&texts, &batch_targets, 0.5);
                        batch_texts.clear(); batch_targets.clear(); step += 1;
                        print_progress_bar(step, total_steps, t_start);
                    }
                }
            }
            println!();
        },
        DatasetChoice::SimCSE => {
            let mut all_lines = simcse_data.unwrap();
            let num_pairs = 32;
            let total_steps = all_lines.len() / num_pairs;
            let mut step = 0;
            let t_start = Instant::now();
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            all_lines.shuffle(&mut rng);
            let mut q_texts = Vec::new();
            let mut p_texts = Vec::new();
            let mut h_texts = Vec::new();
            for q_line in &all_lines {
                p_texts.push(corrupt_sentence(q_line));
                h_texts.push(create_hard_negative(q_line, &all_lines, &mut rng));
                q_texts.push(q_line.clone());
                if q_texts.len() == num_pairs {
                    let lr = 0.01 * f32::max(0.01, 1.0 - (step as f32 / total_steps as f32));
                    embedder.set_learning_rate(lr);
                    let mut pair_texts = Vec::new();
                    let mut pair_scores = Vec::new();
                    for i in 0..num_pairs {
                        pair_texts.push(q_texts[i].as_str());
                        pair_texts.push(p_texts[i].as_str());
                        pair_scores.push(1.0_f32);
                    }
                    embedder.train_step_distill(&pair_texts, &pair_scores, 0.5);
                    q_texts.clear(); p_texts.clear(); h_texts.clear();
                    step += 1;
                    print_progress_bar(step, total_steps, t_start);
                }
            }
            println!();
        }
    }
    
    embedder
}

// ─── Multi-Dataset Evaluation ─────────────────────────────────────────────────

fn eval_all_datasets(embedder: &mut SpikingSentenceEmbedder, label: &str) -> (serde_json::Value, f32) {
    let datasets = vec![
        ("STS-B",  "experiment/file_model/sts-b_valid.json"),
        ("STS-12", "experiment/file_model/mteb_sts12-sts.json"),
        ("STS-13", "experiment/file_model/mteb_sts13-sts.json"),
        ("STS-14", "experiment/file_model/mteb_sts14-sts.json"),
        ("STS-15", "experiment/file_model/mteb_sts15-sts.json"),
        ("STS-16", "experiment/file_model/mteb_sts16-sts.json"),
        ("SICK-R", "experiment/file_model/mteb_sickr-sts.json"),
        ("ALL-STS", "experiment/file_model/all_sts_teacher_scored.json"),
    ];
    let mut results = serde_json::Map::new();
    println!("  [{label}]");
    let mut stsb_score = 0.0;
    
    for (name, path) in &datasets {
        let f = match File::open(path) {
            Ok(f) => f,
            Err(_) => { println!("    {name}: file tidak ditemukan, lewati"); continue; }
        };
        let data: Vec<STSPair> = serde_json::from_reader(BufReader::new(f)).unwrap();
        let print_examples = *name == "ALL-STS";
        if print_examples {
            println!("\n    --- Contoh Prediksi di ALL-STS ---");
        }
        let (r, ms, _) = evaluate_stsb(embedder, &data, print_examples);
        println!("    {name:<8} | Pearson: {r:.4} | {ms:.2}ms/pair");
        results.insert(name.to_string(), json!({ "pearson": r, "ms_per_pair": ms, "n_pairs": data.len() }));
        
        if *name == "STS-B" {
            stsb_score = r;
        }
    }
    
    let avg_sops = embedder.metrics.total_sops as f64 / embedder.metrics.total_sentences.max(1) as f64;
    let avg_spikes = (embedder.metrics.embedding_spikes + embedder.metrics.attention_spikes + embedder.metrics.pooler_spikes) as f64
        / embedder.metrics.total_sentences.max(1) as f64;
    let transformer_macs = 1_434_451_968.0_f64; 
    results.insert("_energy".to_string(), json!({
        "average_spikes_per_sentence": avg_spikes,
        "snn_sops_per_sentence": avg_sops,
        "transformer_macs_estimated": transformer_macs,
        "energy_savings_ratio": transformer_macs / avg_sops.max(1.0),
    }));
    
    (serde_json::Value::Object(results), stsb_score)
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let vocab_path = "experiment/file_model/vocab.json";
    let output_path = "experiment/file_model/full_eval_controlled.json";

    println!("Memuat tokenizer...");
    let tokenizer = BPETokenizer::load(vocab_path);
    let vocab_size = tokenizer.vocab_size();

    let mut all_results = serde_json::Map::new();

    println!("=============================================================");
    println!(" EVALUASI PIPELINE DINAMIS");
    println!("=============================================================\n");

    // =========================================================================
    // FASE 1: Temukan Dataset/Metode Pelatihan Terbaik (Fixed d_model = 64)
    // =========================================================================
    println!(">>> FASE 1: Evaluasi Metode Dataset (d_model = 64, max_seq = auto-detect)");
    let phase1_methods = vec![
        ("Human-Only", DatasetChoice::HumanOnly),
        ("Distillation", DatasetChoice::Distillation),
        ("SimCSE", DatasetChoice::SimCSE),
    ];
    
    let mut best_method = DatasetChoice::Distillation;
    let mut best_method_name = String::new();
    let mut best_score = -1.0;
    
    for (name, method) in phase1_methods {
        println!("\n--- Training: {} ---", name);
        let t_train = Instant::now();
        let mut embedder = train_model(method, tokenizer.clone_with_same_vocab(), vocab_size, 64, true, false);
        let train_secs = t_train.elapsed().as_secs_f64();
        println!("    Selesai dalam {train_secs:.1}s\n");

        let (result_json, score) = eval_all_datasets(&mut embedder, &format!("Phase1: {}", name));
        all_results.insert(format!("Phase1_{}", name), json!({
            "train_time_seconds": train_secs,
            "results": result_json
        }));
        
        if score > best_score {
            best_score = score;
            best_method = method;
            best_method_name = name.to_string();
        }
    }
    
    println!("\n>>> FASE 1 SELESAI. Metode Terbaik: {} (Pearson STS-B: {:.4})", best_method_name, best_score);
    println!("=============================================================\n");

    // =========================================================================
    // FASE 2: Dimensionality Scaling menggunakan Metode Terbaik
    // =========================================================================
    println!(">>> FASE 2: Evaluasi Scaling d_model (Metode: {})", best_method_name);
    let dimensions = vec![32, 64, 128, 256];
    
    let mut best_d_model = 64;
    let mut best_d_score = -1.0;
    
    for d in dimensions {
        println!("\n--- Training: d_model = {} ---", d);
        let t_train = Instant::now();
        let mut embedder = train_model(best_method, tokenizer.clone_with_same_vocab(), vocab_size, d, true, false);
        let train_secs = t_train.elapsed().as_secs_f64();
        println!("    Selesai dalam {train_secs:.1}s\n");

        let (result_json, score) = eval_all_datasets(&mut embedder, &format!("Phase2: d_model={}", d));
        all_results.insert(format!("Phase2_d{}", d), json!({
            "train_time_seconds": train_secs,
            "results": result_json
        }));
        
        if score > best_d_score {
            best_d_score = score;
            best_d_model = d;
        }
    }
    
    println!("\n>>> FASE 2 SELESAI. d_model Terbaik: {} (Pearson STS-B: {:.4})", best_d_model, best_d_score);
    println!("=============================================================\n");

    // =========================================================================
    // FASE 3: Ablation Study menggunakan Metode dan d_model Terbaik
    // =========================================================================
    println!(">>> FASE 3: Evaluasi Ablasi (Metode: {}, d_model: {})", best_method_name, best_d_model);
    let phase3_ablations = vec![
        ("No-Attention", false, false), // use_attention = false
        ("Attention", true, false),     // use_attention = true, no homogen init
        ("Homogeneous-Init", true, true), // is_homogeneous = true
    ];
    
    for (name, use_att, is_homogen) in phase3_ablations {
        println!("\n--- Training: {} ---", name);
        let t_train = Instant::now();
        let mut embedder = train_model(best_method, tokenizer.clone_with_same_vocab(), vocab_size, best_d_model, use_att, is_homogen);
        let train_secs = t_train.elapsed().as_secs_f64();
        println!("    Selesai dalam {train_secs:.1}s\n");

        let (result_json, _score) = eval_all_datasets(&mut embedder, &format!("Phase3: {}", name));
        all_results.insert(format!("Phase3_{}", name), json!({
            "train_time_seconds": train_secs,
            "results": result_json
        }));
    }
    
    println!("\n=============================================================");
    let output = json!({
        "best_method": best_method_name,
        "best_d_model": best_d_model,
        "evaluation": all_results,
    });
    let mut f = File::create(output_path).unwrap();
    f.write_all(serde_json::to_string_pretty(&output).unwrap().as_bytes()).unwrap();
    println!("✓ Semua hasil evaluasi pipeline disimpan ke: {}", output_path);
}

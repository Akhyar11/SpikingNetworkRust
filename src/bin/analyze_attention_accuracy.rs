use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::models::sentence_embedder::{SpikingSentenceEmbedder, SNNConfig};
use std::time::Instant;
use std::fs::File;
use std::io::{BufReader, Write};
use serde::Deserialize;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Deserialize, Clone)]
struct PairScored {
    s1: String,
    s2: String,
    score: f32,
}

#[derive(Deserialize)]
struct STSPair {
    sentence1: String,
    sentence2: String,
    score: f32,
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

fn main() {
    let d_model = 128; // Cukup 128 untuk keseimbangan kecepatan dan akurasi representasi
    let num_epochs = 10;
    
    println!("==========================================================================");
    println!(" ANALISA AKURASI ABSOLUT: ATTENTION vs NO-ATTENTION (Error Tolerance: 0.05)");
    println!("==========================================================================\n");

    let train_dataset_path = "experiment/file_model/teacher_distillation_dataset_scored.json";
    let f_train = File::open(train_dataset_path).expect("dataset training tidak ditemukan");
    let mut train_data: Vec<PairScored> = serde_json::from_reader(BufReader::new(f_train)).unwrap();
    
    // Seed untuk reproduksibilitas pengacakan
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    train_data.shuffle(&mut rng);
    
    let test_dataset_path = "experiment/file_model/all_sts_teacher_scored.json";
    let f_test = File::open(test_dataset_path).expect("dataset testing tidak ditemukan");
    let test_data_raw: Vec<STSPair> = serde_json::from_reader(BufReader::new(f_test)).unwrap();
    
    let test_data: Vec<PairScored> = test_data_raw.into_iter().map(|p| PairScored {
        s1: p.sentence1,
        s2: p.sentence2,
        score: p.score,
    }).collect();

    let tokenizer = BPETokenizer::load("experiment/file_model/vocab_multilingual.json");
    let vocab_size = tokenizer.vocab_size();

    let max_seq_length = train_data.iter().flat_map(|p| vec![p.s1.as_str(), p.s2.as_str()])
        .map(|t| tokenizer.encode(&t.to_lowercase()).len()).max().unwrap_or(16);
    println!("Max sequence length detected: {}", max_seq_length);

    // Setup Config SNN Distilasi
    let config = SNNConfig {
        d_model, max_seq_length, learning_rate: 2.0, 
        clip_min: -1.0, clip_max: 1.0,
        att_beta_range: (0.8, 0.99), att_threshold_range: (-1.0, -0.5),
        bptt_beta_range: (0.8, 0.99), bptt_threshold_range: (0.5, 1.0),
    };

    // 1. Inisialisasi model NO-ATTENTION
    let mut embedder_no_att = SpikingSentenceEmbedder::new(tokenizer.clone_with_same_vocab(), vocab_size, config.clone());
    embedder_no_att.set_use_attention(false);
    for i in 0..d_model {
        for j in 0..d_model { embedder_no_att.pooler.kernel[i * d_model + j] = if i == j { 1.0 } else { 0.0 }; }
        embedder_no_att.pooler.bias[i] = 0.0;
    }

    // 2. Inisialisasi model ATTENTION (Copy Bobot Awal persis sama!)
    let mut embedder_att = SpikingSentenceEmbedder::new(tokenizer.clone_with_same_vocab(), vocab_size, config.clone());
    embedder_att.set_use_attention(true);
    for i in 0..d_model {
        for j in 0..d_model { embedder_att.pooler.kernel[i * d_model + j] = if i == j { 1.0 } else { 0.0 }; }
        embedder_att.pooler.bias[i] = 0.0;
    }
    embedder_att.embedding.weights = embedder_no_att.embedding.weights.clone();
    embedder_att.embedding.beta = embedder_no_att.embedding.beta.clone();
    embedder_att.embedding.threshold = embedder_no_att.embedding.threshold.clone();

    // Fungsi helper untuk training & eval
    let run_experiment = |name: &str, embedder: &mut SpikingSentenceEmbedder| {
        println!("\n==============================================================");
        println!(" MENGUJI MODEL: {}", name);
        println!("==============================================================");
        
        let num_pairs = 32;
        let steps_per_epoch = train_data.len() / num_pairs;
        let total_steps = steps_per_epoch * num_epochs;
        let mut step = 0;
        
        // Memastikan urutan batch SAMA PERSIS untuk kedua model
        let mut train_rng = rand::rngs::StdRng::seed_from_u64(123);
        let t_start_global = Instant::now();
        
        for epoch in 1..=num_epochs {
            let t_start = Instant::now();
            let mut epoch_data = train_data.clone();
            epoch_data.shuffle(&mut train_rng);
            
            let mut batch_texts = Vec::new();
            let mut batch_targets = Vec::new();
            for pair in &epoch_data {
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
                    
                    if step % 10 == 0 || step == total_steps {
                        print_progress_bar(step, total_steps, t_start_global);
                    }
                }
            }
            println!(); // Baris baru setelah progress bar selesai 1 epoch
            
            if epoch == num_epochs || epoch % 5 == 0 {
                // EVALUASI AKURASI
                let mut test_preds = Vec::new();
                let mut test_targets = Vec::new();
                let mut correct_predictions = 0;
                let mut total_error = 0.0;
                
                // Print some examples at the last epoch
                let mut printed_examples = 0;

                for test_chunk in test_data.chunks(32) {
                    let mut eval_texts = Vec::new();
                    let mut eval_strings = Vec::new(); // Menyimpan String untuk meminjamkan &str
                    for p in test_chunk {
                        eval_strings.push(p.s1.to_lowercase());
                        eval_strings.push(p.s2.to_lowercase());
                    }
                    for s in &eval_strings {
                        eval_texts.push(s.as_str());
                    }
                    let embs = embedder.encode(&eval_texts);
                    
                    for i in 0..test_chunk.len() {
                        let pred = cosine_sim(&embs[i*2], &embs[i*2+1]);
                        let target = test_chunk[i].score;
                        let err = (target - pred).abs();
                        
                        test_preds.push(pred);
                        test_targets.push(target);
                        total_error += err;
                        
                        if err <= 0.05 {
                            correct_predictions += 1;
                        }
                        
                        if epoch == num_epochs && printed_examples < 3 {
                            println!("   [Contoh] Target: {:.4} | Prediksi: {:.4} | Error: {:.4}", target, pred, err);
                            printed_examples += 1;
                        }
                    }
                }
                
                let accuracy = (correct_predictions as f32 / test_data.len() as f32) * 100.0;
                let pearson_score = pearson(&test_preds, &test_targets);
                let avg_error = total_error / test_data.len() as f32;
                
                println!("   [EPOCH {}] Time: {}ms | Pearson: {:.4} | Avg Error: {:.4} | Akurasi (Err < 0.05): {:.2}% ({} dari {})", 
                    epoch, t_start.elapsed().as_millis(), pearson_score, avg_error, accuracy, correct_predictions, test_data.len());
            }
        }
    };

    run_experiment("NO-ATTENTION (Baseline)", &mut embedder_no_att);
    run_experiment("ATTENTION (Proposed)", &mut embedder_att);
}

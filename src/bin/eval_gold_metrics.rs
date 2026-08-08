// Evaluasi metrik terhadap GOLD LABEL manusia (bukan skor teacher).
// Menghitung Pearson + Spearman pada 7 benchmark STS standar (MTEB format),
// untuk dua konfigurasi arsitektur: Recurrent Pooler (attention OFF) dan
// Spike-Overlap Attention (attention ON), memakai bobot terlatih yang sama.
//
// Output: experiment/file_model/gold_metrics_snn.json
//
// Cara jalan: cargo run --release --bin eval_gold_metrics

use SpikingNetworkRust::core::bpe::BPETokenizer;
use SpikingNetworkRust::layers::base::Layer;
use SpikingNetworkRust::models::sentence_embedder::{SNNConfig, SpikingSentenceEmbedder};
use serde::Deserialize;
use serde_json::json;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

#[derive(Deserialize)]
struct STSPair {
    #[serde(alias = "s1")]
    sentence1: String,
    #[serde(alias = "s2")]
    sentence2: String,
    score: f32,
}

const DATASETS: [(&str, &str); 7] = [
    ("STS-12", "experiment/file_model/mteb_sts12-sts.json"),
    ("STS-13", "experiment/file_model/mteb_sts13-sts.json"),
    ("STS-14", "experiment/file_model/mteb_sts14-sts.json"),
    ("STS-15", "experiment/file_model/mteb_sts15-sts.json"),
    ("STS-16", "experiment/file_model/mteb_sts16-sts.json"),
    ("STS-B", "experiment/file_model/sts-b_valid.json"),
    ("SICK-R", "experiment/file_model/mteb_sickr-sts.json"),
];

fn load_weights(path: &str) -> serde_json::Value {
    let f = File::open(path).unwrap_or_else(|_| panic!("{} tidak ditemukan!", path));
    serde_json::from_reader(BufReader::new(f)).unwrap()
}

fn apply_weights(embedder: &mut SpikingSentenceEmbedder, init: &serde_json::Value) {
    for group in &["embedding", "attention", "pooler"] {
        let layer: &mut dyn Layer = match *group {
            "embedding" => &mut embedder.embedding,
            "attention" => &mut embedder.attention,
            _ => &mut embedder.pooler,
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

/// Cosine similarity pada vektor yang di-mean-center, TANPA clamp.
/// (encode() sudah melakukan z-score; mean-centering di sini redundan tapi
/// dipertahankan agar identik dengan pipeline evaluasi paper.)
fn cosine_sim_raw(a: &[f32], b: &[f32]) -> f32 {
    let (mut ma, mut mb) = (0.0_f32, 0.0_f32);
    for i in 0..a.len() { ma += a[i]; mb += b[i]; }
    ma /= a.len() as f32; mb /= b.len() as f32;
    let (mut dot, mut na, mut nb) = (0.0_f32, 0.0_f32, 0.0_f32);
    for i in 0..a.len() {
        let (x, y) = (a[i] - ma, b[i] - mb);
        dot += x * y; na += x * x; nb += y * y;
    }
    if na == 0.0 || nb == 0.0 { return 0.0; }
    dot / (na.sqrt() * nb.sqrt())
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n == 0.0 { return 0.0; }
    let (sx, sy): (f64, f64) = (x.iter().sum(), y.iter().sum());
    let (sxx, syy, sxy): (f64, f64, f64) = (
        x.iter().map(|&v| v * v).sum(),
        y.iter().map(|&v| v * v).sum(),
        x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum(),
    );
    let num = n * sxy - sx * sy;
    let den = ((n * sxx - sx * sx).max(0.0) * (n * syy - sy * sy).max(0.0)).sqrt();
    if den == 0.0 { 0.0 } else { num / den }
}

/// Ranking dengan average-tie handling (standar untuk Spearman).
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0_f64; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] { j += 1; }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j { r[idx[k]] = avg; }
        i = j + 1;
    }
    r
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

fn load_dataset(path: &str) -> Vec<STSPair> {
    let f = File::open(path).unwrap_or_else(|_| panic!("{} tidak ditemukan!", path));
    serde_json::from_reader(BufReader::new(f)).unwrap()
}

fn evaluate_dataset(embedder: &mut SpikingSentenceEmbedder, data: &[STSPair]) -> (Vec<f64>, Vec<f64>, f64) {
    let mut preds = Vec::with_capacity(data.len());
    let mut targets = Vec::with_capacity(data.len());
    let t0 = Instant::now();
    for pair in data {
        let s1 = pair.sentence1.to_lowercase();
        let s2 = pair.sentence2.to_lowercase();
        let embs = embedder.encode(&[s1.as_str(), s2.as_str()]);
        preds.push(cosine_sim_raw(&embs[0], &embs[1]) as f64);
        targets.push(pair.score as f64);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / data.len() as f64;
    (preds, targets, ms)
}

fn run_config(vocab_path: &str, weights: &serde_json::Value, use_attention: bool) -> serde_json::Value {    let tokenizer = BPETokenizer::load(vocab_path);
    let vocab_size = tokenizer.vocab_size();
    let d_model = weights.get("d_model").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
    let mut embedder = new_embedder(tokenizer, vocab_size, d_model, 128);
    apply_weights(&mut embedder, weights);
    embedder.set_use_attention(use_attention);

    let mut per_dataset = serde_json::Map::new();
    let mut all_preds: Vec<f64> = Vec::new();
    let mut all_targets: Vec<f64> = Vec::new();
    let mut total_n = 0usize;
    let mut w_pearson = 0.0_f64;
    let mut w_spearman = 0.0_f64;

    for (name, path) in DATASETS {
        let data = load_dataset(path);
        let (preds, targets, ms) = evaluate_dataset(&mut embedder, &data);
        let p_raw = pearson(&preds, &targets);
        let clamped: Vec<f64> = preds.iter().map(|&v| v.max(0.0)).collect();
        let p_clamped = pearson(&clamped, &targets);
        let rho = spearman(&preds, &targets);
        println!(
            "  {:<8} n={:<6} Pearson(raw)={:.4}  Pearson(clamped)={:.4}  Spearman={:.4}  {:.3} ms/pair",
            name, data.len(), p_raw, p_clamped, rho, ms
        );
        w_pearson += p_raw * data.len() as f64;
        w_spearman += rho * data.len() as f64;
        total_n += data.len();
        all_preds.extend_from_slice(&preds);
        all_targets.extend_from_slice(&targets);
        per_dataset.insert(name.to_string(), json!({
            "n_pairs": data.len(),
            "pearson_raw": (p_raw * 10000.0).round() / 10000.0,
            "pearson_clamped": (p_clamped * 10000.0).round() / 10000.0,
            "spearman": (rho * 10000.0).round() / 10000.0,
            "ms_per_pair": (ms * 1000.0).round() / 1000.0,
        }));
    }

    let n_f = total_n as f64;
    json!({
        "use_attention": use_attention,
        "datasets": per_dataset,
        "all_sts_weighted": {
            "total_pairs": total_n,
            "pearson_raw": ((w_pearson / n_f) * 10000.0).round() / 10000.0,
            "spearman": ((w_spearman / n_f) * 10000.0).round() / 10000.0,
            "pooled_pearson": (pearson(&all_preds, &all_targets) * 10000.0).round() / 10000.0,
            "pooled_spearman": (spearman(&all_preds, &all_targets) * 10000.0).round() / 10000.0,
        }
    })
}

fn main() {
    // Pemakaian: eval_gold_metrics [vocab_path] [weights_path] [out_path]
    // Default: model multilingual (artefak Jun-24).
    let args: Vec<String> = std::env::args().collect();
    let vocab_path = args.get(1).map(String::as_str).unwrap_or("experiment/file_model/vocab_multilingual.json");
    let weights_path = args.get(2).map(String::as_str).unwrap_or("experiment/file_model/trained_weights.json");
    let out_path = args.get(3).map(String::as_str).unwrap_or("experiment/file_model/gold_metrics_snn.json");

    println!("=============================================================");
    println!(" EVALUASI GOLD-LABEL (Pearson + Spearman vs Human Gold)");
    println!(" vocab  : {}", vocab_path);
    println!(" weights: {}", weights_path);
    println!("=============================================================\n");

    let weights = load_weights(weights_path);

    // Deteksi kernel attention all-zero (model attention-free): lewati konfigurasi attention.
    let attention_trained = weights
        .get("attention").and_then(|a| a.get("kernel_q")).and_then(|k| k.as_array())
        .map(|k| k.iter().filter_map(|v| v.as_f64()).map(|x| x.abs()).sum::<f64>() > 0.0)
        .unwrap_or(false);

    println!("[1/{}] Konfigurasi Recurrent Pooler (attention OFF)", if attention_trained { 2 } else { 1 });
    let recurrent = run_config(vocab_path, &weights, false);

    let attention = if attention_trained {
        println!("\n[2/2] Konfigurasi Spike-Overlap Attention (attention ON)");
        run_config(vocab_path, &weights, true)
    } else {
        println!("\nKernel attention all-zero -> konfigurasi attention dilewati (model memang attention-free).");
        serde_json::Value::Null
    };

    let out = json!({
        "weights": weights_path,
        "note": "Pearson/Spearman vs human gold labels; lowercase input; cosine pada embedding z-score. pearson_clamped = pipeline paper (sim.max(0)).",
        "recurrent": recurrent,
        "attention": attention,
    });
    std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("\nHasil disimpan ke {}", out_path);
}

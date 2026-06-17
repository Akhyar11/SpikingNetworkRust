"""
Baseline Non-SNN Comparison: GloVe, Word2Vec, TF-IDF
Evaluasi pada STS-B validation set (1.500 pasangan).
Digunakan sebagai *lower baseline* pada Table 1 paper.

Jalankan dari root repo:
    python3 experiment/baseline_nonSNN.py
"""

import json
import re
import math
import time
import numpy as np
from scipy.stats import pearsonr
from collections import Counter

DATASETS = {
    "STS-B": "experiment/file_model/sts-b_valid.json",
    "All-STS-Teacher": "experiment/file_model/all_sts_teacher_scored.json"
}
OUT_PATH  = "experiment/file_model/baseline_results.json"

# ─── Utilities ────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()

def cosine_sim_centered(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson-style cosine (zero-mean sebelum dot product)."""
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def pearson(preds: list, targets: list) -> float:
    r, _ = pearsonr(preds, targets)
    return round(float(r), 4)

def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)

def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ─── Baseline 1: Random (floor) ───────────────────────────────────────────────

def baseline_random(data: list[dict]) -> dict:
    print_header("BASELINE: Random Vectors")
    np.random.seed(42)
    t0 = time.time()
    preds, targets = [], []
    for item in data:
        v1 = np.random.randn(300)
        v2 = np.random.randn(300)
        preds.append(cosine_sim_centered(v1, v2))
        targets.append(item["score"])
    elapsed = time.time() - t0
    r = pearson(preds, targets)
    ms = elapsed * 1000 / len(data)
    print(f"  Pearson r : {r:.4f}")
    print(f"  ms/pair   : {ms:.3f}")
    return {"pearson": r, "ms_per_pair": round(ms, 3), "note": "floor baseline"}

# ─── Baseline 2: TF-IDF Cosine (no external deps) ────────────────────────────

def baseline_tfidf(data: list[dict]) -> dict:
    print_header("BASELINE: TF-IDF Cosine")
    # Buat corpus dari semua kalimat
    all_sents = [item["sentence1"] for item in data] + [item["sentence2"] for item in data]
    all_tokens = [tokenize(s) for s in all_sents]

    # Hitung IDF
    N = len(all_tokens)
    df: Counter = Counter()
    for tokens in all_tokens:
        for t in set(tokens):
            df[t] += 1
    idf = {t: math.log(N / (1 + c)) for t, c in df.items()}
    vocab = sorted(idf.keys())
    v2i = {t: i for i, t in enumerate(vocab)}

    def tfidf_vec(tokens: list[str]) -> np.ndarray:
        tf = Counter(tokens)
        v = np.zeros(len(vocab))
        for t, c in tf.items():
            if t in v2i:
                v[v2i[t]] = (1 + math.log(c)) * idf.get(t, 0)
        return v

    t0 = time.time()
    preds, targets = [], []
    for item in data:
        t1 = tokenize(item["sentence1"])
        t2 = tokenize(item["sentence2"])
        v1 = tfidf_vec(t1)
        v2 = tfidf_vec(t2)
        s = cosine_sim_centered(v1, v2) if (v1.any() and v2.any()) else 0.0
        preds.append(s)
        targets.append(item["score"])
    elapsed = time.time() - t0
    r = pearson(preds, targets)
    ms = elapsed * 1000 / len(data)
    print(f"  Pearson r : {r:.4f}")
    print(f"  ms/pair   : {ms:.3f}")
    return {"pearson": r, "ms_per_pair": round(ms, 3), "vocab_size": len(vocab)}

# ─── Baseline 3 & 4: GloVe & Word2Vec via gensim ─────────────────────────────

def mean_embed(tokens: list[str], model) -> np.ndarray:
    vecs = [model[t] for t in tokens if t in model]
    if not vecs:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

def run_gensim_baseline(name: str, model_key: str, datasets_data: dict) -> dict:
    print_header(f"BASELINE: {name}  [{model_key}]")
    try:
        import gensim.downloader as api
        print(f"  Mengunduh / memuat {model_key}...")
        t_load = time.time()
        model = api.load(model_key)
        load_sec = time.time() - t_load
        print(f"  Load selesai dalam {load_sec:.1f}s  |  dim={model.vector_size}  |  vocab={len(model)}")

        results_dict = {}
        for ds_name, data in datasets_data.items():
            print(f"  Evaluasi pada {ds_name} ({len(data)} pasang)...")
            t0 = time.time()
            preds, targets = [], []
            for item in data:
                t1 = tokenize(item["sentence1"])
                t2 = tokenize(item["sentence2"])
                v1 = mean_embed(t1, model)
                v2 = mean_embed(t2, model)
                preds.append(cosine_sim_centered(v1, v2))
                targets.append(item["score"])
            elapsed = time.time() - t0
            r = pearson(preds, targets)
            ms = elapsed * 1000 / len(data)
            print(f"    Pearson r : {r:.4f} | ms/pair: {ms:.3f}")
            results_dict[ds_name] = {"pearson": r, "ms_per_pair": round(ms, 3)}
        
        return {
            "results": results_dict,
            "dim": model.vector_size,
            "vocab_size": len(model),
            "model_key": model_key,
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    datasets_data = {}
    for name, path in DATASETS.items():
        try:
            d = load_dataset(path)
            datasets_data[name] = d
            print(f"Dataset {name}: {len(d)} pasangan kalimat")
        except Exception as e:
            print(f"Gagal memuat {name}: {e}")

    results = {}

    # TF-IDF dan Random saya lewati dulu agar output rapi, kita fokus pada GloVe dan Word2Vec.
    # GloVe 100d
    results["GloVe-100d"]    = run_gensim_baseline(
        "GloVe Wikipedia+Gigaword 100d", "glove-wiki-gigaword-100", datasets_data)

    # GloVe 300d
    results["GloVe-300d"]    = run_gensim_baseline(
        "GloVe Wikipedia+Gigaword 300d", "glove-wiki-gigaword-300", datasets_data)

    # Word2Vec Google News 300d
    results["Word2Vec-300d"] = run_gensim_baseline(
        "Word2Vec Google News 300d", "word2vec-google-news-300", datasets_data)

    # ─── Ringkasan ────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  TABEL PERBANDINGAN — Baseline Non-SNN")
    print(f"{'='*80}")
    print(f"  {'Model':<28} | {'STS-B (r)':>11} | {'Teacher (r)':>11} | {'ms/pair':>8}")
    print(f"  {'-'*28}-+-{'-'*11}-+-{'-'*11}-+-{'-'*8}")

    rows = [
        "GloVe-100d",
        "GloVe-300d",
        "Word2Vec-300d",
    ]
    for key in rows:
        r_stsb = results.get(key, {}).get("results", {}).get("STS-B", {}).get("pearson", "N/A")
        r_tch  = results.get(key, {}).get("results", {}).get("All-STS-Teacher", {}).get("pearson", "N/A")
        ms_val = results.get(key, {}).get("results", {}).get("STS-B", {}).get("ms_per_pair", "N/A")
        
        str_stsb = f"{r_stsb:.4f}" if isinstance(r_stsb, float) else str(r_stsb)
        str_tch  = f"{r_tch:.4f}" if isinstance(r_tch, float) else str(r_tch)
        str_ms   = f"{ms_val:.3f}" if isinstance(ms_val, float) else str(ms_val)
        
        print(f"  {key:<28} | {str_stsb:>11} | {str_tch:>11} | {str_ms:>8}")

    print(f"{'='*80}\n")

    # Simpan hasil JSON
    with open(OUT_PATH, "w") as f:
        json.dump({"baseline_results": results}, f, indent=2)
    print(f"✓ Hasil disimpan ke: {OUT_PATH}")

if __name__ == "__main__":
    main()

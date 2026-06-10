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

STS_PATH  = "experiment/file_model/sts-b_valid.json"
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

def load_stsb() -> list[dict]:
    with open(STS_PATH) as f:
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

def run_gensim_baseline(name: str, model_key: str, data: list[dict]) -> dict:
    print_header(f"BASELINE: {name}  [{model_key}]")
    try:
        import gensim.downloader as api
        print(f"  Mengunduh / memuat {model_key}...")
        t_load = time.time()
        model = api.load(model_key)
        load_sec = time.time() - t_load
        print(f"  Load selesai dalam {load_sec:.1f}s  |  dim={model.vector_size}  |  vocab={len(model)}")

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
        print(f"  Pearson r : {r:.4f}")
        print(f"  ms/pair   : {ms:.3f}")
        return {
            "pearson": r,
            "ms_per_pair": round(ms, 3),
            "dim": model.vector_size,
            "vocab_size": len(model),
            "model_key": model_key,
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    data = load_stsb()
    print(f"Dataset STS-B validation: {len(data)} pasangan kalimat")

    results = {}

    # Floor baseline
    results["Random-300d"]   = baseline_random(data)

    # TF-IDF (zero external deps)
    results["TF-IDF"]        = baseline_tfidf(data)

    # GloVe 100d (sekitar 128MB, lebih cepat diunduh)
    results["GloVe-100d"]    = run_gensim_baseline(
        "GloVe Wikipedia+Gigaword 100d", "glove-wiki-gigaword-100", data)

    # GloVe 300d
    results["GloVe-300d"]    = run_gensim_baseline(
        "GloVe Wikipedia+Gigaword 300d", "glove-wiki-gigaword-300", data)

    # Word2Vec Google News 300d (~1.6GB — bisa di-skip jika koneksi lambat)
    results["Word2Vec-300d"] = run_gensim_baseline(
        "Word2Vec Google News 300d", "word2vec-google-news-300", data)

    # ─── Ringkasan ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TABEL PERBANDINGAN — Baseline Non-SNN vs SNN Kami")
    print(f"{'='*60}")
    print(f"  {'Model':<28} | {'Pearson (r)':>11} | {'ms/pair':>8}")
    print(f"  {'-'*28}-+-{'-'*11}-+-{'-'*8}")

    rows = [
        ("Random-300d (floor)",      results.get("Random-300d", {}).get("pearson", "N/A")),
        ("TF-IDF Cosine",            results.get("TF-IDF", {}).get("pearson", "N/A")),
        ("GloVe 100d (mean pool)",   results.get("GloVe-100d", {}).get("pearson", "N/A")),
        ("GloVe 300d (mean pool)",   results.get("GloVe-300d", {}).get("pearson", "N/A")),
        ("Word2Vec 300d (mean pool)",results.get("Word2Vec-300d", {}).get("pearson", "N/A")),
        # SNN kita (dari full_eval_controlled.json)
        ("─── SNN Ours ───────────────", "──────"),
        ("SNN-T32+Att (Human-Only)", 0.6091),
        ("SNN-T32+Att (Distil AI)",  0.6315),   # best
    ]
    for label, r in rows:
        r_str = f"{r:.4f}" if isinstance(r, float) else str(r)
        ms_val = results.get(label.split("(")[0].strip(), {}).get("ms_per_pair", "")
        ms_str = f"{ms_val:.3f}" if isinstance(ms_val, float) else "~1.05"
        print(f"  {label:<28} | {r_str:>11} | {ms_str:>8}")

    print(f"{'='*60}\n")

    # Simpan hasil JSON
    with open(OUT_PATH, "w") as f:
        json.dump({"baseline_results": results}, f, indent=2)
    print(f"✓ Hasil disimpan ke: {OUT_PATH}")

if __name__ == "__main__":
    main()

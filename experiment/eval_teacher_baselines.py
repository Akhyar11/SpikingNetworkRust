"""Evaluasi baseline Transformer efisien terhadap GOLD LABEL manusia.

Model yang dievaluasi pada 7 benchmark STS standar (MTEB format, gold labels):
  1. sentence-transformers/all-MiniLM-L6-v2 (fp32)  -> teacher model paper
  2. all-MiniLM-L6-v2 terkuantisasi int8 (dynamic)   -> baseline efisiensi
  3. sentence-transformers/distilbert-base-nli-stsb-mean-tokens -> baseline klasik

Metrik per benchmark: Pearson, Spearman (scipy), latensi ms/pair.
Output: experiment/file_model/gold_metrics_baselines.json
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "STS-12": ROOT / "experiment/file_model/mteb_sts12-sts.json",
    "STS-13": ROOT / "experiment/file_model/mteb_sts13-sts.json",
    "STS-14": ROOT / "experiment/file_model/mteb_sts14-sts.json",
    "STS-15": ROOT / "experiment/file_model/mteb_sts15-sts.json",
    "STS-16": ROOT / "experiment/file_model/mteb_sts16-sts.json",
    "STS-B": ROOT / "experiment/file_model/sts-b_valid.json",
    "SICK-R": ROOT / "experiment/file_model/mteb_sickr-sts.json",
}
OUT = ROOT / "experiment/file_model/gold_metrics_baselines.json"

MODELS = [
    ("all-MiniLM-L6-v2 (fp32, teacher)", "sentence-transformers/all-MiniLM-L6-v2", False),
    ("all-MiniLM-L6-v2 (int8 dynamic)", "sentence-transformers/all-MiniLM-L6-v2", True),
    ("distilbert-base-nli-stsb-mean-tokens", "sentence-transformers/distilbert-base-nli-stsb-mean-tokens", False),
]

BATCH_SIZE = 64
MAX_LEN = 128


def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def embed(model, tokenizer, sentences):
    embs = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i : i + BATCH_SIZE]
        enc = tokenizer(batch, padding="max_length", max_length=MAX_LEN,
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        e = mean_pool(out.last_hidden_state, enc["attention_mask"])
        embs.append(F.normalize(e, p=2, dim=1))
    return torch.cat(embs)


def eval_model(name, hf_id, quantize, datasets):
    print(f"\n=== {name} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id)
    if quantize:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval()

    result = {}
    w_p = w_s = 0.0
    total = 0
    for ds_name, pairs in datasets.items():
        s1 = [p["sentence1"] for p in pairs]
        s2 = [p["sentence2"] for p in pairs]
        gold = np.array([p["score"] for p in pairs], dtype=np.float64)

        t0 = time.perf_counter()
        e1 = embed(model, tokenizer, s1)
        e2 = embed(model, tokenizer, s2)
        dur = time.perf_counter() - t0

        sims = F.cosine_similarity(e1, e2).numpy().astype(np.float64)
        p = pearsonr(sims, gold).statistic
        r = spearmanr(sims, gold).statistic
        ms = dur * 1000.0 / len(pairs)
        w_p += p * len(pairs)
        w_s += r * len(pairs)
        total += len(pairs)
        print(f"  {ds_name:<8} n={len(pairs):<6} Pearson={p:.4f} Spearman={r:.4f} {ms:.3f} ms/pair", flush=True)
        result[ds_name] = {
            "n_pairs": len(pairs),
            "pearson": round(float(p), 4),
            "spearman": round(float(r), 4),
            "ms_per_pair_batch64": round(ms, 3),
        }

    # Latensi batch=1 (apples-to-apples dengan angka SNN per-pair), sampel 500 pair STS-B
    sample = datasets["STS-B"][:500]
    t0 = time.perf_counter()
    for pair in sample:
        enc = tokenizer([pair["sentence1"], pair["sentence2"]], padding="max_length",
                        max_length=MAX_LEN, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        e = F.normalize(mean_pool(out.last_hidden_state, enc["attention_mask"]), p=2, dim=1)
        _ = F.cosine_similarity(e[0:1], e[1:2])
    ms1 = (time.perf_counter() - t0) * 1000.0 / len(sample)

    result["ALL-STS-weighted"] = {
        "total_pairs": total,
        "pearson": round(float(w_p / total), 4),
        "spearman": round(float(w_s / total), 4),
    }
    result["latency_ms_per_pair_batch1_STSB500"] = round(ms1, 3)
    print(f"  ALL-STS weighted: Pearson={result['ALL-STS-weighted']['pearson']} "
          f"Spearman={result['ALL-STS-weighted']['spearman']} | batch1 latency {ms1:.2f} ms/pair", flush=True)
    del model
    return result


def main():
    datasets = {}
    for name, path in DATASETS.items():
        with open(path, encoding="utf-8") as f:
            datasets[name] = json.load(f)
        print(f"Loaded {name}: {len(datasets[name])} pairs", flush=True)

    out = {
        "note": "Pearson/Spearman vs human gold labels. Mean-pooling + cosine. Teks apa adanya (cased). "
                "Latency batch1 diukur pada 500 pair STS-B.",
        "models": {},
    }
    for name, hf_id, quantize in MODELS:
        out["models"][name] = eval_model(name, hf_id, quantize, datasets)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nDisimpan ke {OUT}")


if __name__ == "__main__":
    main()

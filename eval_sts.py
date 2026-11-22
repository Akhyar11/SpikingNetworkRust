"""STS evaluation: Spearman/Pearson (x100) with cosine similarity, one pooled set per benchmark
(STS12-16 test, STS-B test, SICK-R full set as in MTEB). Protocol follows SBERT/SimCSE 'all' setting.

Usage:
  python3 eval_sts.py --bin results/snn_d256_s1          # Rust embeddings (.emb.bin from train_export)
  python3 eval_sts.py --glove                            # validation: avg GloVe 840B (SBERT table)
  python3 eval_sts.py --teacher                          # all-MiniLM-L6-v2
Library: evaluate(embed_fn) where embed_fn(list[str]) -> np.ndarray [n, d]; or evaluate_matrix(E).
"""
import json, argparse, numpy as np, os, sys
from scipy.stats import spearmanr, pearsonr
SETS = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSB', 'SICKR']
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_eval')

def load_sets():
    return {n: [json.loads(l) for l in open(f'{DIR}/{n}.jsonl')] for n in SETS}

def all_sentences():
    return json.load(open(f'{DIR}/all_eval_sentences.json'))

def evaluate_matrix(E, sents=None, shuffle_fn=None):
    """E: embeddings aligned with all_eval_sentences.json order."""
    sents = sents or all_sentences()
    idx = {s: i for i, s in enumerate(sents)}
    E = np.asarray(E, dtype=np.float64)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    res = {}
    for n, d in load_sets().items():
        a = E[[idx[x['s1']] for x in d]]; b = E[[idx[x['s2']] for x in d]]
        cos = (a * b).sum(1); gold = np.array([x['score'] for x in d])
        res[n] = {'spearman': 100 * spearmanr(cos, gold).statistic, 'pearson': 100 * pearsonr(cos, gold).statistic, 'n': len(d)}
    for m in ('spearman', 'pearson'):
        res['avg_' + m] = float(np.mean([res[n][m] for n in SETS]))
    return res

def evaluate(embed_fn):
    sents = all_sentences()
    return evaluate_matrix(embed_fn(sents), sents)

def load_bin(prefix):
    meta = json.load(open(prefix + '.json'))
    E = np.fromfile(prefix + '.emb.bin', dtype='<f4').reshape(-1, meta['d'])
    assert len(E) == meta['n_sentences']
    return E

def show(name, r):
    print(f"{name:28s} " + ' '.join(f"{r[n]['spearman']:6.2f}" for n in SETS) + f" | avg {r['avg_spearman']:6.2f} (pearson avg {r['avg_pearson']:6.2f})")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin'); ap.add_argument('--glove', action='store_true'); ap.add_argument('--teacher', action='store_true')
    a = ap.parse_args()
    print(f"{'':28s} " + ' '.join(f"{n:>6s}" for n in SETS))
    if a.bin: show(a.bin, evaluate_matrix(load_bin(a.bin)))
    if a.teacher:
        import torch; from transformers import AutoTokenizer, AutoModel
        t = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); m = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').eval()
        def f(ss):
            out = []
            with torch.no_grad():
                for i in range(0, len(ss), 128):
                    x = t(ss[i:i+128], padding=True, truncation=True, max_length=256, return_tensors='pt'); h = m(**x).last_hidden_state
                    mk = x['attention_mask'].unsqueeze(-1).float(); out.append(((h * mk).sum(1) / mk.sum(1)).numpy())
            return np.concatenate(out)
        r = evaluate(f); show('all-MiniLM-L6-v2', r); json.dump(r, open('results/teacher_minilm.json', 'w'), indent=1)
    if a.glove:
        from glove_avg import glove_embed
        r = evaluate(glove_embed); show('avg GloVe 840B', r); json.dump(r, open('results/glove840_validation.json', 'w'), indent=1)

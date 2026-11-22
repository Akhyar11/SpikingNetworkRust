"""E1(b),(c): static baselines evaluated with eval_sts protocol. Run from repo root."""
import sys, json, numpy as np, torch
sys.path.insert(0, '.')
from eval_sts import evaluate, show, SETS
from transformers import AutoTokenizer, AutoModel
res = {}
t = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); m = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
Wt = m.embeddings.word_embeddings.weight.detach().numpy()
def static_minilm(ss):
    out = np.zeros((len(ss), Wt.shape[1]), dtype=np.float32)
    for k, s in enumerate(ss):
        ids = t(s, add_special_tokens=False, truncation=True, max_length=128)['input_ids']
        if ids: out[k] = Wt[ids].mean(0)
    return out
r = evaluate(static_minilm); show('static MiniLM tok-emb mean', r); res['static_minilm_tokemb_mean'] = r
res['static_minilm_params'] = {'total': int(Wt.size), 'dim': Wt.shape[1]}
json.dump(res, open('results/static_baselines.json', 'w'), indent=1)

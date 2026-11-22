"""Aggregate all runs -> results/*.csv (mean +- std over seeds, Spearman x100). Run: python3 aggregate.py"""
import json, glob, os, numpy as np, csv
from eval_sts import evaluate_matrix, load_bin, SETS, all_sentences
S = all_sentences()
def rust(prefix, shuf=False):
    f = prefix + ('.shuf.emb.bin' if shuf else '.emb.bin'); m = json.load(open(prefix + '.json'))
    return np.fromfile(f, dtype='<f4').reshape(-1, m['d'])
def torch_(prefix, shuf=False):
    return np.load(prefix + ('.eval_shuf.npy' if shuf else '.eval.npy'))
groups = {  # name -> (loader, prefix pattern)
 'LIF-pooler (Rust, distill) [ours]': (rust, 'results/snn_d256_s{}'),
 'LIF-pooler (Rust, STS-B-train human labels)': (rust, 'results/snn_d256_human_s{}'),
 'LIF-pooler (Rust, ORIGINAL leaky data)': (rust, 'results/snn_d256_leaky_s{}'),
 'mean-pool table, no LIF (torch)': (torch_, 'results/meanpool_s{}'),
 'ANN twin (continuous, exact backprop)': (torch_, 'results/ann_twin_s{}'),
 'SNN, exact surrogate BPTT (torch)': (torch_, 'results/snn_sg_s{}'),
 'SNN, exact BPTT + trained float beta': (torch_, 'results/snn_sg_beta_s{}'),
 'GRU(256) mean-pool (torch)': (torch_, 'results/gru_s{}'),
}
rows = []; shuf_rows = []; summary = {}
for name, (ld, pat) in groups.items():
    res = []; resS = []
    for s in (1, 2, 3):
        p = pat.format(s)
        try:
            res.append(evaluate_matrix(ld(p), S))
            try: resS.append(evaluate_matrix(ld(p, True), S))
            except Exception: pass
        except Exception: pass
    if not res: continue
    def ms(rs, key, sub='spearman'):
        v = [r[key][sub] if isinstance(r[key], dict) else r[key] for r in rs]; return np.mean(v), np.std(v)
    row = {'model': name, 'n_seeds': len(res)}
    for k in SETS + ['avg_spearman', 'avg_pearson']:
        sub = 'spearman'
        m, sd = ms(res, k, sub); row[k] = f'{m:.2f}±{sd:.2f}'
    rows.append(row); summary[name] = {k: float(ms(res, k)[0]) for k in SETS + ['avg_spearman']}
    if resS:
        d = {'model': name, 'n_seeds': len(resS)}
        for k in SETS + ['avg_spearman']:
            a, _ = ms(res[:len(resS)], k); b, sd = ms(resS, k); d[k] = f'{b:.2f}±{sd:.2f} (Δ{b-a:+.2f})'
        shuf_rows.append(d)
def dump(rows, f):
    if rows:
        with open(f, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f); [print(' ', r) for r in rows]
dump(rows, 'results/table_spearman.csv'); dump(shuf_rows, 'results/table_shuffle.csv')
# fixed baselines
fx = []
for name, f in [('all-MiniLM-L6-v2 (teacher)', 'results/teacher_minilm.json'), ('avg GloVe 840B.300d', 'results/glove840_validation.json')]:
    if os.path.exists(f):
        r = json.load(open(f)); fx.append({'model': name, **{k: f"{r[k]['spearman']:.2f}" for k in SETS}, 'avg_spearman': f"{r['avg_spearman']:.2f}"})
sb = 'results/static_baselines.json'
if os.path.exists(sb):
    r = json.load(open(sb))['static_minilm_tokemb_mean']; fx.append({'model': 'static MiniLM token-emb mean-pool', **{k: f"{r[k]['spearman']:.2f}" for k in SETS}, 'avg_spearman': f"{r['avg_spearman']:.2f}"})
dump(fx, 'results/table_fixed_baselines.csv')
# training/inference cost
cost = []
for name, (ld, pat) in groups.items():
    ms_ = [json.load(open(pat.format(s) + '.json')) for s in (1, 2, 3) if os.path.exists(pat.format(s) + '.json')]
    if ms_: cost.append({'model': name, 'train_s_mean': np.mean([m['train_seconds'] for m in ms_]), 'enc_ms_per_sent': np.mean([m['encode_ms_per_sentence'] for m in ms_]), **({'sparsity_excl_pad': np.mean([m['sparsity_excl_pad'] for m in ms_]), 'spikes_per_sentence': np.mean([m['spikes_per_sentence'] for m in ms_])} if 'sparsity_excl_pad' in ms_[0] else {})})
dump(cost, 'results/table_cost.csv')

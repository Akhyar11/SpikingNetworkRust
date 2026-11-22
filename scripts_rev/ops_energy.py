"""Phase 1.3: honest op counts + 45nm energy estimate (Horowitz ISSCC'14: E_MAC=4.6pJ (32b float mult 3.7 + add 0.9), E_AC=0.9pJ).
Counts follow the executed code (src/models/sentence_embedder/{encode,train}.rs, core/lifStep.rs, core/delta.rs).
Memory accesses are NOT counted. Per sentence (inference) and per training step (32 pairs = 64 sentences)."""
import json, glob, numpy as np, csv
D = 256; E_MAC, E_AC, E_MUL, E_ADD = 4.6, 0.9, 3.7, 0.9
meta = [json.load(open(f)) for f in sorted(glob.glob('results/snn_d256_s[123].json'))]
n_eval = np.fromfile('data_eval/eval.tok.bin', dtype='<i4').reshape(-1, 128); n_tok = float((n_eval != 0).sum(1).mean())
tr = np.fromfile('data_eval/train.tok.bin', dtype='<i4').reshape(-1, 128); n_tok_tr = float((tr != 0).sum(1).mean())
s_eval = float(np.mean([m['spikes_per_sentence'] for m in meta])); sparsity = float(np.mean([m['sparsity_excl_pad'] for m in meta]))
rows = []
def add(stage, mac, ac, other=''): rows.append({'stage': stage, 'float_mult': mac, 'adds_or_AC': ac, 'other(cmp/sqrt/div)': other})
n = n_tok
# --- inference per sentence ---
add('inf: embedding LIF (p*beta; state is 0 -> removable in principle)', n * D, n * D, 'cmp n*D')
add('inf: pooler add-only projection (identity kernel; code does D adds/spike)', 0, s_eval * D, 'branch per input')
add('inf: pooler LIF (beta*p float mult, add, clip, cmp, soft reset)', n * D, 2 * n * D, 'cmp 2nD')
add('inf: accumulate E = sum_t V', 0, n * D)
add('inf: z-score + L2 norm', 2 * D, 3 * D, 'sqrt 1, div D')
# --- training step, 64 sentences, 32 pairs ---
B = 64; nt = n_tok_tr
add('train: forward (64 sentences, same stages as above)', B * (n * D + n * D + 2 * D) / 1, B * (s_eval * D + 3 * n * D + 3 * D), '')
add('train: loss (32 pairs): cosine dot + 2 grads*err*margin', 32 * (D + 2 * D), 32 * D, '')
add('train: backward embedding (broadcast error, mask, lr*acc_err per touched row)', B * nt * D, B * nt * D, 'cmp nD')
add('train: dense BPTT dW for pooler IF it were trained (L*Din*Dout MAC)', B * nt * D * D, 0, '')
add('train: delta-rule dW for pooler IF it were trained (code: inp*err for spikes)', B * s_eval * D, B * s_eval * D, '')
for r in rows: r['energy_nJ'] = (r['float_mult'] * E_MUL + r['adds_or_AC'] * E_ADD) / 1000
with open('results/ops_energy.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows: print(f"{r['stage'][:78]:78s} mult={r['float_mult']:.3g} add={r['adds_or_AC']:.3g} E={r['energy_nJ']:.3g} nJ")
dens = 1 - sparsity
print(f'avg tokens/eval sentence {n_tok:.1f}, train {nt:.1f}; spikes/sent {s_eval:.0f}; sparsity(excl pad) {sparsity:.3f}')
print(f'Delta-only dW energy ratio (dense MAC vs spike-gated): {E_MAC/(E_AC*dens):.1f}x if only the gating is counted at density {dens:.3f} (paper claimed 15.5x)')

"""PyTorch baselines for the revision (E1a, E2, E8, E1d + exact surrogate-BPTT of the SNN).
Variants (all: vocab 64000 BPE table x D=256, L<=128, same distillation data/loss as the Rust run):
  meanpool   : mean-pool of the trainable table, no LIF (E1a)
  ann_twin   : same recurrent pooler, continuous activation relu(x-theta), exact backprop (E2)
  snn_sg     : spiking forward identical to the Rust model, EXACT autograd BPTT w/ rectangular surrogate
  snn_sg_beta: as snn_sg but beta is a trained float (sigmoid-param) (E8)
  gru        : GRU(256,256), mean-pooled hidden states (E1d)
Hyper-parameters were fixed a priori (NOT tuned on any STS/SICK test set).
Usage: python3 scripts_rev/torch_models.py --variant ann_twin --seed 1 --epochs 1 --out results/ann_twin_s1
"""
import argparse, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, os, sys
D, V, L = 256, 64000, 128
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

def load_tok(p, n=None):
    a = np.fromfile(p, dtype='<i4').reshape(-1, L)
    return torch.from_numpy(a.astype(np.int64))

class Enc(nn.Module):
    def __init__(self, variant, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        lim = (6 / (V + D)) ** .5 * V ** .5          # same init scale as the Rust SpikingEmbedding
        w = (torch.rand(V, D, generator=g) * 2 - 1) * lim
        self.table = nn.Embedding(V, D, sparse=True); self.table.weight.data.copy_(w.clamp(-1, 1))
        self.variant = variant
        shift_e = torch.randint(2, 8, (D,), generator=g).float(); shift_p = torch.randint(2, 6, (D,), generator=g).float()
        self.theta_e = nn.Parameter(torch.rand(D, generator=g) * .09 + .01, requires_grad=False)
        self.theta_p = nn.Parameter(torch.rand(D, generator=g) * .5 + .5, requires_grad=False)
        beta = 1 - 2 ** (-shift_p)
        if variant == 'snn_sg_beta':
            self.beta_logit = nn.Parameter(torch.log(beta / (1 - beta)))
        else:
            self.register_buffer('beta', beta)
        if variant == 'gru':
            self.gru = nn.GRU(D, D, batch_first=True)
        if variant == 'lstm':
            self.gru = nn.LSTM(D, D, batch_first=True)

    def get_beta(self):
        return torch.sigmoid(self.beta_logit) if self.variant == 'snn_sg_beta' else self.beta

    def forward(self, tok, return_spikes=False):
        T = int((tok != 0).sum(1).max().clamp(min=1)); tok = tok[:, :T]
        mask = (tok != 0).float().unsqueeze(-1); x = self.table(tok)            # [B,T,D]
        if self.variant == 'meanpool':
            e = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        elif self.variant in ('gru', 'lstm'):
            h, _ = self.gru(x * mask); e = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            if self.variant == 'ann_twin':
                a = F.relu(x - self.theta_e) * mask
            else:
                s = SpikeFn.apply(x - self.theta_e) * mask; a = s
            beta = self.get_beta(); p = torch.zeros(tok.shape[0], D); acc = torch.zeros(tok.shape[0], D); nsp = 0
            for t in range(T):
                v = torch.clamp(beta * p + a[:, t], max=1.0)                   # pre-reset potential (min(1, .) as in lifStep.rs)
                acc = acc + v * mask[:, t]
                if self.variant == 'ann_twin':
                    p = v
                else:
                    sp = SpikeFn.apply(v - self.theta_p); p = v - sp * self.theta_p
            e = acc
        e = e - e.mean(1, keepdim=True)
        return F.normalize(e, dim=1, eps=1e-8)

class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u); return (u >= 0).float()
    @staticmethod
    def backward(ctx, g):
        u, = ctx.saved_tensors; return g * (u.abs() <= 1.0).float()          # rectangular window w=1.0 (as maskSurrogate)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--epochs', type=float, default=1); ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--out', required=True); ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--data', default='decontam'); ap.add_argument('--max_steps', type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
    m = Enc(a.variant, a.seed)
    nparams = sum(p.numel() for p in m.parameters() if p.requires_grad)
    dense = [p for n, p in m.named_parameters() if n != 'table.weight' and p.requires_grad]
    opt_s = torch.optim.SparseAdam([m.table.weight], lr=0.01)
    opt_d = torch.optim.Adam(dense, lr=1e-3) if dense else None
    if a.data == 'decontam':
        tok = load_tok(f'{ROOT}/data_eval/train.tok.bin'); sc = torch.tensor([x['score'] for x in json.load(open(f'{ROOT}/data_eval/train_decontam.json'))])
    else:
        raise SystemExit('only decontam supported here')
    P = len(sc); steps = int(a.epochs * (P // a.bs)); a_tok, b_tok = tok[0::2], tok[1::2]
    sched_total = steps; t0 = time.time(); g = torch.Generator().manual_seed(a.seed); perm = torch.randperm(P, generator=g); pos = 0; losses = []
    m.train()
    for step in range(steps):
        if pos + a.bs > P: perm = torch.randperm(P, generator=g); pos = 0
        i = perm[pos:pos + a.bs]; pos += a.bs
        lr_f = 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * step / sched_total))
        for o, base in ((opt_s, 0.01), (opt_d, 1e-3)):
            if o is not None:
                for pg in o.param_groups: pg['lr'] = base * lr_f
        ea, eb = m(a_tok[i]), m(b_tok[i])
        loss = F.mse_loss((ea * eb).sum(1), sc[i])
        opt_s.zero_grad(); 
        if opt_d: opt_d.zero_grad()
        loss.backward(); opt_s.step()
        if opt_d: opt_d.step()
        with torch.no_grad(): m.table.weight.clamp_(-1, 1)          # same clip as Rust
        losses.append(loss.item())
        if step % 200 == 0: print(f'step {step}/{steps} loss {np.mean(losses[-200:]):.4f} {time.time()-t0:.0f}s', flush=True)
        if a.max_steps and step >= a.max_steps: break
    train_s = time.time() - t0
    m.eval(); out = {}
    for name in ('eval', 'eval_shuf'):
        et = load_tok(f'{ROOT}/data_eval/{name}.tok.bin'); embs = []; t1 = time.time()
        with torch.no_grad():
            for j in range(0, len(et), 128): embs.append(m(et[j:j + 128]).numpy())
        E = np.concatenate(embs); np.save(f'{a.out}.{name}.npy', E.astype(np.float32))
        if name == 'eval': enc_ms = (time.time() - t1) * 1000 / len(et)
    json.dump({'variant': a.variant, 'seed': a.seed, 'epochs': a.epochs, 'steps': steps, 'train_seconds': train_s,
               'trainable_params_total': nparams, 'embedding_params': V * D, 'non_embedding_params': nparams - V * D,
               'encode_ms_per_sentence': enc_ms, 'final_loss': float(np.mean(losses[-200:])), 'bs': a.bs, 'threads': a.threads},
              open(f'{a.out}.json', 'w'), indent=1)
    print('done', train_s)
if __name__ == '__main__': main()

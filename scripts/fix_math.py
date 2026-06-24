import re

with open('tmlr-style-file-main/tmlr-style-file-main/main.tex', 'r') as f:
    text = f.read()

replacements = {
    r'S_{emb}': r'\mS_{emb}',
    r'V_{emb}': r'\mV_{emb}',
    r'\beta_{emb}': r'\bm{\beta}_{emb}',
    r'X^{(t)}': r'\mX^{(t)}',
    r'W_{emb}': r'\mW_{emb}',
    r'\theta_{emb}': r'\bm{\theta}_{emb}',
    r'S^{(t)}': r'\mS^{(t)}',
    r'V^{(t)}': r'\mV^{(t)}',
    r'Q_{i,d}': r'\mQ_{i,d}',
    r'K_{j,d}': r'\mK_{j,d}',
    r'M_{i,j}': r'\mM_{i,j}',
    r'S_{scores}': r'\mS_{scores}',
    r'S_{att}': r'\mS_{att}',
    r'S_{agg}': r'\mS_{agg}',
    r'X_{pool}': r'\mX_{pool}',
    r'W_{d}': r'\mW_{d}',
    r'V_{pool}': r'\mV_{pool}',
    r'\tilde{E}': r'\tilde{\mE}',
    r'\beta_{pool}': r'\bm{\beta}_{pool}',
    r'\delta_{out}': r'\bm{\delta}_{out}',
    r'\tilde{\delta}': r'\tilde{\bm{\delta}}',
    r'\Delta W': r'\Delta \mW',
    r' E ': r' \mE ',
    r' E =': r' \mE =',
    r'(Q)': r'(\mQ)',
    r'(K)': r'(\mK)',
    r'(V)': r'(\mV)',
    r'Spikes Q': r'Spikes \mQ',
    r'Spikes K': r'Spikes \mK',
    r'Spikes V': r'Spikes \mV',
    r'Q_{i,d}^{(t)}': r'\mQ_{i,d}^{(t)}',
    r'K_{j,d}^{(t)}': r'\mK_{j,d}^{(t)}',
}

for old, new in replacements.items():
    text = text.replace(old, new)

# Fix some exact replacements that might need care
text = text.replace(r'E \in', r'\mE \in')

with open('tmlr-style-file-main/tmlr-style-file-main/main.tex', 'w') as f:
    f.write(text)


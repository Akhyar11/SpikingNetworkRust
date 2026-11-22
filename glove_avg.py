"""Average-GloVe sentence embedder using the sentence-transformers 840B.300d checkpoint (the SBERT-paper baseline)."""
import json, glob, os, string, numpy as np
from safetensors.numpy import load_file
_S = {}
def _load():
    if _S: return
    d = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--sentence-transformers--average_word_embeddings_glove.840B.300d/snapshots/*/0_WordEmbeddings'))[0]
    cfg = json.load(open(d + '/whitespacetokenizer_config.json'))
    W = list(load_file(d + '/model.safetensors').values())[0]
    _S['w2i'] = {w: i for i, w in enumerate(cfg['vocab'])}; _S['W'] = W; _S['lower'] = cfg.get('do_lower_case', False)
def glove_embed(sents):
    _load(); w2i, W = _S['w2i'], _S['W']; out = np.zeros((len(sents), W.shape[1]), dtype=np.float32)
    for k, s in enumerate(sents):
        ids = []
        for t in (s.lower() if _S['lower'] else s).split():
            for c in (t, t.strip(string.punctuation), t.lower(), t.lower().strip(string.punctuation)):
                if c in w2i: ids.append(w2i[c]); break
        if ids: out[k] = W[ids].mean(0)
    return out

# Fallback: GloVe 6B.300d (gensim 'glove-wiki-gigaword-300', lowercase). NOT the 840B vectors used in the SBERT paper.
def glove6b_embed(sents):
    if 'kv' not in _S:
        import gensim.downloader as api
        _S['kv'] = api.load('glove-wiki-gigaword-300')
    kv = _S['kv']; out = np.zeros((len(sents), 300), dtype=np.float32)
    for k, s in enumerate(sents):
        ids = []
        for t in s.lower().split():
            for c in (t, t.strip(string.punctuation)):
                if c in kv.key_to_index: ids.append(kv.key_to_index[c]); break
        if ids: out[k] = kv.vectors[ids].mean(0)
    return out

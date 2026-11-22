import os
# Menghindari Out of Memory (OOM) error yang sering terjadi di Kaggle GPU T4
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import json
import optax
from dataclasses import dataclass
from transformers import AutoTokenizer
import functools
from jax import lax

@dataclass
class Config:
    vocab_size: int = 32000
    d_model: int = 256
    max_length: int = 128
    learning_rate: float = 1e-3
    pad_token_id: int = 0

cfg = Config()

def init_parameters(key, cfg: Config):
    keys = jax.random.split(key, 3)
    params = {}

    # Embedding Matrix
    emb_weight = (
        jax.random.normal(keys[0], (cfg.vocab_size, cfg.d_model)) * 0.02
    )
    # Hyperpolarized Padding (set pad token embedding to 0.0 for LSTM)
    params["embedding_weight"] = emb_weight.at[cfg.pad_token_id].set(0.0)

    # LSTM Encoder Weights (d_model input, d_model hidden -> 4 * d_model output)
    # W_ih, W_hh combined into one matrix W (shape: [2*d_model, 4*d_model])
    limit_enc = np.sqrt(1.0 / cfg.d_model)
    params["enc_lstm_w"] = jax.random.uniform(
        keys[1],
        (2 * cfg.d_model, 4 * cfg.d_model),
        minval=-limit_enc,
        maxval=limit_enc
    )
    params["enc_lstm_b"] = jnp.zeros((4 * cfg.d_model,))
    # Initialize forget gate bias to 1.0
    params["enc_lstm_b"] = params["enc_lstm_b"].at[cfg.d_model : 2*cfg.d_model].set(1.0)

    return params

def lstm_step(carry, x, w, b, d_model):
    h, c = carry
    # z_t = [x_t, h_{t-1}]
    z = jnp.concatenate([x, h], axis=-1)
    
    # Gates
    gates = jnp.dot(z, w) + b
    
    # Split gates
    i, f, g, o = jnp.split(gates, 4, axis=-1)
    
    i = jax.nn.sigmoid(i)
    f = jax.nn.sigmoid(f)
    g = jnp.tanh(g)
    o = jax.nn.sigmoid(o)
    
    c_new = f * c + i * g
    h_new = o * jnp.tanh(c_new)
    
    return (h_new, c_new), h_new

def embedding_forward(params, token_ids):
    weight = params["embedding_weight"]
    return weight[token_ids]

def encoder_forward(params, embedding):
    d_model = cfg.d_model
    
    def step_fn(carry, x):
        return lstm_step(carry, x, params["enc_lstm_w"], params["enc_lstm_b"], d_model)
    
    # Initial states
    h0 = jnp.zeros(d_model, dtype=embedding.dtype)
    c0 = jnp.zeros(d_model, dtype=embedding.dtype)
    
    _, hidden_history = lax.scan(step_fn, (h0, c0), embedding)
    return hidden_history

def model_forward_single(params, token_ids):
    embedding = embedding_forward(params, token_ids)
    
    # Lakukan LSTM Encoding
    hidden_history = encoder_forward(params, embedding)
    
    # Pooler: Sama seperti main.py, kita menggunakan sum pooling sepanjang axis waktu
    sentence_embedding = jnp.sum(hidden_history, axis=0)
    
    return sentence_embedding

# Vectorize model_forward to automatically handle batch dimension
model_forward = jax.vmap(model_forward_single, in_axes=(None, 0))

def compute_loss(params, tokens_1, tokens_2, target_scores, margin=0.5):
    emb_1 = model_forward(params, tokens_1)
    emb_2 = model_forward(params, tokens_2)
    
    emb_1_centered = emb_1 - jnp.mean(emb_1, axis=-1, keepdims=True)
    emb_2_centered = emb_2 - jnp.mean(emb_2, axis=-1, keepdims=True)
    
    emb_1_norm = emb_1_centered / jnp.sqrt(jnp.sum(emb_1_centered**2, axis=-1, keepdims=True) + 1e-8)
    emb_2_norm = emb_2_centered / jnp.sqrt(jnp.sum(emb_2_centered**2, axis=-1, keepdims=True) + 1e-8)
    
    predicted_scores = jnp.sum(emb_1_norm * emb_2_norm, axis=-1)
    
    mse = (predicted_scores - target_scores) ** 2
    loss = jnp.mean(mse * margin)
    
    return loss

DATASET = "/kaggle/input/datasets/akhyarsafrudin/train-distilation-dataset/teacher_distillation_dataset_scored.json"
VAL_DATASET = "/kaggle/input/datasets/akhyarsafrudin/all-sts-distilation/all_sts_teacher_scored.json"

# Coba gunakan path STS-B (lokal atau sesuaikan di Kaggle jika berbeda)
STS_B_DATASET = "experiment/file_model/sts-b_valid.json"

def setup_optimizer(total_steps, base_lr=2.0):
    scheduler = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=total_steps,
        alpha=0.01
    )
    optimizer = optax.adam(learning_rate=scheduler)
    return optimizer

@functools.partial(jax.jit, static_argnames=['optimizer'])
def train_step(params, opt_state, tokens_1, tokens_2, target_scores, optimizer):
    loss, grads = jax.value_and_grad(compute_loss)(
        params, tokens_1, tokens_2, target_scores
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

@jax.jit
def predict_scores(params, tokens_1, tokens_2):
    emb_1 = model_forward(params, tokens_1)
    emb_2 = model_forward(params, tokens_2)
    
    emb_1_centered = emb_1 - jnp.mean(emb_1, axis=-1, keepdims=True)
    emb_2_centered = emb_2 - jnp.mean(emb_2, axis=-1, keepdims=True)
    
    emb_1_norm = emb_1_centered / jnp.sqrt(jnp.sum(emb_1_centered**2, axis=-1, keepdims=True) + 1e-8)
    emb_2_norm = emb_2_centered / jnp.sqrt(jnp.sum(emb_2_centered**2, axis=-1, keepdims=True) + 1e-8)
    
    return jnp.sum(emb_1_norm * emb_2_norm, axis=-1)

def evaluate_metrics(params, tokens_1, tokens_2, target_scores, batch_size):
    num_samples = len(target_scores)
    all_preds = []
    
    for i in range(0, num_samples, batch_size):
        b_tok_1 = tokens_1[i:i + batch_size]
        b_tok_2 = tokens_2[i:i + batch_size]
        preds = predict_scores(params, b_tok_1, b_tok_2)
        all_preds.append(np.array(preds))
        
    all_preds = np.concatenate(all_preds)
    targets = np.array(target_scores)
    
    pred_mean = np.mean(all_preds)
    tgt_mean = np.mean(targets)
    
    cov = np.sum((all_preds - pred_mean) * (targets - tgt_mean))
    std_pred = np.sqrt(np.sum((all_preds - pred_mean)**2))
    std_tgt = np.sqrt(np.sum((targets - tgt_mean)**2))
    
    pearson_corr = cov / (std_pred * std_tgt + 1e-8)
    
    try:
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(all_preds, targets)
    except ImportError:
        spearman_corr = 0.0
        
    return float(pearson_corr), float(spearman_corr)

def prepare_dataset(file_path, tokenizer, max_length):
    print(f"Loading dataset from {file_path}...")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Dataset not found locally. Creating dummy data for testing...")
        data = [{"s1": "hello world", "s2": "hi earth", "score": 0.8}] * 100

    s1_texts = [item.get("s1", item.get("sentence1", item.get("text1"))) for item in data]
    s2_texts = [item.get("s2", item.get("sentence2", item.get("text2"))) for item in data]
    scores = [float(item.get("score", item.get("similarity_score", item.get("label", 0.0)))) for item in data]

    print("Tokenizing sentences...")
    encoded_s1 = tokenizer(
        s1_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    encoded_s2 = tokenizer(
        s2_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )

    token_ids_1 = jnp.array(encoded_s1["input_ids"])
    token_ids_2 = jnp.array(encoded_s2["input_ids"])
    y_scores = jnp.array(scores, dtype=jnp.float32)

    return token_ids_1, token_ids_2, y_scores

def create_batches(token_ids_1, token_ids_2, scores, batch_size):
    num_samples = len(scores)
    indices = np.random.permutation(num_samples)
    
    for i in range(0, num_samples, batch_size):
        batch_idx = indices[i:i + batch_size]
        yield (
            token_ids_1[batch_idx],
            token_ids_2[batch_idx],
            scores[batch_idx]
        )

def main():
    print("Available JAX devices:", jax.devices())
    if jax.devices()[0].device_kind.lower() == 'cpu':
        print("⚠️ WARNING: JAX berjalan di CPU!")
    else:
        print(f"✅ JAX berjalan menggunakan GPU: {jax.devices()[0].device_kind}")

    print("Initializing tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("PulseNet-Labs/spiking-sentence-embedder", trust_remote_code=True)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    except Exception as e:
        print(f"Could not load tokenizer ({e}). Using dummy pad_token_id.")
        tokenizer = None
        pad_token_id = 0

    global cfg
    cfg.pad_token_id = pad_token_id
    key = jax.random.PRNGKey(42)
    
    print("Initializing LSTM parameters...")
    params = init_parameters(key, cfg)
    
    if tokenizer is not None:
        tokens_1, tokens_2, scores = prepare_dataset(DATASET, tokenizer, cfg.max_length)
        val_tokens_1, val_tokens_2, val_scores = prepare_dataset(VAL_DATASET, tokenizer, cfg.max_length)
        
        # Fallback for STS-B dataset in kaggle
        if not os.path.exists(STS_B_DATASET):
            possible_kaggle_path = "/kaggle/input/datasets/akhyarsafrudin/sts-b-distilation/sts-b_valid.json"
            if os.path.exists(possible_kaggle_path):
                stsb_path = possible_kaggle_path
            else:
                stsb_path = STS_B_DATASET # will trigger dummy
        else:
            stsb_path = STS_B_DATASET
            
        stsb_tokens_1, stsb_tokens_2, stsb_scores = prepare_dataset(stsb_path, tokenizer, cfg.max_length)
    else:
        tokens_1 = jnp.zeros((100, cfg.max_length), dtype=jnp.int32)
        tokens_2 = jnp.zeros((100, cfg.max_length), dtype=jnp.int32)
        scores = jnp.ones((100,), dtype=jnp.float32)
        val_tokens_1, val_tokens_2, val_scores = tokens_1, tokens_2, scores
        stsb_tokens_1, stsb_tokens_2, stsb_scores = tokens_1, tokens_2, scores

    batch_size = 32
    epochs = 10
    total_steps = (len(scores) // batch_size) * epochs
    
    optimizer = setup_optimizer(total_steps, cfg.learning_rate)
    opt_state = optimizer.init(params)
    
    print(f"Starting Standard BPTT (LSTM) training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = create_batches(tokens_1, tokens_2, scores, batch_size)
        num_batches = 0
        
        for batch in batches:
            b_tok_1, b_tok_2, b_scores = batch
            params, opt_state, loss = train_step(params, opt_state, b_tok_1, b_tok_2, b_scores, optimizer)
            epoch_loss += loss
            num_batches += 1
            
        avg_loss = epoch_loss / max(1, num_batches)
        
        pearson_val, _ = evaluate_metrics(params, val_tokens_1, val_tokens_2, val_scores, batch_size)
        pearson_stsb, spearman_stsb = evaluate_metrics(params, stsb_tokens_1, stsb_tokens_2, stsb_scores, batch_size)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - ALL-STS Pearson: {pearson_val:.4f} - STS-B Pearson: {pearson_stsb:.4f} - STS-B Spearman: {spearman_stsb:.4f}")
        
    print("Training complete! Parameters updated.")
    
    print("Saving LSTM weights to 'baseline_lstm_weights.json'...")
    np_params = jax.tree_util.tree_map(lambda x: np.array(x).tolist(), params)
    with open("baseline_lstm_weights.json", "w") as f:
        json.dump(np_params, f)
    print("Save complete!")

if __name__ == "__main__":
    main()

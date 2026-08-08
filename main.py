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

@dataclass
class Config:
    vocab_size: int = 32000
    d_model: int = 256
    max_length: int = 128
    beta_min: float = 1 - 2**-2
    beta_max: float = 1 - 2**-7
    theta_min: float = 0.001
    theta_max: float = 0.2
    surrogate_width: float = 1.0
    # Adam Optimizer meledak (NaN) jika menggunakan LR 2.0. Nilai 2.0 biasanya untuk SGD/Hebbian.
    # Untuk Optax Adam, learning rate standar adalah 1e-3 atau 2e-4.
    learning_rate: float = 1e-3
    membrane_max: float = 1.0
    pad_token_id: int = 0

# Inisialisasi secara global agar bisa diakses oleh SNN step (lif_step, spike_function)
cfg = Config()

def init_parameters(key, cfg: Config):

    keys = jax.random.split(key, 6)

    params = {}

    # Embedding Matrix
    emb_weight = (
        jax.random.normal(
            keys[0],
            (cfg.vocab_size, cfg.d_model)
        ) * 0.02
    )
    # Hyperpolarized Padding (set pad token embedding to -1.0)
    params["embedding_weight"] = emb_weight.at[cfg.pad_token_id].set(-1.0)

    # Embedding Leak
    params["embedding_beta"] = (
        jax.random.uniform(
            keys[1],
            (cfg.d_model,),
            minval=cfg.beta_min,
            maxval=cfg.beta_max
        )
    )

    # Embedding Threshold
    params["embedding_theta"] = jax.random.uniform(
        keys[2],
        (cfg.d_model,),
        minval=cfg.theta_min,
        maxval=cfg.theta_max,
    )

    # Pool Weight (Identity Initialization as per paper)
    params["pool_weight"] = jnp.eye(cfg.d_model)

    # Pool Leak
    params["pool_beta"] = (
        jax.random.uniform(
            keys[4],
            (cfg.d_model,),
            minval=cfg.beta_min,
            maxval=cfg.beta_max
        )
    )

    # Pool Threshold
    params["pool_theta"] = jax.random.uniform(
        keys[5],
        (cfg.d_model,),
        minval=cfg.theta_min,
        maxval=cfg.theta_max,
    )

    return params

# Configuration and keys will be initialized in main()

@jax.custom_gradient
def spike_function(x, theta):
    spike = (x >= theta).astype(x.dtype)

    def grad(dy):
        mask = (jnp.abs(x - theta) < cfg.surrogate_width).astype(x.dtype)
        dx = dy * mask
        return dx, None
        
    return spike, grad

def lif_step(v, x, beta, theta):
    pre = beta * v + x
    spike = spike_function(pre, theta)
    pre = jnp.minimum(pre, cfg.membrane_max)
    v = pre - spike * theta

    return v, spike

from jax import lax

def embedding_forward(params, token_ids):
    weight = params["embedding_weight"]
    return weight[token_ids]
    

def encoder_forward(params, embedding):
    def encoder_step(carry, x):
        v = carry
        v, spike = lif_step(
            v,
            x,
            params["embedding_beta"],
            params["embedding_theta"],
        )
        
        return v, spike
    
    # Membrane potential awal
    v = jnp.zeros(
        embedding.shape[-1],
        dtype=embedding.dtype,
    )
    v_final, spikes = lax.scan(encoder_step, v, embedding)

    return spikes


def pooler_forward(params, spikes):
    v0 = jnp.zeros(
        spikes.shape[-1],
        dtype=spikes.dtype,
    )
    
    def pool_step(carry, spike):
        v = carry
        # MENGATASI BOTTLENECK 40 MENIT (2342 detik):
        # Memaksa GPU memisahkan elemen demi elemen lalu menjumlahkan manual di JAX akan
        # menonaktifkan "Tensor Cores" (chip khusus akselerasi matriks pada GPU Tesla T4).
        # Walaupun input kita spars, operasi "dot product" jauh lebih dioptimasi di tingkat 
        # instruksi perakitan (assembly) GPU.
        x_pool = jnp.dot(spike, params["pool_weight"])

        v, pool_spike = lif_step(
            v,
            x_pool,
            params["pool_beta"],
            params["pool_theta"],
        )
        
        return v, v
        
    _, membrane_history = lax.scan(
        pool_step,
        v0,
        spikes,
    )

    embedding = jnp.sum(
        membrane_history,
        axis=0,
    )
    
    return embedding
    

# Tokenizer will be initialized in main()

def model_forward_single(params, token_ids):
    embedding = embedding_forward(
        params,
        token_ids,
    )
    
    spikes = encoder_forward(
        params,
        embedding,
    )
    
    sentence_embedding = pooler_forward(
        params,
        spikes,
    )
    
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

# Setup Dataset
DATASET = "/kaggle/input/datasets/akhyarsafrudin/train-distilation-dataset/teacher_distillation_dataset_scored.json"
VAL_DATASET = "/kaggle/input/datasets/akhyarsafrudin/all-sts-distilation/all_sts_teacher_scored.json"

def setup_optimizer(total_steps, base_lr=2.0):
    # Penjadwal Cosine Decay
    scheduler = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=total_steps,
        alpha=0.01 # Titik henti pada akhir epoch (1% dari base_lr)
    )
    
    # Bungkus dalam Adam optimizer
    optimizer = optax.adam(learning_rate=scheduler)
    return optimizer

@functools.partial(jax.jit, static_argnames=['optimizer'])
def train_step(params, opt_state, tokens_1, tokens_2, target_scores, optimizer):
    loss, grads = jax.value_and_grad(compute_loss)(
        params, 
        tokens_1, 
        tokens_2, 
        target_scores
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

def evaluate_pearson(params, tokens_1, tokens_2, target_scores, batch_size):
    num_samples = len(target_scores)
    all_preds = []
    
    for i in range(0, num_samples, batch_size):
        b_tok_1 = tokens_1[i:i + batch_size]
        b_tok_2 = tokens_2[i:i + batch_size]
        preds = predict_scores(params, b_tok_1, b_tok_2)
        all_preds.append(np.array(preds))
        
    all_preds = np.concatenate(all_preds)
    targets = np.array(target_scores)
    
    # Compute Pearson Correlation
    pred_mean = np.mean(all_preds)
    tgt_mean = np.mean(targets)
    
    cov = np.sum((all_preds - pred_mean) * (targets - tgt_mean))
    std_pred = np.sqrt(np.sum((all_preds - pred_mean)**2))
    std_tgt = np.sqrt(np.sum((targets - tgt_mean)**2))
    
    pearson_corr = cov / (std_pred * std_tgt + 1e-8)
    return pearson_corr

def prepare_dataset(file_path, tokenizer, max_length):
    print(f"Loading dataset from {file_path}...")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Dataset not found locally. Creating dummy data for testing...")
        data = [{"s1": "hello world", "s2": "hi earth", "score": 0.8}] * 100

    # Menangani variasi format key pada JSON dataset (misal: s1 vs sentence1)
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
    # Cek ketersediaan GPU JAX
    print("Available JAX devices:", jax.devices())
    if jax.devices()[0].device_kind.lower() == 'cpu':
        print("⚠️ WARNING: JAX berjalan di CPU! Pastikan Anda sudah mengaktifkan 'GPU T4x2' di menu Accelerator Kaggle.")
    else:
        print(f"✅ JAX berjalan menggunakan GPU: {jax.devices()[0].device_kind}")

    # Initialize tokenizer
    print("Initializing tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("PulseNet-Labs/spiking-sentence-embedder", trust_remote_code=True)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    except Exception as e:
        print(f"Could not load tokenizer from HuggingFace ({e}). Using dummy pad_token_id.")
        tokenizer = None
        pad_token_id = 0

    # Update pad_token_id global
    global cfg
    cfg.pad_token_id = pad_token_id
    key = jax.random.PRNGKey(42)
    
    print("Initializing parameters...")
    params = init_parameters(key, cfg)
    
    # Load dataset
    if tokenizer is not None:
        tokens_1, tokens_2, scores = prepare_dataset(DATASET, tokenizer, cfg.max_length)
        val_tokens_1, val_tokens_2, val_scores = prepare_dataset(VAL_DATASET, tokenizer, cfg.max_length)
    else:
        # Dummy data if no tokenizer
        tokens_1 = jnp.zeros((100, cfg.max_length), dtype=jnp.int32)
        tokens_2 = jnp.zeros((100, cfg.max_length), dtype=jnp.int32)
        scores = jnp.ones((100,), dtype=jnp.float32)
        
        val_tokens_1, val_tokens_2, val_scores = tokens_1, tokens_2, scores

    batch_size = 32
    epochs = 10
    total_steps = (len(scores) // batch_size) * epochs
    
    optimizer = setup_optimizer(total_steps, cfg.learning_rate)
    opt_state = optimizer.init(params)
    
    print(f"Starting training for {epochs} epochs...")
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
        
        # Evaluate Pearson Correlation
        pearson_val = evaluate_pearson(params, val_tokens_1, val_tokens_2, val_scores, batch_size)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Pearson: {pearson_val:.4f}")
        
    print("Training complete! Parameters updated.")

if __name__ == "__main__":
    main()


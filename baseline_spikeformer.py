import os
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
    learning_rate: float = 1e-3
    membrane_max: float = 1.0
    pad_token_id: int = 0
    # Specific to Spikeformer
    num_heads: int = 4

cfg = Config()

def init_parameters(key, cfg: Config):
    keys = jax.random.split(key, 10)
    params = {}
    # Embedding
    emb_weight = jax.random.normal(keys[0], (cfg.vocab_size, cfg.d_model)) * 0.02
    params["embedding_weight"] = emb_weight.at[cfg.pad_token_id].set(-1.0)
    params["embedding_beta"] = jax.random.uniform(keys[1], (cfg.d_model,), minval=cfg.beta_min, maxval=cfg.beta_max)
    params["embedding_theta"] = jax.random.uniform(keys[2], (cfg.d_model,), minval=cfg.theta_min, maxval=cfg.theta_max)

    # Spikeformer Attention projections
    params["q_weight"] = jax.random.normal(keys[3], (cfg.d_model, cfg.d_model)) * 0.02
    params["k_weight"] = jax.random.normal(keys[4], (cfg.d_model, cfg.d_model)) * 0.02
    params["v_weight"] = jax.random.normal(keys[5], (cfg.d_model, cfg.d_model)) * 0.02
    
    params["qkv_beta"] = jax.random.uniform(keys[6], (cfg.d_model,), minval=cfg.beta_min, maxval=cfg.beta_max)
    params["qkv_theta"] = jax.random.uniform(keys[7], (cfg.d_model,), minval=cfg.theta_min, maxval=cfg.theta_max)
    
    # Output projection
    params["o_weight"] = jax.random.normal(keys[8], (cfg.d_model, cfg.d_model)) * 0.02
    
    return params

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

def embedding_forward(params, token_ids):
    weight = params["embedding_weight"]
    return weight[token_ids]

def encoder_forward(params, embedding):
    def encoder_step(carry, x):
        v = carry
        v, spike = lif_step(v, x, params["embedding_beta"], params["embedding_theta"])
        return v, spike
    
    v = jnp.zeros(embedding.shape[-1], dtype=embedding.dtype)
    _, spikes = jax.lax.scan(encoder_step, v, embedding)
    return spikes

def spikeformer_attention(params, spikes):
    # Project to Q, K, V
    q_x = jnp.dot(spikes, params["q_weight"])
    k_x = jnp.dot(spikes, params["k_weight"])
    v_x = jnp.dot(spikes, params["v_weight"])
    
    # LIF layer to emit Q, K, V spikes
    def qkv_step(carry, inputs):
        vq, vk, vv = carry
        qx, kx, vx = inputs
        vq, q_spike = lif_step(vq, qx, params["qkv_beta"], params["qkv_theta"])
        vk, k_spike = lif_step(vk, kx, params["qkv_beta"], params["qkv_theta"])
        vv, v_spike = lif_step(vv, vx, params["qkv_beta"], params["qkv_theta"])
        return (vq, vk, vv), (q_spike, k_spike, v_spike)
    
    carry0 = (
        jnp.zeros(cfg.d_model, dtype=spikes.dtype),
        jnp.zeros(cfg.d_model, dtype=spikes.dtype),
        jnp.zeros(cfg.d_model, dtype=spikes.dtype)
    )
    _, (q_spikes, k_spikes, v_spikes) = jax.lax.scan(qkv_step, carry0, (q_x, k_x, v_x))
    
    # Spiking Self Attention
    scale = jnp.sqrt(cfg.d_model)
    attn_scores = jnp.dot(q_spikes, k_spikes.T) / scale
    
    # Threshold attention scores to create attention spikes
    attn_spikes, _ = spike_function(attn_scores, 0.5)
    
    # Gating V spikes
    out_spikes = jnp.dot(attn_spikes, v_spikes)
    
    # Project Output
    out = jnp.dot(out_spikes, params["o_weight"])
    return out

def model_forward_single(params, token_ids):
    embedding = embedding_forward(params, token_ids)
    spikes = encoder_forward(params, embedding)
    
    # Spikeformer baseline attention
    attn_out = spikeformer_attention(params, spikes)
    
    # Pooler (Average pooling for sequence embedding)
    sentence_embedding = jnp.mean(attn_out, axis=0)
    return sentence_embedding
    
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
    return jnp.mean(mse * margin)

DATASET = "/kaggle/input/datasets/akhyarsafrudin/train-distilation-dataset/teacher_distillation_dataset_scored.json"
VAL_DATASET = "/kaggle/input/datasets/akhyarsafrudin/all-sts-distilation/all_sts_teacher_scored.json"

def setup_optimizer(total_steps, base_lr=2.0):
    scheduler = optax.cosine_decay_schedule(init_value=base_lr, decay_steps=total_steps, alpha=0.01)
    return optax.adam(learning_rate=scheduler)

@functools.partial(jax.jit, static_argnames=['optimizer'])
def train_step(params, opt_state, tokens_1, tokens_2, target_scores, optimizer):
    loss, grads = jax.value_and_grad(compute_loss)(params, tokens_1, tokens_2, target_scores)
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
    
    pred_mean = np.mean(all_preds)
    tgt_mean = np.mean(targets)
    cov = np.sum((all_preds - pred_mean) * (targets - tgt_mean))
    std_pred = np.sqrt(np.sum((all_preds - pred_mean)**2))
    std_tgt = np.sqrt(np.sum((targets - tgt_mean)**2))
    return cov / (std_pred * std_tgt + 1e-8)

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
    encoded_s1 = tokenizer(s1_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
    encoded_s2 = tokenizer(s2_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")

    return jnp.array(encoded_s1["input_ids"]), jnp.array(encoded_s2["input_ids"]), jnp.array(scores, dtype=jnp.float32)

def create_batches(token_ids_1, token_ids_2, scores, batch_size):
    num_samples = len(scores)
    indices = np.random.permutation(num_samples)
    for i in range(0, num_samples, batch_size):
        batch_idx = indices[i:i + batch_size]
        yield token_ids_1[batch_idx], token_ids_2[batch_idx], scores[batch_idx]

def main():
    print("Available JAX devices:", jax.devices())
    try:
        tokenizer = AutoTokenizer.from_pretrained("PulseNet-Labs/spiking-sentence-embedder", trust_remote_code=True)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        tokenizer = None
        pad_token_id = 0

    global cfg
    cfg.pad_token_id = pad_token_id
    key = jax.random.PRNGKey(42)
    
    print("Initializing Spikeformer Baseline parameters...")
    params = init_parameters(key, cfg)
    
    if tokenizer is not None:
        tokens_1, tokens_2, scores = prepare_dataset(DATASET, tokenizer, cfg.max_length)
        val_tokens_1, val_tokens_2, val_scores = prepare_dataset(VAL_DATASET, tokenizer, cfg.max_length)
    else:
        tokens_1 = jnp.zeros((100, cfg.max_length), dtype=jnp.int32)
        tokens_2 = jnp.zeros((100, cfg.max_length), dtype=jnp.int32)
        scores = jnp.ones((100,), dtype=jnp.float32)
        val_tokens_1, val_tokens_2, val_scores = tokens_1, tokens_2, scores

    batch_size = 32
    epochs = 10
    total_steps = (len(scores) // batch_size) * epochs
    
    optimizer = setup_optimizer(total_steps, cfg.learning_rate)
    opt_state = optimizer.init(params)
    
    print(f"Starting Spikeformer baseline training for {epochs} epochs...")
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
        pearson_val = evaluate_pearson(params, val_tokens_1, val_tokens_2, val_scores, batch_size)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Pearson: {pearson_val:.4f}")

if __name__ == "__main__":
    main()

---
language:
- id
- en
pipeline_tag: feature-extraction
tags:
- spiking-neural-network
- snn
- sentence-similarity
- neuromorphic
- attention-free
- indonesian
license: apache-2.0
---

# PulseEmbed-v2: Attention-Free Spiking Sentence Embedder

This is the official PyTorch/HuggingFace implementation of **PulseEmbed-v2**, an extremely efficient, Attention-Free Spiking Neural Network (SNN) for Semantic Textual Similarity (STS). 

This version (V2) drastically improves upon previous baselines by completely dropping the quadratic spatial attention routing ($\mathcal{O}(L^2)$) in favor of a linear, hardware-friendly **Recurrent Pooler**. It was trained natively in Rust for absolute deterministic bit-exact parity and has been ported to PyTorch for standard deployment.

## Model Details
- **Architecture**: Attention-Free Spiking Neural Network (SNN) with Leaky-Integrate-and-Fire (LIF) neurons.
- **Dimensionality**: `d_model = 256`
- **Layers**: 
  - Token-level Temporal Embedding
  - **Attention-Free Recurrent Pooler** (Add-Only BPTT Dynamics)
- **Task**: Semantic Textual Similarity / Sentence Embeddings
- **Languages**: Indonesian (ID), English (EN)
- **Training Paradigm**: Knowledge Distillation from a continuous Transformer teacher via Mean Squared Error on Pearson Correlation scores over the ALL-STS dataset.

## Performance & Benchmarks
Tested strictly on out-of-domain Zero-Shot benchmarks (STS-12 to STS-16, STS-B, SICK-R) against the continuous ground-truth representation:
- **Pearson Correlation**: **0.8030** (Shatters the V1 baseline of 0.758).
- **Efficiency**: Achieves these results using exclusively sparse logical additions without any $Q \times K^T$ dense spatial attention multiplications.

### Evaluation Samples (PyTorch SNN vs Teacher)
Below are examples of how the Attention-Free SNN matches the continuous dense Teacher model:

```text
S1: A group of kids is playing in a yard and an old man is standing in the background
S2: A group of boys in a yard is playing and a man is standing in the background
PyTorch SNN Pred: 0.7320 | Target Guru: 0.8421

S1: A group of children is playing in the house and there is no man standing in the background
S2: A group of kids is playing in a yard and an old man is standing in the background
PyTorch SNN Pred: 0.6991 | Target Guru: 0.5353

S1: The young boys are playing outdoors and the man is smiling nearby
S2: The kids are playing outdoors near a man with a smile
PyTorch SNN Pred: 0.7287 | Target Guru: 0.8083

S1: The kids are playing outdoors near a man with a smile
S2: A group of kids is playing in a yard and an old man is standing in the background
PyTorch SNN Pred: 0.6313 | Target Guru: 0.6244

S1: The young boys are playing outdoors and the man is smiling nearby
S2: A group of kids is playing in a yard and an old man is standing in the background
PyTorch SNN Pred: 0.4549 | Target Guru: 0.5061

==> Hasil Akhir Pearson Correlation (PyTorch): 0.8030
```

## Usage

This model requires custom architecture code (`modeling_spiking.py`) to run. You must set `trust_remote_code=True` when loading the model.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 1. Load Tokenizer and Spiking Model
model_id = "PulseNet-Labs/spiking-sentence-embedder-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
model.eval()

# 2. Input sentences
sentences = [
    "Sistem neuromorfik ini sangat hemat energi.",
    "Jaringan saraf spiking mengonsumsi daya yang rendah tanpa attention."
]

# 3. Tokenize
inputs = tokenizer(sentences, padding="max_length", max_length=128, truncation=True, return_tensors="pt")

# Convert PAD tokens to 0 to align with SNN temporal initialization behavior
inputs.input_ids[inputs.input_ids == tokenizer.pad_token_id] = 0

# 4. Forward Pass (Temporal SNN Simulation via BPTT Pooler)
with torch.no_grad():
    embeddings = model(**inputs)

# 5. Compute Cosine Similarity
similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
print(f"Semantic Similarity: {similarity.item():.4f}")
```

## Citing & Authors
If you use this model in your research, please refer to the corresponding ICLR 2026 manuscript:
*"Is Spike-Driven Self-Attention Necessary? The Inefficiency of Spike-Overlap Attention in Spiking Sentence Embeddings"* by **Muhammad Akhyar**.

**Organization:** [PulseNet-Labs](https://huggingface.co/PulseNet-Labs)

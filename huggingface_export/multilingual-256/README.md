---
language:
- id
- en
- fr
- es
- zh
- ar
- ru
- de
- ja
- ko
pipeline_tag: feature-extraction
tags:
- spiking-neural-network
- snn
- sentence-similarity
- neuromorphic
- attention-free
- multilingual
license: apache-2.0
---

# PulseEmbed-Multilingual-v1: Attention-Free Spiking Sentence Embedder (256d)

This is the official PyTorch/HuggingFace implementation of **PulseEmbed-Multilingual-v1**, an extremely efficient, Attention-Free Spiking Neural Network (SNN) for Multilingual Semantic Textual Similarity (STS). 

This version extends the architecture to support multiple languages using a 64,000 vocabulary size while drastically improving upon previous baselines by completely dropping the quadratic spatial attention routing ($\mathcal{O}(L^2)$) in favor of a linear, hardware-friendly **Recurrent Pooler**. It was trained natively in Rust for absolute deterministic bit-exact parity and has been ported to PyTorch for standard deployment.

## Model Details
- **Architecture**: Attention-Free Spiking Neural Network (SNN) with Leaky-Integrate-and-Fire (LIF) neurons.
- **Dimensionality**: `d_model = 256`
- **Vocabulary Size**: `64,000` (Multilingual BPE)
- **Layers**: 
  - Token-level Temporal Embedding
  - **Attention-Free Recurrent Pooler** (Add-Only BPTT Dynamics)
- **Task**: Semantic Textual Similarity / Sentence Embeddings
- **Languages**: 20+ Languages (including Indonesian, English, Spanish, French, etc.)
- **Training Paradigm**: Knowledge Distillation from a continuous Multilingual Transformer teacher via Mean Squared Error on Pearson Correlation scores over the Multilingual ALL-STS dataset.

## Usage

This model requires custom architecture code (`modeling_spiking.py`) to run. You must set `trust_remote_code=True` when loading the model.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 1. Load Tokenizer and Spiking Model
model_id = "PulseNet-Labs/spiking-sentence-embedder-multilingual-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
model.eval()

# 2. Input sentences
sentences = [
    "Sistem neuromorfik ini sangat hemat energi.",
    "This neuromorphic system is very energy efficient."
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
If you use this model in your research, please refer to the corresponding TMLR manuscript:
*"Is Spike-Driven Self-Attention Necessary? The Inefficiency of Spike-Overlap Attention in Spiking Sentence Embeddings"* by **Muhammad Akhyar**.

**Organization:** [PulseNet-Labs](https://huggingface.co/PulseNet-Labs)

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
- hebbian-learning
- indonesian
license: apache-2.0
---

# Spiking Sentence Embedder (PulseNet-Labs)

This is the official PyTorch/HuggingFace implementation of the **Spiking Sentence Embedder** featuring **Sparse Coincidence-Based Semantic Attention**. The model was originally implemented in Rust to simulate true neuromorphic hardware constraints and has been carefully ported to PyTorch to guarantee 100% mathematical bit-exact parity for seamless deployment.

## Model Details
- **Architecture**: Spiking Neural Network (SNN) with Leaky-Integrate-and-Fire (LIF) neurons.
- **Layers**: 
  - Token-level Temporal Embedding
  - Coincidence-Based Semantic Attention
  - Dense BPTT Pooler with residual dynamics
- **Task**: Semantic Textual Similarity / Sentence Embeddings
- **Languages**: Indonesian (ID), English (EN)
- **Training Paradigm**: Knowledge Distillation via Hebbian Plasticity (Contrastive Hebbian Learning) from a teacher model.
- **Publication / DOI**: [10.5281/zenodo.20739462](https://doi.org/10.5281/zenodo.20739462)

## Usage

This model requires custom architecture code (`modeling_spiking.py`) to run. You must set `trust_remote_code=True` when loading the model.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 1. Load Tokenizer and Spiking Model
tokenizer = AutoTokenizer.from_pretrained("PulseNet-Labs/spiking-sentence-embedder", trust_remote_code=True)
model = AutoModel.from_pretrained("PulseNet-Labs/spiking-sentence-embedder", trust_remote_code=True)
model.eval()

# 2. Input sentences
sentences = [
    "Sistem neuromorfik ini sangat hemat energi.",
    "Jaringan saraf spiking mengonsumsi daya yang rendah."
]

# 3. Tokenize
inputs = tokenizer(sentences, padding="max_length", max_length=128, truncation=True, return_tensors="pt")

# Convert PAD tokens to 0 to align with SNN initialization behavior
inputs.input_ids[inputs.input_ids == tokenizer.pad_token_id] = 0

# 4. Forward Pass (Temporal SNN Simulation)
with torch.no_grad():
    embeddings = model(**inputs)

# 5. Compute Pearson/Cosine Similarity
# Note: For strict SNN metric space validation, mean-centering is recommended
emb_centered = embeddings - embeddings.mean(dim=-1, keepdim=True)
similarity = F.cosine_similarity(emb_centered[0].unsqueeze(0), emb_centered[1].unsqueeze(0))

print(f"Semantic Similarity: {similarity.item():.4f}")
```

## Performance & Benchmarks
Tested on the bilingual STS-B dataset:
- **Pearson Correlation (vs Teacher)**: 0.7514
- Achieves zero-shot generalization with highly sparse binary activations, drastically reducing theoretical energy consumption compared to dense attention counterparts.

## Citing & Authors
If you use this model in your research, please refer to our DOI manuscript: [https://doi.org/10.5281/zenodo.20739462](https://doi.org/10.5281/zenodo.20739462).
**Organization:** [PulseNet-Labs](https://huggingface.co/PulseNet-Labs)

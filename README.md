# SpikingNetworkRust: Biologically Plausible Sentence Embedders

`SpikingNetworkRust` is a high-performance Rust implementation of a Spiking Neural Network (SNN) tailored specifically for Natural Language Processing (NLP) tasks, particularly **Sentence Embeddings**. It bridges the gap between biologically plausible neural computing and modern semantic representation models, targeting the ICLR 2026 conference.

## 🚀 Key Features

*   **Spiking Architecture:** Replaces continuous floating-point activations with discrete, binary spikes (`0` or `1`) across time steps, mimicking the brain's biological neurons (Leaky Integrate-and-Fire / LIF model).
*   **Zero-Copy Memory & Sparse Tensors:** Optimized for CPU cache layouts using SoA (Struct of Arrays) to maintain real-time training and inference speeds natively in Rust.
*   **Dual Learning Paradigms:**
    *   **Unsupervised SimCSE:** Leverages dropout-based contrastive learning to push separate semantic instances apart and pull identical ones together.
    *   **Supervised Hebbian Distillation:** Transfers semantic topology from state-of-the-art Transformer models (e.g., `all-MiniLM-L6-v2`) to the SNN using a target-score modulated Hebbian push/pull learning rule.

## 🧠 Training Methodologies

Currently, this repository features multiple training binaries targeting different learning paradigms:

### 1. Unsupervised Contrastive Learning
The SNN learns semantic representations directly from raw text corpus without human labels. 
- **Method:** Passing the same sentence twice with different random dropout masks. The SNN uses Contrastive Hebbian Learning to pull the augmented views closer while pushing apart negative samples.
- **Results:** Achieved a Pearson Correlation of **0.525** on STS-B under strictly controlled initializations.

### 2. Knowledge Distillation (Distil AI)
Trains the SNN by mimicking a "Teacher" transformer model (`all-MiniLM-L6-v2`).
- **Method:** Generating a distillation dataset that combines over 29,000 synthetic sentence pairs with clamped Cosine Similarity scores from the Teacher. The SNN uses this similarity score as a coefficient to pull or push its spike representations, guided by a strict error tolerance margin ($m=0.05$) and Cosine Annealing learning rate.
- **Results:** Evaluated on the STS-B dataset via `full_eval_controlled` binary (ensuring identical weight initialization), the distilled SNN achieved a Pearson Correlation of **0.691** (vs Human) and **0.751** (vs Teacher).

### 3. Controlled Architectural Ablation & Energy Efficiency
A comprehensive, seeded evaluation pipeline to benchmark various training strategies, sequence contexts, and biological sensitivity parameters.
- **Method:** Training all configurations using identical seeded initialization weights to eliminate bias (`experiment/file_model/init_weights.json`).
- **Results (384d, T=128):**
  - **Homogeneous LIF Dynamics:** Forcing uniform neuron parameters ($\beta=0.90$) surprisingly improved the continuous semantic topology tracking, achieving a Teacher Pearson of **0.758** and yielding extreme computational sparsity.
  - **Energy Efficiency:** The Homogeneous Distillation model emits only ~2,494 spikes per sentence, requiring just **~1.9 Million Add-only SOPs** compared to a Transformer's **~1.43 Billion MACs**, achieving an approximate **748x** theoretical computational sparsity reduction.

## 🛠 Reproducibility

Please refer to [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for detailed instructions on generating weights, downloading the distillation dataset, and reproducing the 748x efficiency benchmarks and 0.758 Pearson correlation reported in our ICLR 2026 manuscript.

## 📈 Out-of-Domain Generalization

Our SNN embedding architecture maintains robust cross-domain continuous representation capabilities. When zero-shot evaluated on standardized external benchmarks (MTEB), the SNN maintained strong semantic alignment:
- STS-12: 0.464
- STS-14: 0.476
- SICK-R: 0.485

## License
MIT License

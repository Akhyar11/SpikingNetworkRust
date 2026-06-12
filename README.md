# SpikingNetworkRust: Biologically Plausible Sentence Embedders

`SpikingNetworkRust` is a high-performance Rust implementation of a Spiking Neural Network (SNN) tailored specifically for Natural Language Processing (NLP) tasks, particularly **Sentence Embeddings**. It bridges the gap between biologically plausible neural computing and modern semantic representation models.

## 🚀 Key Features

*   **Spiking Architecture:** Replaces continuous floating-point activations with discrete, binary spikes (`0` or `1`) across time steps, mimicking the brain's biological neurons (Leaky Integrate-and-Fire / LIF model).
*   **Zero-Copy Memory & Sparse Tensors:** Optimized for CPU cache layouts using SoA (Struct of Arrays) to maintain real-time training and inference speeds natively in Rust.
*   **Dual Learning Paradigms:**
    *   **Unsupervised SimCSE:** Leverages dropout-based contrastive learning to push separate semantic instances apart and pull identical ones together.
    *   **Supervised Hebbian Distillation:** Transfers semantic topology from state-of-the-art Transformer models (e.g., `MiniLM-L6-v2`) to the SNN using a target-score modulated Hebbian push/pull learning rule.

## 🧠 Training Methodologies

Currently, this repository features multiple training binaries targeting different learning paradigms:

### 1. Unsupervised Contrastive Learning (`train_simcse`)
*(Currently active training method)*
The SNN learns semantic representations directly from raw text corpus (`mini_corpus.txt` - Wikipedia ID & EN) without human labels. 
- **Method:** Passing the same sentence twice with different random dropout masks. The SNN uses Contrastive Hebbian Learning to pull the augmented views closer while pushing apart negative samples (CutMix / Lexical Swaps).
- **Latency Target:** Sub-millisecond inference per sentence.
- **Results:** Achieved a Pearson Correlation of **0.4923** on STS-B under strictly controlled initializations.

### 2. Knowledge Distillation (`train_distillation`)
Trains the SNN by mimicking a "Teacher" transformer model (`MiniLM-L6-v2`).
- **Method:** Generating over 100,000 sentence pairs and asking the Teacher for their Cosine Similarity. The SNN uses this similarity score as a coefficient to pull or push its spike representations.
- **Results:** Evaluated on the STS-B dataset via `full_eval_controlled` binary (ensuring identical weight initialization), the distilled SNN achieved a Pearson Correlation of **0.6171**, proving that dense vector space guidance significantly improves SNN linguistic topology.

### 3. Pure Human Annotations (`train_human_only`)
An experiment to test SNN generalization on a small, highly accurate dataset.
- **Method:** Training purely on 14,740 manually annotated sentence pairs from STS-B (English and Indonesian).
- **Results:** Achieved a Pearson Correlation of **0.5924**. This experiment empirically demonstrated that SNNs benefit far more from large-scale machine-distilled continuous targets than from sparse binary human annotations.

### 4. Controlled Architectural Ablation & Energy Efficiency (`full_eval_controlled`)
A comprehensive, seeded evaluation pipeline to benchmark various training strategies, sequence contexts, and biological sensitivity parameters.
- **Method:** Training all configurations using identical seeded initialization weights to eliminate bias (`experiment/file_model/init_weights.json`).
- **Results (STS-B):**
  - **Longer Context (T=64):** Achieved the highest semantic correlation of **0.6299**.
  - **Biological Sensitivity:** Forcing uniform neuron parameters (Homogeneous LIF, $\beta=0.90$) significantly dropped correlation by ~15% (from 0.6171 to **0.5212**), proving the necessity of heterogeneous biological timescales.
  - **Energy Efficiency:** The baseline Distillation model emits only ~504 spikes per sentence, requiring just **~64,587 Add-only SOPs** compared to a Transformer's **~10.2 Million MACs**, achieving an approximate **158x** theoretical computational sparsity reduction.

## 🛠 Usage

To evaluate the current model against the STS-B valid dataset:
```bash
cargo run --release --bin eval
```

To resume training using the Unsupervised SimCSE pipeline:
```bash
cargo run --release --bin train_simcse
```

To run a direct side-by-side comparison against the MiniLM Teacher (requires Node.js wrapper):
```bash
npx tsx evaluate_snn_minilm.js
```

## 📈 Evaluation Examples

**Sample 1:** Sensitivity to Lexical Overlap
* A: "A woman is playing the guitar."
* B: "A man is playing guitar."
* Actual Human Score: `0.4800`
* SNN Prediction: `0.7036` (distilled) / `0.8693` (human-only)

## License
MIT License

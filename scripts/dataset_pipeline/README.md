# Dataset Generation Pipeline

This directory contains Node.js scripts used to generate the datasets for the Spiking Neural Network training process. This ensures full provenance and reproducibility of the data directly from the source repository.

## Setup

These scripts require Node.js. Install the necessary dependencies before running:
```bash
npm install
```
This will install `@xenova/transformers` required for the knowledge distillation pipeline.

## Scripts

### 1. `generate_teacher_dataset.js`
Generates the `teacher_distillation_dataset.json` used for training the network via knowledge distillation.
- **Teacher Model**: `Xenova/all-MiniLM-L6-v2`
- **Process**: Reads from `../../experiment/file_model/mini_corpus20mb.txt` and generates positive/negative sentence pairs using techniques like Lexical Swap and CutMix. The pairs are then evaluated using the teacher model to get target cosine similarity scores.
- **Run**: `node generate_teacher_dataset.js`

### 2. `create_human_only_dataset.js`
Generates the `human_only_dataset.json`.
- **Process**: Merges the pure English STS-B (`sts-b_train.json`) with an Indonesian augmented/translated dataset (`data_stsb.train.modified_indo.csv`). This results in a bilingual training dataset.
- **Run**: `node create_human_only_dataset.js`

## Outputs
All generated JSON outputs are written directly to `../../experiment/file_model/` so they can be consumed by the Rust evaluation and training pipeline.

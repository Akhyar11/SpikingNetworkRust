# Reproducibility Guide

## Prerequisites
- Rust 1.70+
- Ubuntu 20.04+ (or macOS 12+)

## Step 1: Generate Initial Weights
```bash
cargo run --release --bin generate_init_weights
```
Produces: `experiment/file_model/init_weights.json`

## Step 2: Download Teacher Dataset
Expected: `experiment/file_model/teacher_distillation_dataset_scored.json`
Count: 29,720 pairs

## Step 3: Train with Seeded Initialization
```bash
cargo run --release --bin full_eval_controlled 2>&1 | tee training.log
```
*(Note: `full_eval_controlled` is the unified training and evaluation orchestrator in this project).*

## Expected Results (±0.005 variance due to floating-point):
- Pearson (STS-B): 0.691 ± 0.005
- Teacher Correlation: 0.758 ± 0.005
- SOPs per sentence: ~3,064,916 ± 10% (Heterogeneous) and ~1,915,477 (Homogeneous)

## Debugging
If results don't match:
1. Verify seed value (should be 42)
2. Check learning rate schedule applied (Cosine Annealing is enabled in `full_eval_controlled.rs`)
3. Verify weight clipping `[-1, 1]` is active
4. Check margin tolerance (0.05) in `contrastiveHebbian`

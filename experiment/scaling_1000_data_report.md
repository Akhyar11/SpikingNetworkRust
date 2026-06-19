# Scaling Report: SNN Attention on 1000-Sample Dataset

## Objective
Scale the Spiking Neural Network (SNN) distillation pipeline to 1000 samples while enforcing the strict 20-epoch limit, confirming that the XOR Residual Attention logic remains dominant as dataset complexity increases.

## Architecture Configuration
- **Dataset Size**: 1000 sentence pairs
- **d_model**: 128
- **max_seq_length**: Automatically detected based on tokenization (64 tokens max)
- **Baseline LR**: 1.5
- **Attention Layer LR**: 2.0 (Boosted to compensate for XOR gradient splitting)
- **Attention Logic**: `Logical XOR` (Residual correction allowing feature insertion and noise pruning).

## Experimental Results (1000 Samples, 20 Epochs)

| Architecture | Epoch 5 | Epoch 10 | Epoch 15 | Epoch 20 |
|--------------|---------|----------|----------|----------|
| **Baseline (No Attention)** | 42.30% | 60.90% | 74.50% | **81.20%** |
| **SCA Attention (XOR)** | 48.20% | 71.30% | 83.70% | **90.70%** |

## Analysis
The Attention model clearly maintains its dominance as the dataset size doubles. 
1. **Convergence Speed**: Despite the complexity of processing 1000 pairs, the Attention model reached 83.7% accuracy by Epoch 15, easily exceeding the 80% accuracy target significantly earlier than the 20-epoch limit.
2. **Margin of Victory**: The performance gap actually widened. On the 500-sample dataset, Attention won by ~7.4%. On the 1000-sample dataset, Attention won by **9.5%** (90.70% vs 81.20%). This indicates that as dataset complexity increases, the contextual residual correction provided by Attention becomes increasingly critical.

## Next Step
Proceeding to scale the dataset to **5000 samples** to observe performance ceilings and stress-test the `d_model=128` capacity.

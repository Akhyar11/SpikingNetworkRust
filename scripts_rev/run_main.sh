#!/bin/bash
# Main runs: D=256 attention-free LIF model, decontaminated distillation data, seeds 1-3
cd "$(dirname "$0")/.."
for s in 1 2 3; do
  ./target/release/train_export --d 256 --seed $s --epochs 10 --data data_eval/train_decontam.json --out results/snn_d256_s$s > results/snn_d256_s$s.log 2>&1
done

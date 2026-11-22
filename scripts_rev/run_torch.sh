#!/bin/bash
cd "$(dirname "$0")/.."
for s in 1 2 3; do for v in meanpool ann_twin snn_sg snn_sg_beta gru; do
  python3 scripts_rev/torch_models.py --variant $v --seed $s --epochs 2 --threads 3 --out results/${v}_s$s > results/${v}_s$s.log 2>&1
done; done

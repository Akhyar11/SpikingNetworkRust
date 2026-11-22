#!/bin/bash
# E6: fair CPU speed benchmark, machine otherwise idle. 100 train steps x 32 pairs (64 sentences), 3 repeats.
cd "$(dirname "$0")/.."
for rep in 1 2 3; do
 for th in 1 8; do
  for dense in 0 1; do
    if [ $dense = 1 ]; then export DENSE_POOLER=1; else unset DENSE_POOLER; fi
    RAYON_NUM_THREADS=$th ./target/release/train_export --d 256 --seed $rep --epochs 1 --data results/bench/small3200.json --out results/bench/rust_t${th}_dense${dense}_r$rep > /dev/null 2>&1
  done
  unset DENSE_POOLER
  for v in meanpool snn_sg gru lstm; do
    python3 scripts_rev/torch_models.py --variant $v --seed $rep --epochs 1 --bs 32 --max_steps 100 --threads $th --out results/bench/${v}_t${th}_r$rep > /dev/null 2>&1
  done
 done
done
echo finished > results/bench/DONE

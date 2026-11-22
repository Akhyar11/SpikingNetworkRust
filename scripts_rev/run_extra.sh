#!/bin/bash
# E5 (human-label regression on STS-B train) x3 seeds, then E4 leaky-data replication (original 250K set, seed 1)
cd "$(dirname "$0")/.."
for s in 1 2 3; do
  ./target/release/train_export --d 256 --seed $s --epochs 10 --data data_eval/train_stsb_human.json --out results/snn_d256_human_s$s > results/snn_d256_human_s$s.log 2>&1
done
while pgrep -f run_main.sh > /dev/null; do sleep 30; done
./target/release/train_export --d 256 --seed 1 --epochs 10 --data experiment/file_model/teacher_distillation_dataset_scored.json --out results/snn_d256_leaky_s1 > results/snn_d256_leaky_s1.log 2>&1

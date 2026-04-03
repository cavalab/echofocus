#!/bin/bash
#SBATCH --account=chip-lacava
#SBATCH --partition=gpu-chip-lacava,bch-gpu,bch-gpu-pe
#SBATCH --gres=gpu:large:1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=echofocus_measure_multiquery
#SBATCH --output=slurm_logs/%j_echofocus_measure_multiquery.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=600G
#SBATCH --qos=normal

hostname
set -euo pipefail

# Hyperparameters matched to trained_models/EchoFocus_Measure/train_args.csv:
# batch_number=32, learning_rate=1e-5, seed=0, encoder_depth=5,
# parallel_processes=8, clip_dropout=0.5, epoch_early_stop=10.
# train_args.csv records epoch_lim=-1, which is eval-only; using total_epochs=12
# to match the saved checkpoint perf_log length.
uv run echofocus train \
  --model_name=EchoFocus_Measure_MultiQuery \
  --config=config.json \
  --dataset=bch_internal \
  --task=measure \
  --transformer_type=multiquery \
  --end_to_end=False \
  --batch_number=32 \
  --learning_rate=0.00001 \
  --seed=0 \
  --encoder_depth=5 \
  --parallel_processes=8 \
  --clip_dropout=0.5 \
  --epoch_early_stop=10 \
  --total_epochs=12 \
  --gpu_monitor \
  --ram_monitor

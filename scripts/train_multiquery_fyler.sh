#!/bin/bash
#SBATCH --account=chip-lacava
#SBATCH --partition=gpu-chip-lacava,bch-gpu,bch-gpu-pe
#SBATCH --gres=gpu:large:1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=echofocus_fyler_multiquery
#SBATCH --output=slurm_logs/%j_echofocus_fyler_multiquery.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=600G
#SBATCH --qos=normal

hostname
set -euo pipefail

# Hyperparameters matched to trained_models/EchoFocus_Fyler/train_args.csv:
# Batch_Number=512, learning_rate=1e-4, Rand_Seed=0, encoder_depth=10,
# parallel_count=8, clip_dropout=0.5, early_stop=10, epoch_end=20.
uv run echofocus train \
  --model_name=EchoFocus_Fyler_MultiQuery \
  --config=config.json \
  --dataset=bch_internal \
  --task=fyler \
  --transformer_type=multiquery \
  --end_to_end=False \
  --batch_number=512 \
  --learning_rate=0.0001 \
  --seed=0 \
  --encoder_depth=10 \
  --parallel_processes=8 \
  --clip_dropout=0.5 \
  --epoch_early_stop=10 \
  --total_epochs=20 \
  --gpu_monitor \
  --ram_monitor

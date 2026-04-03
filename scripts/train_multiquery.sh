#!/bin/bash
#SBATCH --account=chip-lacava
#SBATCH --partition=gpu-chip-lacava
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=echofocus_chd_multiquery
#SBATCH --output=slurm_logs/%j_echofocus_chd_multiquery.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=600G
#SBATCH --qos=unlimited

hostname
set -euo pipefail

tmppath="/temp_work/ch226520"
export TMPDIR=${tmppath}
export TMP=${tmppath}
export TEMP=${tmppath}
export TORCH_SHM_DIR=/temp_work/ch226520/torch-shm
mkdir -p "${TORCH_SHM_DIR}"

# Hyperparameters matched to trained_models/EchoFocus_CHD/train_args.csv:
# Batch_Number=128, learning_rate=1e-4, Rand_Seed=0, encoder_depth=1,
# parallel_count=8, clip_dropout=0.5, early_stop=10, epoch_end=30.
uv run echofocus train \
  --model_name=EchoFocus_CHD_MultiQuery \
  --config=config.json \
  --dataset=bch_internal \
  --task=chd \
  --transformer_type=multiquery \
  --end_to_end=False \
  --batch_number=128 \
  --learning_rate=0.0001 \
  --seed=0 \
  --encoder_depth=1 \
  --parallel_processes=8 \
  --clip_dropout=0.5 \
  --epoch_early_stop=10 \
  --total_epochs=30 \
  --gpu_monitor \
  --ram_monitor

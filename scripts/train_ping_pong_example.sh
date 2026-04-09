#!/bin/bash
#SBATCH --account=chip-lacava
#SBATCH --partition=gpu-chip-lacava,bch-gpu,bch-gpu-pe
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=echofocus_e2e_chd_pingpong
#SBATCH --output=slurm_logs/%j_echofocus_e2e_chd_pingpong.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=600G
#SBATCH --qos=normal
hostname
#!/usr/bin/env bash
set -euo pipefail

# Ping-pong end-to-end (raw clips)
uv run echofocus train_ping_pong \
  --model_name=chd_e2e_ping_pong_260405 \
  --load_transformer_path "./trained_models/EchoFocus_CHD/best_checkpoint.pt" \
  --config=config.json \
  --dataset=bch_internal \
  --task=chd \
  --end_to_end=True \
  --cache_panecho_embeddings=True \
  --num_clips=1 \
  --clip_len=16 \
  --total_epochs=100 \
  --amp=True \
  --checkpoint_panecho \
  --use_hdf5_index \
  --parallel_processes 4 \
  --panecho_lr_ratio 0.1 \
  --gpu_monitor \
  --ram_monitor \
  --max_videos_per_study 50

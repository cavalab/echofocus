#!/bin/bash
#SBATCH --account=chip-lacava
#SBATCH --partition=gpu-chip-lacava
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=echofocus_e2e_chd_cache
#SBATCH --output=slurm_logs/%j_echofocus_e2e_chd_cache.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --qos=unlimited
hostname
#!/usr/bin/env bash
set -euo pipefail

# Ping-pong end-to-end (raw clips)
uv run echofocus evaluate \
  --model_name=chd_e2e_ping_pong_260402 \
  --load_transformer_path "./trained_models/EchoFocus_CHD/best_checkpoint.pt" \
  --config=config.json \
  --dataset=outside \
  --split "(0,0,100)" \
  --task=chd \
  --end_to_end=True \
  --cache_panecho_embeddings=True \
  --num_clips=4 \
  --clip_len=16 \
  --total_epochs=-1 \
  --amp=True \
  --checkpoint_panecho \
  --use_hdf5_index \
  --parallel_processes 4 \
  --gpu_monitor \
  --ram_monitor \
  --folds "('test',)"
  # --max_videos_per_study 10 \
  # --sample_limit 100

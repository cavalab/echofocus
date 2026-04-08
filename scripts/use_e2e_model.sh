#!/bin/bash
#SBATCH --account=chip-lacava
#SBATCH --partition=gpu-chip-lacava,bch-gpu,bch-gpu-pe
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=echofocus_e2e_chd_eval
#SBATCH --output=slurm_logs/%j_echofocus_e2e_chd_eval.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --qos=normal
hostname
#!/usr/bin/env bash
set -euo pipefail

# Ping-pong end-to-end (raw clips)
# --load_transformer_path "./trained_models/EchoFocus_CHD/best_checkpoint.pt" \
  # --model_name=chd_e2e_ping_pong_2600403 \
  #--end_to_end True \
uv run echofocus evaluate \
  --model_name EchoFocus_CHD \
  --config=config.json \
  --dataset=bch_internal \
  --split "(64,16,20)" \
  --task=chd \
  --cache_panecho_embeddings=True \
  --num_clips=16 \
  --clip_len=16 \
  --total_epochs=-1 \
  --amp=True \
  --checkpoint_panecho \
  --use_hdf5_index \
  --parallel_processes 4 \
  --gpu_monitor \
  --ram_monitor \
  --folds "('test',)" #\
  # --max_videos_per_study 25 
  # --sample_limit 100

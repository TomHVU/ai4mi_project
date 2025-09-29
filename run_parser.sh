#!/bin/bash
#SBATCH -n 4
#SBATCH --ntasks 16
#SBATCH --mem=32G
#SBATCH -p gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm/training_1.out

conda activate ai4mi

mkdir -p data/SEGTHOR_tmp/train/img
mkdir data/SEGTHOR_tmp/train/gt
mkdir -p data/SEGTHOR_tmp/val/img
mkdir data/SEGTHOR_tmp/val/gt

python slice_segthor.py \
--source_dir "data/segthor_train" \
--dest_dir "data/SEGTHOR" \
--retains 10 \
--seed 42

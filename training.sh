#!/bin/bash
#SBATCH -n 4
#SBATCH --ntasks 16
#SBATCH --mem=32G
#SBATCH -p gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/scur0607/ai4mi_project/slurm/training.out
#SBATCH --chdir=/home/scur0607/ai4mi_project
#SBATCH --time=02:00:00

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh

# Loading module
conda activate ai4mi

# Run script
python main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest results/SEGTHOR/ce --gpu

# python main.py \
# --epochs 25 \
# --dataset SEGTHOR_CLEAN \
# --mode full \
# --dest results/SEGTHOR/ce \
# --gpu
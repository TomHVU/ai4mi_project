#!/bin/bash
#SBATCH -n 4
#SBATCH --ntasks 16
#SBATCH --mem=32G
#SBATCH -p gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm/training_1.out


# Loading module
conda activate ai4ml

python main.py \
--epochs 3 \
--dataset SEGTHOR_CLEAN \
--mode partial \
--dest results/ \
--gpu
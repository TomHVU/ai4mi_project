#!/bin/bash
#SBATCH -n 4
#SBATCH --ntasks 16
#SBATCH --mem=32G
#SBATCH -p gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm/training_1.out


# Loading module
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/.ai4ml/bin/activate

python3.11 main.py \
--epochs 3 \
--dataset SEGTHOR_CLEAN \
--mode partial \
--dest results/ \
--gpu
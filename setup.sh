#!/bin/bash
#SBATCH -J vggt_setup
#SBATCH --mem=40GB           # Critical: Increase memory for the compiler
#SBATCH --gres=gpu:1 -C gmem12
#SBATCH --time=02:00:00      # Increase time to 2 hours
#SBATCH --output=runs/setup_job.out
# Ensure this matches your Torch cu128 setup

module load anaconda3
module load cuda/11.4

export PATH="/home/de575594/.conda/envs/vggt/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate vggt 

pip install -r requirements.txt
#!/bin/bash
#SBATCH -J frames_test
#SBATCH --mem=24GB
#SBATCH --output=runs/frames_test_%j.out
export PATH="/home/de575594/.conda/envs/sip_plus_rl/bin:$PATH"
eval "$(conda shell.bash hook)"
module purge
module load anaconda3
module load cuda/12.4
module load gcc/9

conda activate sip_plus_rl

# Set percent of videos to process (e.g. 5 for 5%, 100 for all)
PERCENT=${PERCENT:-100}

CUDA_LAUNCH_BLOCKING=1 python create_frames_test.py --percent "$PERCENT" "$@"
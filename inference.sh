#!/bin/bash
#SBATCH -J vggt_inference
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1 -C gmem24
#SBATCH --time=04:00:00
#SBATCH --output=runs/inference_job_%j.out

# ==== CONFIGURATION ====
INPUT_DIR="/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/val/extracted_clips"
OUTPUT_DIR="/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/vggt-output"
MAX_VIDEOS=40
# =======================

module load anaconda3
module load cuda/11.4
export PATH="/home/de575594/.conda/envs/vggt/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate vggt

echo "================================================"
echo "Starting VGGT inference"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Max videos: $MAX_VIDEOS"
echo "Start time: $(date)"
echo "================================================"

# Run inference with max_videos parameter
python inference_vggt.py "$INPUT_DIR" "$OUTPUT_DIR" "$MAX_VIDEOS"

echo ""
echo "================================================"
echo "Job complete!"
echo "End time: $(date)"
echo "================================================"
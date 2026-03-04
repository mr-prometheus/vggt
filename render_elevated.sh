#!/bin/bash
#SBATCH -J render_elevated
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=runs/render_elevated_%j.out

# ==== CONFIGURATION ====
OUTPUT_DIR="/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/vggt-output-train"
# =======================

module load anaconda3
export PATH="/home/de575594/.conda/envs/vggt/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate vggt

echo "================================================"
echo "Rendering elevated views from point clouds"
echo "Output dir: $OUTPUT_DIR"
echo "Start time: $(date)"
echo "================================================"

python render_elevated_views.py "$OUTPUT_DIR"

echo ""
echo "================================================"
echo "Job complete!"
echo "End time: $(date)"
echo "================================================"

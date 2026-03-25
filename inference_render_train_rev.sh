#!/bin/bash
#SBATCH -J vggt_infer_train_rev
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1 -C gmem24
#SBATCH --output=runs/inference_render_train_rev_%j.out

# ==== CONFIGURATION ====
VIDEO_DIR="/home/c3-0/datasets/BDD_Dataset/Videos/bdd100k/videos/train/"
FILENAME_LIST="gama_list/train_day.list"
TEMP_FRAMES_DIR="extracted_clips_train_tmp_rev"   # separate dir to avoid collision
OUTPUT_DIR="/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/vggt-output-train-rendered"
PROGRESS_CSV="runs/inference_progress_train.csv"
CSV_LOCK="runs/inference_progress_train.csv.lock"  # shared lock file
FORCE_RECOMPUTE=false   # set to true to recompute clips that already have rendered outputs
# =======================

export PATH="/home/de575594/.conda/envs/vggt/bin:$PATH"
eval "$(conda shell.bash hook)"
module purge
module load anaconda3
module load cuda/11.4

conda activate vggt

mkdir -p runs
mkdir -p "$TEMP_FRAMES_DIR"

FORCE_FLAG=""
FORCE_PY="False"
if [ "$FORCE_RECOMPUTE" = true ]; then
    FORCE_FLAG="--force"
    FORCE_PY="True"
fi

echo "===================================================="
echo "VGGT Inference + Render Pipeline (TRAIN — REVERSED)"
echo "Video source : $VIDEO_DIR"
echo "List         : $FILENAME_LIST (processed in reverse)"
echo "Temp frames  : $TEMP_FRAMES_DIR"
echo "Output       : $OUTPUT_DIR"
echo "Progress CSV : $PROGRESS_CSV"
echo "Lock file    : $CSV_LOCK"
echo "Force        : $FORCE_RECOMPUTE"
echo "Start time   : $(date)"
echo "===================================================="

# ======================================================
# STARTUP SANITY CHECK
# Scan OUTPUT_DIR, verify each clip has all render files,
# and build/update the progress CSV before the main loop.
# ======================================================
export VGGT_OUTPUT_DIR="$OUTPUT_DIR"
export VGGT_PROGRESS_CSV="$PROGRESS_CSV"
export VGGT_CSV_LOCK="$CSV_LOCK"

echo ""
echo "[sanity] Scanning output directory and updating progress CSV..."

# Acquire exclusive lock for the sanity-check CSV write
(
  flock -x 200
  python - <<'PYEOF'
import os, csv, fcntl
from pathlib import Path
from datetime import datetime

output_dir  = Path(os.environ['VGGT_OUTPUT_DIR'])
csv_path    = Path(os.environ['VGGT_PROGRESS_CSV'])
render_files = ['ground_0deg.png', 'elevated_45deg.png', 'elevated_110deg.png']

existing = {}
if csv_path.exists():
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            existing[row['video_id']] = row

if output_dir.exists():
    for video_dir in sorted(output_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        clip_dirs   = [d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('clip_')]
        clips_total = len(clip_dirs)
        clips_done  = sum(1 for d in clip_dirs if all((d / f).exists() for f in render_files))

        if clips_total == 0:
            status = 'not_started'
        elif clips_done == clips_total:
            status = 'complete'
        else:
            status = 'partial'

        existing[video_dir.name] = {
            'video_id':     video_dir.name,
            'clips_done':   clips_done,
            'clips_total':  clips_total,
            'status':       status,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

fieldnames = ['video_id', 'clips_done', 'clips_total', 'status', 'last_updated']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in existing.values():
        writer.writerow({k: row.get(k, '') for k in fieldnames})

n_complete = sum(1 for r in existing.values() if r['status'] == 'complete')
n_partial  = sum(1 for r in existing.values() if r['status'] == 'partial')
n_other    = len(existing) - n_complete - n_partial
print(f"[sanity] {len(existing)} videos tracked in CSV")
print(f"[sanity]   complete={n_complete}  partial={n_partial}  other={n_other}")
PYEOF
) 200>"$CSV_LOCK"

echo ""

# ======================================================
# MAIN LOOP — one video at a time, processed in REVERSE
# ======================================================
TEMP_LIST="/tmp/vggt_single_rev_$$.list"

total=0
skipped=0
processed=0
failed=0

# tac reverses the line order of the list file
while IFS= read -r line || [[ -n "$line" ]]; do
    video_id=$(echo "$line" | awk '{print $1}')
    [[ -z "$video_id" ]] && continue

    video_stem="${video_id%.*}"
    total=$((total + 1))

    # --------------------------------------------------
    # Check progress CSV: skip if already complete
    # (simple read — no lock needed, worst case we redo
    #  a video the other job is finishing simultaneously,
    #  which the output-dir check in inference_and_render.py
    #  will handle safely)
    # --------------------------------------------------
    if [ "$FORCE_RECOMPUTE" = false ] && [ -f "$PROGRESS_CSV" ]; then
        csv_status=$(awk -F',' -v vid="$video_stem" 'NR>1 && $1==vid {print $4}' "$PROGRESS_CSV")
        clips_done=$(awk -F',' -v vid="$video_stem" 'NR>1 && $1==vid {print $2}' "$PROGRESS_CSV")
        clips_total=$(awk -F',' -v vid="$video_stem" 'NR>1 && $1==vid {print $3}' "$PROGRESS_CSV")

        if [ "$csv_status" = "complete" ]; then
            echo "[SKIP] $video_stem — complete ($clips_done/$clips_total clips)"
            skipped=$((skipped + 1))
            continue
        elif [ "$csv_status" = "partial" ]; then
            echo "[RESUME] $video_stem — partial ($clips_done/$clips_total clips done), will complete"
        fi
    fi

    echo ""
    echo "----------------------------------------------------"
    echo "[$((processed + skipped + failed + 1))] Video: $video_stem"
    echo "Time: $(date)"
    echo "----------------------------------------------------"

    # --- Step 1: extract frames for this video only ---
    echo "[1/3] Extracting frames..."
    echo "$video_id" > "$TEMP_LIST"

    export VGGT_VIDEO_DIR="$VIDEO_DIR"
    export VGGT_TEMP_LIST="$TEMP_LIST"
    export VGGT_TEMP_FRAMES="$TEMP_FRAMES_DIR"
    export VGGT_FORCE="$FORCE_PY"

    python - <<'PYEOF'
import os, sys
sys.path.insert(0, '.')
from create_frames_train import process_videos

process_videos(
    directory_path=os.environ['VGGT_VIDEO_DIR'],
    filename_list_path=os.environ['VGGT_TEMP_LIST'],
    output_csv='/tmp/vggt_meta_train_rev.csv',
    output_frames_dir=os.environ['VGGT_TEMP_FRAMES'],
    clip_duration_seconds=1.0,
    frames_per_clip=8,
    frame_skip=1,
    force_reprocess=os.environ.get('VGGT_FORCE', 'False') == 'True',
)
PYEOF

    extract_status=$?
    if [ $extract_status -ne 0 ]; then
        echo "[ERROR] Frame extraction failed for $video_stem (exit $extract_status)"
        failed=$((failed + 1))
        rm -rf "${TEMP_FRAMES_DIR:?}/$video_stem"
        # Mark as failed in CSV (locked write)
        export VGGT_VIDEO_STEM="$video_stem"
        export VGGT_CLIP_STATUS="failed"
        (
          flock -x 200
          python - <<'PYEOF'
import os, csv
from pathlib import Path
from datetime import datetime
csv_path   = Path(os.environ['VGGT_PROGRESS_CSV'])
video_stem = os.environ['VGGT_VIDEO_STEM']
existing = {}
if csv_path.exists():
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f): existing[row['video_id']] = row
existing[video_stem] = {'video_id': video_stem, 'clips_done': 0, 'clips_total': 0,
                        'status': 'failed', 'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
fieldnames = ['video_id', 'clips_done', 'clips_total', 'status', 'last_updated']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in existing.values(): writer.writerow({k: row.get(k, '') for k in fieldnames})
PYEOF
        ) 200>"$CSV_LOCK"
        continue
    fi

    # Verify frames were actually extracted
    if [ ! -d "$TEMP_FRAMES_DIR/$video_stem" ]; then
        echo "[WARN] No frames extracted for $video_stem (video not found in source dir?)"
        failed=$((failed + 1))
        continue
    fi

    # --- Step 2: run VGGT inference on this video's clips ---
    echo "[2/3] Running VGGT inference..."
    python inference_and_render.py "$TEMP_FRAMES_DIR" "$OUTPUT_DIR" $FORCE_FLAG
    infer_status=$?

    # --- Step 3: delete frames to free space ---
    echo "[3/3] Cleaning up frames for $video_stem..."
    rm -rf "${TEMP_FRAMES_DIR:?}/$video_stem"

    # --------------------------------------------------
    # Update progress CSV for this video (locked write)
    # --------------------------------------------------
    export VGGT_VIDEO_STEM="$video_stem"
    (
      flock -x 200
      python - <<'PYEOF'
import os, csv
from pathlib import Path
from datetime import datetime

output_dir   = Path(os.environ['VGGT_OUTPUT_DIR'])
csv_path     = Path(os.environ['VGGT_PROGRESS_CSV'])
video_stem   = os.environ['VGGT_VIDEO_STEM']
render_files = ['ground_0deg.png', 'elevated_45deg.png', 'elevated_110deg.png']

video_dir = output_dir / video_stem
clips_total, clips_done = 0, 0
if video_dir.exists():
    clip_dirs   = [d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('clip_')]
    clips_total = len(clip_dirs)
    clips_done  = sum(1 for d in clip_dirs if all((d / f).exists() for f in render_files))

if clips_total == 0:
    status = 'failed'
elif clips_done == clips_total:
    status = 'complete'
else:
    status = 'partial'

existing = {}
if csv_path.exists():
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f): existing[row['video_id']] = row

existing[video_stem] = {
    'video_id':     video_stem,
    'clips_done':   clips_done,
    'clips_total':  clips_total,
    'status':       status,
    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}

fieldnames = ['video_id', 'clips_done', 'clips_total', 'status', 'last_updated']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in existing.values(): writer.writerow({k: row.get(k, '') for k in fieldnames})

print(f"[progress] {video_stem}: {clips_done}/{clips_total} clips → {status}")
PYEOF
    ) 200>"$CSV_LOCK"

    if [ $infer_status -ne 0 ]; then
        echo "[ERROR] Inference failed for $video_stem (exit $infer_status)"
        failed=$((failed + 1))
    else
        processed=$((processed + 1))
        echo "[OK] Done: $video_stem"
    fi

done < <(tac "$FILENAME_LIST")

rm -f "$TEMP_LIST"
rmdir "$TEMP_FRAMES_DIR" 2>/dev/null   # only removes if empty

echo ""
echo "===================================================="
echo "Pipeline complete! (REVERSED run)"
echo "Total videos in list : $total"
echo "Processed this run   : $processed"
echo "Skipped (complete)   : $skipped"
echo "Failed               : $failed"
echo "Progress CSV         : $PROGRESS_CSV"
echo "End time             : $(date)"
echo "===================================================="

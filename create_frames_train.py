import os
import csv
import math
import cv2
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set


def get_video_metadata(video_path: str) -> Optional[Dict]:
    try:
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        filename = os.path.basename(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration_seconds = frame_count / fps if fps > 0 else 0
        resolution = f"{width}x{height}"

        cap.release()

        return {
            'filename': filename,
            'file_path': video_path,
            'duration_seconds': round(duration_seconds, 2),
            'file_size_mb': round(file_size_mb, 2),
            'resolution': resolution,
            'fps': fps,
            'total_frames': int(frame_count)
        }

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None


def load_processed_videos(output_csv: str) -> Set[str]:
    processed = set()

    if not os.path.exists(output_csv):
        return processed

    try:
        with open(output_csv, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if 'filename' in row:
                    processed.add(row['filename'])
                    processed.add(Path(row['filename']).stem)
    except Exception as e:
        print(f"Warning: Could not read existing CSV: {e}")

    return processed


def check_video_clips_exist(video_name: str, output_frames_dir: str,
                             expected_clips: int, frames_per_clip: int) -> bool:
    video_output_dir = os.path.join(output_frames_dir, video_name)

    if not os.path.exists(video_output_dir):
        return False

    metadata_path = os.path.join(video_output_dir, "clip_metadata.csv")
    if not os.path.exists(metadata_path):
        return False

    clip_dirs = [d for d in os.listdir(video_output_dir)
                 if d.startswith("clip_") and os.path.isdir(os.path.join(video_output_dir, d))]

    if len(clip_dirs) < expected_clips:
        return False

    clips_to_check = [clip_dirs[0], clip_dirs[len(clip_dirs) // 2], clip_dirs[-1]]

    for clip_name in clips_to_check:
        clip_dir = os.path.join(video_output_dir, clip_name)
        frame_files = [f for f in os.listdir(clip_dir) if f.endswith('.jpg')]
        if len(frame_files) < frames_per_clip:
            return False

    return True


def get_processing_status(video_path: str, output_frames_dir: str,
                           clip_duration_seconds: float, frames_per_clip: int) -> Dict:
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_frames_dir, video_name)

    status = {
        'video_name': video_name,
        'video_path': video_path,
        'output_dir_exists': os.path.exists(video_output_dir),
        'metadata_exists': False,
        'expected_clips': 0,
        'actual_clips': 0,
        'is_complete': False,
        'needs_processing': True
    }

    metadata = get_video_metadata(video_path)
    if metadata:
        status['expected_clips'] = int(metadata['duration_seconds'] / clip_duration_seconds)

    if not status['output_dir_exists']:
        return status

    metadata_path = os.path.join(video_output_dir, "clip_metadata.csv")
    status['metadata_exists'] = os.path.exists(metadata_path)

    if os.path.exists(video_output_dir):
        clip_dirs = [d for d in os.listdir(video_output_dir)
                     if d.startswith("clip_") and os.path.isdir(os.path.join(video_output_dir, d))]
        status['actual_clips'] = len(clip_dirs)

    if status['metadata_exists'] and status['actual_clips'] >= status['expected_clips'] - 1:
        status['is_complete'] = check_video_clips_exist(
            video_name, output_frames_dir,
            status['expected_clips'] - 1,
            frames_per_clip
        )

    status['needs_processing'] = not status['is_complete']

    return status


def extract_clips(
    video_path: str,
    output_base_dir: str,
    clip_duration_seconds: float = 1.0,
    frames_per_clip: int = 8,
    frame_skip: int = 1,
):
    import time

    print(f"\n  [extract] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [extract] ERROR: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = Path(video_path).stem
    video_duration = total_frames / fps

    print(f"  [extract] Video info: {total_frames} frames, {fps:.2f} fps, duration: {video_duration:.2f}s")

    clip_frame_span = frames_per_clip + (frames_per_clip - 1) * frame_skip

    print(f"  [extract] Clip config: {frames_per_clip} frames, skip={frame_skip}, span={clip_frame_span} frames (~{clip_frame_span/fps:.3f}s)")

    num_clips = math.ceil(video_duration / clip_duration_seconds)
    print(f"  [extract] Expected clips: {num_clips} (1 clip per {clip_duration_seconds}s)")

    video_output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    total_time = 0

    for clip_idx in range(num_clips):
        clip_start = time.time()

        clip_start_time = clip_idx * clip_duration_seconds
        clip_start_frame = int(clip_start_time * fps)

        if clip_start_frame >= total_frames:
            print(f"  [extract] Clip {clip_idx} start exceeds video length, stopping")
            break

        clip_dir = os.path.join(video_output_dir, f"clip_{clip_idx:04d}")
        os.makedirs(clip_dir, exist_ok=True)

        frame_indices = [clip_start_frame + i * (frame_skip + 1) for i in range(frames_per_clip)]

        for frame_num, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(f"  [extract] WARNING: Failed to read frame {frame_idx}")
                continue

            timestamp = frame_idx / fps
            filename = f"frame_{frame_num:02d}_idx{frame_idx:06d}_t{timestamp:.3f}s.jpg"
            cv2.imwrite(os.path.join(clip_dir, filename), frame)

        clip_time = time.time() - clip_start
        total_time += clip_time

        if (clip_idx + 1) % 5 == 0 or clip_idx == 0:
            avg_time = total_time / (clip_idx + 1)
            remaining = num_clips - (clip_idx + 1)
            eta = avg_time * remaining
            print(f"  [extract] Clip {clip_idx + 1}/{num_clips} | "
                  f"Time: {clip_time:.2f}s | Avg: {avg_time:.2f}s | ETA: {eta:.1f}s")

    cap.release()

    metadata_path = os.path.join(video_output_dir, "clip_metadata.csv")
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'start_time_s', 'start_frame', 'frame_indices'])
        for clip_idx in range(num_clips):
            clip_start_time = clip_idx * clip_duration_seconds
            clip_start_frame = int(clip_start_time * fps)
            if clip_start_frame + clip_frame_span > total_frames:
                break
            frame_indices = [clip_start_frame + i * (frame_skip + 1) for i in range(frames_per_clip)]
            writer.writerow([
                f"clip_{clip_idx:04d}",
                f"{clip_start_time:.3f}",
                clip_start_frame,
                str(frame_indices)
            ])

    print(f"  [extract] Done with video: {video_name} (total time: {total_time:.1f}s)")


def load_filename_list(txt_file_path: str) -> List[str]:
    filenames = []
    with open(txt_file_path, 'r') as f:
        for line in f:
            filename = line.strip().split()[0]
            if filename:
                filenames.append(filename)
    return filenames


def process_videos(
    directory_path: str,
    filename_list_path: str,
    output_csv: str = 'metadata_clips.csv',
    output_frames_dir: str = 'extracted_clips',
    clip_duration_seconds: float = 1.0,
    frames_per_clip: int = 8,
    frame_skip: int = 1,
    video_percent: float = 100.0,
    force_reprocess: bool = False,
    verify_completeness: bool = True,
):
    import time
    import sys

    print("=" * 60)
    print("VIDEO PREPROCESSING PIPELINE - CLIP EXTRACTION")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print("=" * 60 + "\n")

    print(f"[STEP 1] Loading filename list from: {filename_list_path}")
    target_filenames = load_filename_list(filename_list_path)
    print(f"         Loaded {len(target_filenames)} filenames")

    target_set = set()
    for fn in target_filenames:
        target_set.add(fn)
        target_set.add(Path(fn).stem)

    print(f"\n[STEP 2] Scanning directory: {directory_path}")
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}

    all_video_files = []
    for file_path in Path(directory_path).rglob('*'):
        if file_path.suffix.lower() in video_extensions:
            all_video_files.append(file_path)
    print(f"         Found {len(all_video_files)} total video files")

    video_files = []
    for file_path in all_video_files:
        if file_path.name in target_set or file_path.stem in target_set:
            video_files.append(str(file_path))

    print(f"         Matched {len(video_files)} videos from list")

    if len(video_files) == 0:
        print("\nERROR: No matching videos found!")
        print("First 5 filenames in list:", target_filenames[:5])
        print("First 5 videos in directory:", [f.name for f in all_video_files[:5]])
        return

    # Apply video_percent cap
    if video_percent < 100.0:
        cap_count = max(1, math.ceil(len(video_files) * video_percent / 100.0))
        print(f"\n[STEP 2b] Applying {video_percent}% cap: {cap_count}/{len(video_files)} videos will be considered")
        video_files = video_files[:cap_count]

    print(f"\n[STEP 3] Checking for already processed videos")
    print(f"         Output CSV: {output_csv}")
    print(f"         Output directory: {output_frames_dir}")
    print(f"         Force reprocess: {force_reprocess}")
    print(f"         Verify completeness: {verify_completeness}")

    processed_in_csv = load_processed_videos(output_csv)
    print(f"         Found {len(processed_in_csv)} entries in existing CSV")

    videos_to_process = []
    videos_already_done = []
    videos_incomplete = []

    for video_path in video_files:
        video_name = Path(video_path).stem
        filename = os.path.basename(video_path)

        if force_reprocess:
            videos_to_process.append(video_path)
            continue

        in_csv = filename in processed_in_csv or video_name in processed_in_csv

        if verify_completeness and in_csv:
            status = get_processing_status(
                video_path, output_frames_dir,
                clip_duration_seconds, frames_per_clip
            )

            if status['is_complete']:
                videos_already_done.append((video_path, status))
            else:
                videos_incomplete.append((video_path, status))
                videos_to_process.append(video_path)
        elif in_csv:
            videos_already_done.append((video_path, {'actual_clips': '?', 'expected_clips': '?'}))
        else:
            videos_to_process.append(video_path)

    print(f"\n         Processing Status Summary:")
    print(f"         +-- Already complete: {len(videos_already_done)} videos")
    print(f"         +-- Incomplete (will reprocess): {len(videos_incomplete)} videos")
    print(f"         +-- Need processing: {len(videos_to_process) - len(videos_incomplete)} videos")
    print(f"         Total to process: {len(videos_to_process)} videos")

    if videos_incomplete:
        print(f"\n         Incomplete videos:")
        for _, status in videos_incomplete[:10]:
            print(f"           - {status['video_name']}: {status['actual_clips']}/{status['expected_clips']} clips")
        if len(videos_incomplete) > 10:
            print(f"           ... and {len(videos_incomplete) - 10} more")

    if len(videos_to_process) == 0:
        print("\n" + "=" * 60)
        print("ALL VIDEOS ALREADY PROCESSED!")
        print("=" * 60)
        print(f"All {len(video_files)} videos have been processed.")
        print("Use force_reprocess=True to reprocess all videos.")
        return

    print(f"\n[STEP 4] Creating output directories")
    os.makedirs(output_frames_dir, exist_ok=True)
    print(f"         Output directory: {output_frames_dir}")

    print(f"\n[STEP 5] Processing {len(videos_to_process)} videos (skipping {len(videos_already_done)} already done)")
    print(f"         Clip config: {frames_per_clip} frames/clip, skip={frame_skip}, "
          f"temporal position={clip_duration_seconds}s")
    print("=" * 60)

    existing_data = []
    if os.path.exists(output_csv) and not force_reprocess:
        with open(output_csv, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_data.append(row)
        print(f"         Loaded {len(existing_data)} existing entries from CSV")

    fieldnames = ['filename', 'file_path', 'duration_seconds', 'file_size_mb',
                  'resolution', 'fps', 'total_frames', 'num_clips']

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in existing_data:
            filename = row.get('filename', '')
            if filename not in [os.path.basename(vp) for vp in videos_to_process]:
                writer.writerow(row)

        total_start = time.time()
        processed_count = 0

        for i, video_path in enumerate(videos_to_process, 1):
            video_start = time.time()
            print(f"\n{'=' * 60}")
            print(f"VIDEO {i}/{len(videos_to_process)}: {os.path.basename(video_path)}")
            print(f"(Overall progress: {len(videos_already_done) + i}/{len(video_files)})")
            print(f"{'=' * 60}")

            metadata = get_video_metadata(video_path)
            if metadata:
                num_clips = int(metadata['duration_seconds'] / clip_duration_seconds)
                print(f"  Metadata: {metadata['duration_seconds']}s, {metadata['resolution']}, "
                      f"{metadata['file_size_mb']:.1f}MB, {metadata['fps']:.2f}fps")
                print(f"  Expected clips: {num_clips}")

                csv_metadata = {k: v for k, v in metadata.items()}
                csv_metadata['num_clips'] = num_clips
                writer.writerow(csv_metadata)
                csvfile.flush()

                extract_clips(
                    video_path,
                    output_frames_dir,
                    clip_duration_seconds=clip_duration_seconds,
                    frames_per_clip=frames_per_clip,
                    frame_skip=frame_skip,
                )
                processed_count += 1
            else:
                print(f"  ERROR: Could not get metadata for {video_path}")

            video_time = time.time() - video_start
            print(f"\n  Video completed in {video_time:.1f}s")

            if i % 10 == 0:
                elapsed = time.time() - total_start
                avg_per_video = elapsed / i
                remaining = (len(videos_to_process) - i) * avg_per_video
                print(f"\n  === Progress: {i}/{len(videos_to_process)} | "
                      f"Elapsed: {elapsed / 60:.1f}min | ETA: {remaining / 60:.1f}min ===")

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Videos processed this run: {processed_count}")
    print(f"Videos skipped (already done): {len(videos_already_done)}")
    print(f"Total videos considered: {len(video_files)}")
    print(f"Total time this run: {total_time / 60:.1f} minutes")
    print(f"Metadata saved to: {output_csv}")
    print(f"Clips saved to: {output_frames_dir}/")
    print("\nOutput structure:")
    print(f"  {output_frames_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames into clips")
    parser.add_argument("--percent", type=float, default=100.0,
                        help="Percentage of matched videos to process (e.g. 5 for 5%%)")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocess all videos even if already done")
    args = parser.parse_args()

    directory = "/home/c3-0/datasets/BDD_Dataset/Videos/bdd100k/videos/train/"
    filename_list = "gama_list/train_day.list"

    output_csv = "metadata_clips_train.csv"
    output_frames_dir = "extracted_clips_train"

    CLIP_DURATION_SECONDS = 1.0
    FRAMES_PER_CLIP = 8
    FRAME_SKIP = 1
    VERIFY_COMPLETENESS = True

    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
    elif not os.path.exists(filename_list):
        print(f"Filename list does not exist: {filename_list}")
    else:
        process_videos(
            directory_path=directory,
            filename_list_path=filename_list,
            output_csv=output_csv,
            output_frames_dir=output_frames_dir,
            clip_duration_seconds=CLIP_DURATION_SECONDS,
            frames_per_clip=FRAMES_PER_CLIP,
            frame_skip=FRAME_SKIP,
            video_percent=args.percent,
            force_reprocess=args.force,
            verify_completeness=VERIFY_COMPLETENESS,
        )

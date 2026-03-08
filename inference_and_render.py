import torch
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from vggt.models.vggt import VGGT
import json


# ============================================
# Render Function (vectorized)
# ============================================
def render_from_camera(points, colors, extrinsics, intrinsics, width=518, height=518, point_size=2):
    """Render point cloud from a camera pose using projection (vectorized z-buffer)."""
    world_to_cam = np.linalg.inv(extrinsics)

    points_h = np.hstack([points, np.ones((len(points), 1))])
    points_cam = (world_to_cam @ points_h.T).T[:, :3]

    valid = points_cam[:, 2] > 0.01
    points_cam = points_cam[valid]
    colors_valid = colors[valid]

    if len(points_cam) == 0:
        print("  WARNING: No points in front of camera!")
        return np.zeros((height, width, 3), dtype=np.uint8)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_proj = (points_cam[:, 0] * fx / points_cam[:, 2]) + cx
    y_proj = (points_cam[:, 1] * fy / points_cam[:, 2]) + cy

    in_bounds = (x_proj >= 0) & (x_proj < width) & (y_proj >= 0) & (y_proj < height)
    x_coords = x_proj[in_bounds].astype(int)
    y_coords = y_proj[in_bounds].astype(int)
    colors_vis = colors_valid[in_bounds]
    depths = points_cam[in_bounds, 2]

    if len(x_coords) == 0:
        print("  WARNING: No points project into image bounds!")
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Sort far-to-near so closer points overwrite farther ones
    order = np.argsort(-depths)
    x_coords = x_coords[order]
    y_coords = y_coords[order]
    colors_vis = colors_vis[order]

    # Vectorized splatting with point_size
    offsets = np.arange(-point_size, point_size + 1)
    dx, dy = np.meshgrid(offsets, offsets)
    dx = dx.flatten()
    dy = dy.flatten()
    k = len(dx)

    px = (x_coords[:, None] + dx[None, :]).flatten()
    py = (y_coords[:, None] + dy[None, :]).flatten()
    c  = np.repeat(colors_vis, k, axis=0)

    valid_splat = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px = px[valid_splat]
    py = py[valid_splat]
    c  = c[valid_splat]

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[py, px] = c
    return image


# ============================================
# Camera pose construction helpers
# ============================================
def build_110_camera(extrinsics_orig, intrinsics, point_cloud_center):
    """Return (extrinsics, intrinsics) for the 110-degree elevated view."""
    original_position = extrinsics_orig[:3, 3]
    original_forward  = extrinsics_orig[:3, 2]

    to_center = point_cloud_center - original_position
    horizontal_distance = np.linalg.norm(to_center[:2])

    elevation_angle = 110 * np.pi / 180
    vertical_offset = horizontal_distance * np.tan(elevation_angle)

    forward_horizontal = original_forward.copy()
    forward_horizontal[1] = 0
    norm = np.linalg.norm(forward_horizontal)
    if norm > 1e-6:
        forward_horizontal /= norm

    new_position = point_cloud_center.copy()
    new_position[1] += abs(vertical_offset) * 1.5
    new_position -= forward_horizontal * horizontal_distance * 2.0

    original_right = extrinsics_orig[:3, 0]
    look_at = point_cloud_center - new_position
    look_at /= np.linalg.norm(look_at)

    right = original_right.copy()
    up = np.cross(look_at, right)
    norm = np.linalg.norm(up)
    if norm > 1e-6:
        up /= norm
    right = np.cross(up, look_at)

    ext = extrinsics_orig.copy()
    ext[:3, 0] = right
    ext[:3, 1] = up
    ext[:3, 2] = look_at
    ext[:3, 3] = new_position

    intr = intrinsics.copy()
    intr[0, 0] *= 0.08
    intr[1, 1] *= 0.08

    return ext, intr


def build_45_camera(extrinsics_orig, intrinsics, point_cloud_center):
    """Return (extrinsics, intrinsics) for the 45-degree elevated view."""
    original_position = extrinsics_orig[:3, 3]
    original_forward  = extrinsics_orig[:3, 2]

    to_center = point_cloud_center - original_position
    horizontal_distance = np.linalg.norm(to_center[:2])
    elevation_height = horizontal_distance * np.tan(np.radians(45))

    forward_horizontal = original_forward.copy()
    forward_horizontal[1] = 0
    norm = np.linalg.norm(forward_horizontal)
    if norm > 1e-6:
        forward_horizontal /= norm

    ext = extrinsics_orig.copy()
    ext[1, 3] += elevation_height
    ext[:3, 3] -= forward_horizontal * elevation_height

    intr = intrinsics.copy()
    intr[0, 0] *= 0.7
    intr[1, 1] *= 0.7

    return ext, intr


# ============================================
# Process point cloud and render views
# ============================================
def process_and_render_point_cloud(world_points, confidences, images_rgb, camera_poses, 
                                   intrinsics, output_dir, confidence_threshold=0.5):
    """Process point cloud and render elevated views without saving .ply files"""
    
    # Flatten and filter points
    num_frames, height, width, _ = world_points.shape
    world_points_flat = world_points.reshape(-1, 3)
    confidences_flat = confidences.reshape(-1)
    
    mask = confidences_flat > confidence_threshold
    filtered_points = world_points_flat[mask]
    
    colors_flat = images_rgb.reshape(-1, 3)
    colors = (colors_flat[mask] * 255).astype(np.uint8)
    
    # Transform coordinates (flip Y and Z)
    filtered_points_transformed = filtered_points.copy()
    filtered_points_transformed[:, 1] = -filtered_points[:, 1]
    filtered_points_transformed[:, 2] = -filtered_points[:, 2]
    
    print(f"  Filtered {len(filtered_points_transformed)} points (confidence > {confidence_threshold})")
    
    # Calculate point cloud center
    point_cloud_center = filtered_points_transformed.mean(axis=0)
    print(f"  Point cloud center: {point_cloud_center}")
    
    # Transform camera poses to match coordinate system
    camera_poses_transformed = camera_poses.copy()
    camera_poses_transformed[:, 1, :] *= -1
    camera_poses_transformed[:, 2, :] *= -1
    
    # Get first camera pose for elevated views
    extrinsics_0 = camera_poses_transformed[0]
    intrinsics_0 = intrinsics[0]
    
    # Define views to render
    views = {
        "ground_0deg": (extrinsics_0, intrinsics_0),
        "elevated_45deg": build_45_camera(extrinsics_0, intrinsics_0, point_cloud_center),
        "elevated_110deg": build_110_camera(extrinsics_0, intrinsics_0, point_cloud_center),
    }
    
    # Render each view
    rendered_paths = []
    for name, (ext, intr) in views.items():
        print(f"  Rendering {name} ...")
        img = render_from_camera(filtered_points_transformed, colors, ext, intr, point_size=2)
        out_path = output_dir / f"{name}.png"
        Image.fromarray(img).save(out_path)
        rendered_paths.append(out_path)
        print(f"    [OK] Saved -> {out_path.name}")
    
    return filtered_points_transformed, colors, rendered_paths


def load_and_preprocess_images(image_paths, size=518):
    """Load and preprocess images to the required format"""
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((size, size), Image.BILINEAR)
        img_array = np.array(img).astype(np.float32) / 255.0
        images.append(img_array)
    
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    return images


def save_camera_poses(camera_poses, intrinsics, image_files, output_dir):
    """Save camera poses as JSON and NPZ"""
    # Transform camera poses to match coordinate system (flip Y and Z)
    camera_poses_transformed = camera_poses.copy()
    camera_poses_transformed[:, 1, :] *= -1
    camera_poses_transformed[:, 2, :] *= -1
    
    camera_data = {
        "num_cameras": len(image_files),
        "image_width": 518,
        "image_height": 518,
        "cameras": []
    }
    
    for i, (pose, K) in enumerate(zip(camera_poses_transformed, intrinsics)):
        camera_info = {
            "frame_id": i,
            "image_path": str(image_files[i]),
            "extrinsics": pose.tolist(),
            "intrinsics": K.tolist(),
            "rotation": pose[:3, :3].tolist(),
            "translation": pose[:3, 3].tolist(),
            "focal_length": [float(K[0, 0]), float(K[1, 1])],
            "principal_point": [float(K[0, 2]), float(K[1, 2])]
        }
        camera_data["cameras"].append(camera_info)
    
    # Save as JSON
    camera_json_path = output_dir / "camera_poses.json"
    with open(camera_json_path, "w") as f:
        json.dump(camera_data, f, indent=2)
    
    # Save as NPZ
    camera_npz_path = output_dir / "camera_poses.npz"
    np.savez(
        camera_npz_path,
        extrinsics=camera_poses,
        intrinsics=intrinsics,
        image_paths=np.array([str(f) for f in image_files])
    )
    
    print(f"  [OK] Camera poses saved to {camera_json_path.name}")
    print(f"  [OK] Camera poses saved to {camera_npz_path.name}")


def process_clip(model, clip_dir, output_dir, device, dtype):
    """Process a single clip directory and render elevated views"""
    original_dir = clip_dir 
    
    image_files = sorted(original_dir.glob("*.jpg"))
    
    if len(image_files) == 0:
        print(f"No images found in {original_dir}")
        return
    
    print(f"Processing {len(image_files)} images from {clip_dir.name}")
    
    # Load images
    images = load_and_preprocess_images(image_files).to(device)
    
    # Run inference
    print("Running VGGT inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batch = images.unsqueeze(0)
            predictions = model(images_batch)
    
    print("[OK] Inference complete!")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract camera poses
    if "camera_poses" in predictions:
        camera_poses = predictions["camera_poses"][0].cpu().numpy()
    elif "extrinsics" in predictions:
        camera_poses = predictions["extrinsics"][0].cpu().numpy()
    else:
        print("Warning: Camera poses not found in predictions")
        camera_poses = np.tile(np.eye(4), (len(image_files), 1, 1))
    
    # Extract intrinsics
    if "intrinsics" in predictions:
        intrinsics = predictions["intrinsics"][0].cpu().numpy()
    else:
        focal_length = 518 * 0.8
        cx, cy = 518 / 2, 518 / 2
        intrinsics = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        intrinsics = np.tile(intrinsics, (len(image_files), 1, 1))
    
    # Extract 3D points
    world_points = predictions["world_points"][0].cpu().numpy()
    confidences = predictions["world_points_conf"][0].cpu().numpy()
    
    print(f"World points shape: {world_points.shape}")
    print(f"Confidences shape: {confidences.shape}")
    
    # Get RGB colors
    images_rgb = images_batch[0].permute(0, 2, 3, 1).cpu().numpy()
    
    # Process point cloud and render elevated views (NO .ply files saved)
    print("\nRendering elevated views...")
    filtered_points, colors, rendered_paths = process_and_render_point_cloud(
        world_points, confidences, images_rgb, camera_poses, intrinsics, output_dir
    )
    
    # Save camera poses
    save_camera_poses(camera_poses, intrinsics, image_files, output_dir)
    
    # Save lightweight metadata (optional - much smaller than full point cloud)
    np.savez_compressed(
        output_dir / "metadata.npz",
        num_points=len(filtered_points),
        point_cloud_center=filtered_points.mean(axis=0),
        num_cameras=len(image_files),
        image_files=[str(f.name) for f in image_files]
    )
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"[OK] Rendered views saved:")
    for path in rendered_paths:
        print(f"     - {path.name}")
    print(f"[OK] Camera poses (JSON): {output_dir / 'camera_poses.json'}")
    print(f"[OK] Number of cameras: {len(image_files)}")
    print(f"[OK] Number of 3D points: {len(filtered_points)}")
    print(f"{'='*50}")


EXPECTED_RENDER_FILES = ["ground_0deg.png", "elevated_45deg.png", "elevated_110deg.png"]


def is_clip_done(output_clip_dir: Path) -> bool:
    """Return True if all expected rendered images already exist for this clip."""
    return all((output_clip_dir / f).exists() for f in EXPECTED_RENDER_FILES)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VGGT inference and rendering")
    parser.add_argument("input_dir", type=Path, help="Directory containing video folders")
    parser.add_argument("output_dir", type=Path, help="Directory to save outputs")
    parser.add_argument("max_videos", type=int, nargs="?", default=None,
                        help="Maximum number of videos to process (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Force recompute even if outputs already exist")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_base = args.output_dir
    max_videos = args.max_videos
    force = args.force
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_base}")
    if max_videos:
        print(f"Max videos to process: {max_videos}")
    else:
        print(f"Processing all videos")
    
    print("\nLoading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print("[OK] Model loaded successfully!")
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        sys.exit(1)
    
    # Get all video directories
    all_video_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    # Limit to max_videos if specified
    if max_videos:
        video_dirs = all_video_dirs[:max_videos]
        print(f"\nFound {len(all_video_dirs)} total videos, processing first {len(video_dirs)}")
    else:
        video_dirs = all_video_dirs
        print(f"\nFound {len(video_dirs)} videos to process")
    
    # Track statistics
    success_count = 0
    fail_count = 0
    failed_videos = []
    
    # Process each video
    for video_idx, video_dir in enumerate(video_dirs, 1):
        video_id = video_dir.name
        output_video_dir = output_base / video_id
        
        print(f"\n{'='*60}")
        print(f"[{video_idx}/{len(video_dirs)}] Processing video: {video_id}")
        print(f"{'='*60}")
        
        # Get all clips in this video
        clip_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith("clip_")])
        
        if len(clip_dirs) == 0:
            print(f"Warning: No clips found in {video_id}")
            fail_count += 1
            failed_videos.append(video_id)
            continue
        
        print(f"Found {len(clip_dirs)} clips in {video_id}")
        
        video_success = True
        for clip_dir in clip_dirs:
            clip_name = clip_dir.name
            output_clip_dir = output_video_dir / clip_name

            if not force and is_clip_done(output_clip_dir):
                print(f"\n[SKIP] {video_id}/{clip_name} -- renders already exist")
                continue

            print(f"\nProcessing {video_id}/{clip_name}")

            try:
                process_clip(model, clip_dir, output_clip_dir, device, dtype)
            except Exception as e:
                print(f"Error processing {clip_name}: {e}")
                import traceback
                traceback.print_exc()
                video_success = False
                continue
        
        if video_success:
            success_count += 1
            print(f"[OK] Successfully processed all clips in {video_id}")
        else:
            fail_count += 1
            failed_videos.append(video_id)
            print(f"[FAIL] Some clips failed in {video_id}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos processed: {len(video_dirs)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    if failed_videos:
        print(f"\nFailed videos:")
        for vid in failed_videos:
            print(f"  - {vid}")
    
    print(f"\nResults saved to: {output_base}")
    print(f"Note: No .ply files saved - only rendered images and metadata")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
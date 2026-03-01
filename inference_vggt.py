import torch
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from vggt.models.vggt import VGGT
import open3d as o3d
import json

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

def save_point_cloud(world_points, confidences, images_rgb, output_dir, confidence_threshold=0.5):
    """Save point cloud with color information"""
    num_frames, height, width, _ = world_points.shape
    world_points_flat = world_points.reshape(-1, 3)
    confidences_flat = confidences.reshape(-1)
    
    mask = confidences_flat > confidence_threshold
    filtered_points = world_points_flat[mask]
    
    colors_flat = images_rgb.reshape(-1, 3)
    colors = (colors_flat[mask] * 255).astype(np.uint8)
    
    # Transform coordinates for Open3D (flip Y and Z)
    filtered_points_open3d = filtered_points.copy()
    filtered_points_open3d[:, 1] = -filtered_points[:, 1]
    filtered_points_open3d[:, 2] = -filtered_points[:, 2]
    
    # Save as PLY (manual method)
    output_path_manual = output_dir / "point_cloud.ply"
    with open(output_path_manual, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(filtered_points_open3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(filtered_points_open3d, colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
    
    print(f"[OK] Saved point cloud (manual) with {len(filtered_points_open3d)} points to {output_path_manual}")
    
    # Save using Open3D
    output_path_o3d = output_dir / "point_cloud_o3d.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points_open3d)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud(str(output_path_o3d), pcd)
    
    print(f"[OK] Saved point cloud (Open3D) with {len(filtered_points_open3d)} points to {output_path_o3d}")
    
    return filtered_points_open3d, colors

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
    
    print(f"[OK] Camera poses saved to {camera_json_path}")
    print(f"[OK] Camera poses saved to {camera_npz_path}")

def process_clip(model, clip_dir, output_dir, device, dtype):
    """Process a single clip directory"""
    original_dir = clip_dir / "original"
    
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
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract camera poses
    if "camera_poses" in predictions:
        camera_poses = predictions["camera_poses"][0].cpu().numpy()
    elif "extrinsics" in predictions:
        camera_poses = predictions["extrinsics"][0].cpu().numpy()
    else:
        print("Warning: Camera poses not found in predictions")
        print(f"Available keys: {predictions.keys()}")
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
    
    # Save point cloud
    filtered_points, colors = save_point_cloud(world_points, confidences, images_rgb, output_dir)
    
    # Save camera poses
    save_camera_poses(camera_poses, intrinsics, image_files, output_dir)
    
    # Save complete output
    np.savez_compressed(
        output_dir / "vggt_output.npz",
        world_points=world_points,
        confidences=confidences,
        extrinsics=camera_poses,
        intrinsics=intrinsics,
        image_files=[str(f.name) for f in image_files]
    )
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"[OK] Point cloud: {output_dir / 'point_cloud_o3d.ply'}")
    print(f"[OK] Camera poses (JSON): {output_dir / 'camera_poses.json'}")
    print(f"[OK] Camera poses (NPZ): {output_dir / 'camera_poses.npz'}")
    print(f"[OK] Number of cameras: {len(image_files)}")
    print(f"[OK] Number of 3D points: {len(filtered_points)}")
    print(f"{'='*50}")

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python inference_vggt.py <input_dir> <output_dir> [max_videos]")
        print("  <input_dir>   : Directory containing video folders")
        print("  <output_dir>  : Directory to save outputs")
        print("  [max_videos]  : Optional - Maximum number of videos to process (default: all)")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_base = Path(sys.argv[2])
    max_videos = int(sys.argv[3]) if len(sys.argv) == 4 else None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_base}")
    if max_videos:
        print(f"Max videos to process: {max_videos}")
    else:
        print(f"Processing all videos")
    
    print("Loading VGGT model...")
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
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
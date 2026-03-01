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
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0
        images.append(img_array)
    
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).permute(0, 3, 1, 2)
    return images

def save_point_cloud(world_points, confidences, images_rgb, output_dir, confidence_threshold=0.5):
    num_frames, height, width, _ = world_points.shape
    world_points_flat = world_points.reshape(-1, 3)
    confidences_flat = confidences.reshape(-1)
    
    mask = confidences_flat > confidence_threshold
    filtered_points = world_points_flat[mask]
    
    colors_flat = images_rgb.reshape(-1, 3)
    colors = (colors_flat[mask] * 255).astype(np.uint8)
    
    filtered_points_open3d = filtered_points.copy()
    filtered_points_open3d[:, 1] = -filtered_points[:, 1]
    filtered_points_open3d[:, 2] = -filtered_points[:, 2]
    
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
    
    print(f"Saved point cloud (manual) with {len(filtered_points_open3d)} points to {output_path_manual}")
    
    output_path_o3d = output_dir / "point_cloud_o3d.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points_open3d)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud(str(output_path_o3d), pcd)
    
    print(f"Saved point cloud (Open3D) with {len(filtered_points_open3d)} points to {output_path_o3d}")
    
    return filtered_points_open3d, colors

def save_camera_poses(camera_poses, intrinsics, image_files, output_dir):
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
    
    camera_json_path = output_dir / "camera_poses.json"
    with open(camera_json_path, "w") as f:
        json.dump(camera_data, f, indent=2)
    
    camera_npz_path = output_dir / "camera_poses.npz"
    np.savez(
        camera_npz_path,
        extrinsics=camera_poses,
        intrinsics=intrinsics,
        image_paths=np.array([str(f) for f in image_files])
    )
    
    print(f"Saved camera poses to {camera_json_path} and {camera_npz_path}")

def process_clip(model, clip_dir, output_dir, device, dtype):
    original_dir = clip_dir / "original"
    
    image_files = sorted(original_dir.glob("*.jpg"))
    
    if len(image_files) == 0:
        print(f"No images found in {original_dir}")
        return
    
    print(f"Processing {len(image_files)} images from {clip_dir.name}")
    
    images = load_and_preprocess_images(image_files).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            images_batch = images.unsqueeze(0)
            predictions = model(images_batch)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if "camera_poses" in predictions:
        camera_poses = predictions["camera_poses"][0].cpu().numpy()
    elif "extrinsics" in predictions:
        camera_poses = predictions["extrinsics"][0].cpu().numpy()
    else:
        camera_poses = np.tile(np.eye(4), (len(image_files), 1, 1))
    
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
    
    world_points = predictions["world_points"][0].cpu().numpy()
    confidences = predictions["world_points_conf"][0].cpu().numpy()
    
    images_rgb = images_batch[0].permute(0, 2, 3, 1).cpu().numpy()
    
    filtered_points, colors = save_point_cloud(world_points, confidences, images_rgb, output_dir)
    
    save_camera_poses(camera_poses, intrinsics, image_files, output_dir)
    
    np.savez_compressed(
        output_dir / "vggt_output.npz",
        world_points=world_points,
        confidences=confidences,
        extrinsics=camera_poses,
        intrinsics=intrinsics,
        image_files=[str(f.name) for f in image_files]
    )
    
    print(f"Processing complete for {clip_dir.name}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python inference_vggt.py <input_dir> <output_dir> <video_id>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_base = Path(sys.argv[2])
    video_id = sys.argv[3]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_base}")
    print(f"Video ID: {video_id}")
    
    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print("Model loaded successfully!")
    
    video_dir = input_dir / video_id
    output_video_dir = output_base / video_id
    
    if not video_dir.exists():
        print(f"Error: Video directory {video_dir} does not exist!")
        sys.exit(1)
    
    clip_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith("clip_")])
    
    print(f"Found {len(clip_dirs)} clips in {video_id}")
    
    for clip_dir in clip_dirs:
        clip_name = clip_dir.name
        output_clip_dir = output_video_dir / clip_name
        
        print(f"\n{'='*60}")
        print(f"Processing {video_id}/{clip_name}")
        print(f"{'='*60}")
        
        try:
            process_clip(model, clip_dir, output_clip_dir, device, dtype)
        except Exception as e:
            print(f"Error processing {clip_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Results saved to: {output_video_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
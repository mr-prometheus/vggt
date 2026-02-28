import torch
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def load_and_preprocess_images(image_paths, size=518):
    """Load and preprocess images for VGGT"""
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        # Resize to 518x518 (must be multiple of 14 for patch size)
        img = img.resize((size, size), Image.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0
        images.append(img_array)
    
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).permute(0, 3, 1, 2)  # (N, 3, H, W)
    return images

def save_point_cloud(point_map, depth_conf, output_path):
    """Save point cloud as .ply file"""
    # point_map shape: (N, H, W, 3)
    # depth_conf shape: (N, H, W)
    
    points_list = []
    
    for i in range(point_map.shape[0]):
        pts = point_map[i].reshape(-1, 3)
        conf = depth_conf[i].reshape(-1)
        
        # Filter by confidence threshold
        valid_mask = conf > 0.5
        pts = pts[valid_mask]
        
        points_list.append(pts)
    
    all_points = np.vstack(points_list)
    
    # Save as PLY
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for pt in all_points:
            f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
    
    print(f"Saved point cloud with {len(all_points)} points to {output_path}")

def process_clip(model, clip_dir, output_dir, device, dtype):
    """Process a single clip directory"""
    original_dir = clip_dir / "original"
    
    # Get all image files sorted
    image_files = sorted(original_dir.glob("*.jpg"))
    
    if len(image_files) == 0:
        print(f"No images found in {original_dir}")
        return
    
    print(f"Processing {len(image_files)} images from {clip_dir.name}")
    
    # Load images
    images = load_and_preprocess_images(image_files).to(device)
    
    # Run inference
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            images_batch = images[None]  # Add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            
            # Predict depth maps
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
            
            # Predict cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
            
            # Convert depth to point map
            point_map = unproject_depth_map_to_point_map(
                depth_map.squeeze(0),
                extrinsic.squeeze(0),
                intrinsic.squeeze(0)
            )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy - check if already numpy or tensor
    if isinstance(point_map, torch.Tensor):
        point_map_np = point_map.cpu().numpy()
    else:
        point_map_np = point_map
    
    if isinstance(depth_conf, torch.Tensor):
        depth_conf_np = depth_conf.squeeze(0).cpu().numpy()
    else:
        depth_conf_np = depth_conf.squeeze(0) if len(depth_conf.shape) > 3 else depth_conf
    
    extrinsic_np = extrinsic.squeeze(0).cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).cpu().numpy()
    depth_map_np = depth_map.squeeze(0).cpu().numpy()
    
    # Save point cloud as PLY
    output_file = output_dir / "point_cloud.ply"
    save_point_cloud(point_map_np, depth_conf_np, output_file)
    
    # Save individual .npy files
    np.save(output_dir / "extrinsics.npy", extrinsic_np)
    np.save(output_dir / "intrinsics.npy", intrinsic_np)
    np.save(output_dir / "point_map.npy", point_map_np)
    np.save(output_dir / "depth_map.npy", depth_map_np)
    np.save(output_dir / "depth_conf.npy", depth_conf_np)
    
    # Save everything in a single .npz file
    npz_output = output_dir / "vggt_output.npz"
    np.savez_compressed(
        npz_output,
        point_map=point_map_np,
        depth_map=depth_map_np,
        depth_conf=depth_conf_np,
        extrinsics=extrinsic_np,
        intrinsics=intrinsic_np,
        image_files=[str(f.name) for f in image_files]
    )
    print(f"Saved npz file to {npz_output}")

def main():
    # Get paths from command line arguments
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
    
    # Load model
    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Process video
    video_dir = input_dir / video_id
    output_video_dir = output_base / video_id
    
    if not video_dir.exists():
        print(f"Error: Video directory {video_dir} does not exist!")
        sys.exit(1)
    
    # Get all clip directories
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

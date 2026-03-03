import numpy as np
from PIL import Image
import json
import open3d as o3d
import sys
from pathlib import Path


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

    print(f"  Points visible in image: {len(x_coords)}")
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
# Process one clip output directory
# ============================================
def process_clip_dir(clip_output_dir: Path):
    ply_path  = clip_output_dir / "point_cloud_o3d.ply"
    json_path = clip_output_dir / "camera_poses.json"

    if not ply_path.exists() or not json_path.exists():
        print(f"  Skipping {clip_output_dir.name}: missing ply or json")
        return

    print(f"\nLoading point cloud from {ply_path.name} ...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    print(f"  Loaded {len(points)} points")

    with open(json_path) as f:
        camera_data = json.load(f)

    point_cloud_center = points.mean(axis=0)
    print(f"  Point cloud center: {point_cloud_center}")

    camera_info   = camera_data["cameras"][0]
    extrinsics_0  = np.array(camera_info["extrinsics"])
    intrinsics_0  = np.array(camera_info["intrinsics"])

    views = {
        "ground_0deg":   (extrinsics_0, intrinsics_0),
        "elevated_45deg": build_45_camera(extrinsics_0, intrinsics_0, point_cloud_center),
        "elevated_110deg": build_110_camera(extrinsics_0, intrinsics_0, point_cloud_center),
    }

    for name, (ext, intr) in views.items():
        print(f"\n  Rendering {name} ...")
        img = render_from_camera(points, colors, ext, intr, point_size=2)
        out_path = clip_output_dir / f"{name}.png"
        Image.fromarray(img).save(out_path)
        print(f"  [OK] Saved -> {out_path.name}")


# ============================================
# Main
# ============================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python render_elevated_views.py <output_dir>")
        print("  <output_dir> : Root vggt-output directory (same as inference_vggt.py output)")
        sys.exit(1)

    output_base = Path(sys.argv[1])
    if not output_base.exists():
        print(f"Error: {output_base} does not exist!")
        sys.exit(1)

    # Find all directories that contain point_cloud_o3d.ply
    clip_dirs = sorted(p.parent for p in output_base.rglob("point_cloud_o3d.ply"))

    if not clip_dirs:
        print(f"No point cloud outputs found under {output_base}")
        sys.exit(1)

    print(f"Found {len(clip_dirs)} clip output directories to render")

    success, fail = 0, 0
    for i, clip_dir in enumerate(clip_dirs, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(clip_dirs)}] {clip_dir.relative_to(output_base)}")
        print(f"{'='*60}")
        try:
            process_clip_dir(clip_dir)
            success += 1
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            fail += 1

    print(f"\n{'='*60}")
    print("DONE")
    print(f"  Rendered: {success}  |  Failed: {fail}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import json
import argparse
import random
import sys
import numpy as np
try:
    import mathutils
except ImportError:
    raise ImportError("This script requires 'mathutils' (from Blender). "
                      "Run it in Blender's Python environment or install a compatible backport.")
import math

def print_progress_bar(current, total, length=50):
    """Print a simple progress bar to stdout."""
    if total == 0:
        return
    percent = float(current) / total
    arrow = '-' * int(round(percent * length) - 1) + '>'
    spaces = ' ' * (length - len(arrow))
    sys.stdout.write(f"\rProgress: [{arrow}{spaces}] {int(percent * 100)}% ({current}/{total})")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Sample N unique SE(3) poses pointing to origin using mathutils.")
    parser.add_argument("--name", type=str, required=True, help="Base name for output files")
    parser.add_argument("--N", type=int, default=10, help="Number of poses to sample")
    parser.add_argument("--radius_min", type=float, default=1.8, help="Min radius")
    parser.add_argument("--radius_max", type=float, default=2.2, help="Max radius")
    parser.add_argument("--elev_min", type=float, default=-90, help="Min elevation (degrees)")
    parser.add_argument("--elev_max", type=float, default=90, help="Max elevation (degrees)")
    parser.add_argument("--azim_min", type=float, default=0, help="Min azimuth (degrees)")
    parser.add_argument("--azim_max", type=float, default=360, help="Max azimuth (degrees)")
    parser.add_argument("--seed", type=int, default=111, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for .npy files")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("./view", exist_ok=True)

    poses = []
    seen_poses = []

    target_count = args.N
    attempt = 0
    max_attempts = args.N * 100  # avoid infinite loop

    print(f"Sampling {args.N} unique poses (seed={args.seed})...")
    print_progress_bar(0, args.N)

    while len(poses) < target_count and attempt < max_attempts:
        attempt += 1

        # Sample spherical coordinates
        radius = random.uniform(args.radius_min, args.radius_max)
        elevation = random.uniform(args.elev_min, args.elev_max)
        azimuth = random.uniform(args.azim_min, args.azim_max)

        elev_rad = math.radians(elevation)
        azim_rad = math.radians(azimuth)

        x = radius * math.cos(elev_rad) * math.cos(azim_rad)
        y = radius * math.sin(elev_rad)
        z = radius * math.cos(elev_rad) * math.sin(azim_rad)

        location = mathutils.Vector((x, y, z))
        direction = -location
        if direction.length == 0:
            direction = mathutils.Vector((0, 0, -1))
        else:
            direction.normalize()

        quat = direction.to_track_quat('-Z', 'Y')
        rot_matrix = quat.to_matrix().to_4x4()
        c2w = mathutils.Matrix.Translation(location) @ rot_matrix
        c2w_np = np.array(c2w).astype(np.float32)

        # Check duplicate
        is_duplicate = False
        for existing in seen_poses:
            if np.allclose(c2w_np, existing, atol=1e-5):
                is_duplicate = True
                break

        if not is_duplicate:
            seen_poses.append(c2w_np)

            euler_rad = [float(angle) for angle in quat.to_euler()]  # XYZ order, radians

            npy_path = os.path.join(args.output_dir, f"{args.name}_{len(poses)}.npy")
            np.save(npy_path, c2w_np)

            poses.append({
                "id": len(poses),
                "euler": euler_rad,
                "transform": c2w_np.tolist()
            })

            # Update progress bar
            print_progress_bar(len(poses), args.N)

    print()  # newline after progress bar

    if len(poses) < args.N:
        print(f"Warning: Only generated {len(poses)} unique poses out of {args.N} requested (after {attempt} attempts).")

    # Save JSON
    json_path = "./view/all_views.json"
    with open(json_path, 'w') as f:
        json.dump(poses, f, indent=2)

    print(f"\n✅ Done!")
    print(f"Generated {len(poses)} poses.")
    print(f"JSON saved to: {json_path}")
    print(f".npy files saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()
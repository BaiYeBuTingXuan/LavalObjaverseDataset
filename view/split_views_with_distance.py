#!/usr/bin/env python3
import os
import json
import argparse
import sys
import numpy as np
from pathlib import Path

def rotation_from_c2w(c2w):
    return c2w[:3, :3]

def chordal_distance(R1, R2):
    """Chordal distance on SO(3): ||R1 - R2||_F / sqrt(2) ∈ [0, 2]"""
    return np.linalg.norm(R1 - R2, 'fro') / np.sqrt(2)

def print_progress_bar(current, total, prefix="Progress", length=50):
    """Print a simple progress bar to stdout."""
    if total <= 0:
        return
    percent = float(current) / total
    arrow = '-' * int(round(percent * length) - 1) + '>'
    spaces = ' ' * (length - len(arrow))
    sys.stdout.write(f"\r{prefix}: [{arrow}{spaces}] {int(percent * 100)}% ({current}/{total})")
    sys.stdout.flush()

def farthest_point_sampling_rotations(rotations):
    n = len(rotations)
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Precompute full distance matrix
    print("Computing pairwise chordal distances...")
    dists = np.zeros((n, n))
    total_pairs = n * (n - 1) // 2
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = chordal_distance(rotations[i], rotations[j])
            dists[i, j] = d
            dists[j, i] = d
            count += 1
            # Optional: show progress for large N
            if n > 100 and count % (total_pairs // 20 or 1) == 0:
                print_progress_bar(count, total_pairs, prefix="Distance computation")

    if n > 100:
        print()  # newline after distance computation

    selected = [0]
    remaining = list(range(1, n))
    
    print("Running Farthest Point Sampling (FPS)...")
    while remaining:
        min_dists = []
        for r in remaining:
            d_to_selected = min(dists[r, s] for s in selected)
            min_dists.append(d_to_selected)
        next_idx = remaining[np.argmax(min_dists)]
        selected.append(next_idx)
        remaining.remove(next_idx)
        
        # Update FPS progress
        print_progress_bar(len(selected), n, prefix="FPS")
    
    print()  # newline after FPS
    return selected

def compute_average_pairwise_distance(set1_poses, set2_poses):
    """Compute average chordal distance between two sets of poses (based on rotation)."""
    if not set1_poses or not set2_poses:
        return float('nan')
    total = 0.0
    count = 0
    n1, n2 = len(set1_poses), len(set2_poses)
    total_pairs = n1 * n2
    processed = 0
    
    for p1 in set1_poses:
        R1 = rotation_from_c2w(p1)
        for p2 in set2_poses:
            R2 = rotation_from_c2w(p2)
            total += chordal_distance(R1, R2)
            count += 1
            processed += 1
            # Show progress only if large
            if total_pairs > 1000 and processed % (total_pairs // 20 or 1) == 0:
                print_progress_bar(processed, total_pairs, prefix="Distance calc")
    
    if total_pairs > 1000:
        print()
    return total / count if count > 0 else float('nan')

def main():
    parser = argparse.ArgumentParser(
        description="Split poses into train/val/test with uniform coverage and compute inter-set distances."
    )
    parser.add_argument("data_dir", type=str, help="Directory containing .npy pose files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility of initial shuffle if needed)")
    parser.add_argument("--output_prefix", type=str, default="", help="Prefix for output JSON files (e.g., 'scene1_')")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    npy_files = sorted([f for f in data_path.iterdir() if f.suffix.lower() == '.npy'])
    
    if not npy_files:
        raise ValueError(f"No .npy files found in {args.data_dir}")
    
    # Load poses
    poses = []
    filenames = []
    for f in npy_files:
        try:
            c2w = np.load(f).astype(np.float64)
            if c2w.shape != (4, 4):
                continue
            poses.append(c2w)
            filenames.append(f.name)
        except Exception as e:
            print(f"Warning: skipping {f} due to error: {e}")
            continue

    if len(poses) < 3:
        raise ValueError("Need at least 3 valid poses to split into three sets.")
    
    # Extract rotations
    rotations = [rotation_from_c2w(p) for p in poses]
    
    # FPS ordering based on rotation
    ordered_indices = farthest_point_sampling_rotations(rotations)
    ordered_filenames = [filenames[i] for i in ordered_indices]
    ordered_poses = [poses[i] for i in ordered_indices]
    
    # Assign in round-robin 8:1:1 pattern
    assignments = []
    for i in range(len(ordered_filenames)):
        mod = i % 10
        if mod < 8:
            assignments.append('train')
        elif mod == 8:
            assignments.append('val')
        else:
            assignments.append('test')
    
    # Group
    train_files, val_files, test_files = [], [], []
    train_poses, val_poses, test_poses = [], [], []
    
    for fname, group, pose in zip(ordered_filenames, assignments, ordered_poses):
        if group == 'train':
            train_files.append(fname)
            train_poses.append(pose)
        elif group == 'val':
            val_files.append(fname)
            val_poses.append(pose)
        else:
            test_files.append(fname)
            test_poses.append(pose)
    
    # Sort for readability
    train_files.sort()
    val_files.sort()
    test_files.sort()
    
    # Save three JSONs
    prefix = args.output_prefix
    with open(f"{prefix}training_view.json", 'w') as f:
        json.dump(train_files, f, indent=2)
    with open(f"{prefix}valid_view.json", 'w') as f:
        json.dump(val_files, f, indent=2)
    with open(f"{prefix}test_view.json", 'w') as f:
        json.dump(test_files, f, indent=2)
    
    print("\n✅ Split completed. Output files:")
    print(f"  - {prefix}training_view.json ({len(train_files)} views)")
    print(f"  - {prefix}valid_view.json ({len(val_files)} views)")
    print(f"  - {prefix}test_view.json ({len(test_files)} views)")
    
    # Compute average pairwise distances between sets
    print("\n📊 Computing average chordal distances between sets (based on rotation):")
    
    d_train_val = compute_average_pairwise_distance(train_poses, val_poses)
    d_train_test = compute_average_pairwise_distance(train_poses, test_poses)
    d_val_test = compute_average_pairwise_distance(val_poses, test_poses)
    
    print(f"  Train ↔ Val:   {d_train_val:.4f}")
    print(f"  Train ↔ Test:  {d_train_test:.4f}")
    print(f"  Val   ↔ Test:  {d_val_test:.4f}")
    
    # Save distance report
    distance_report = {
        "train_val": float(d_train_val),
        "train_test": float(d_train_test),
        "val_test": float(d_val_test)
    }
    with open(f"{prefix}distance_report.json", 'w') as f:
        json.dump(distance_report, f, indent=2)
    print(f"\n📄 Distance report saved to: {prefix}distance_report.json")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import random

def main():
    parser = argparse.ArgumentParser(description="Split envmaps into Train/Val/Test (90/5/5) by original source.")
    parser.add_argument("source_root", help="Root directory of preprocessed EXR envmaps (with rotated versions)")
    parser.add_argument("--output_dir", type=str, default="./laval/info/", help="Directory to save output JSON files (default: current dir)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    random.seed(args.seed)

    # Structure: { category: { source_stem: [list of all rotated .exr paths] } }
    category_groups = {}

    # Walk through all .exr files
    for exr_path in source_root.rglob("*.exr"):
        try:
            rel_path = exr_path.relative_to(source_root)
            if len(rel_path.parts) < 2:
                continue  # skip if not in a subfolder
            
            category = rel_path.parts[0]
            source_stem = rel_path.parts[1]

            if category not in category_groups:
                category_groups[category] = {}
            if source_stem not in category_groups[category]:
                category_groups[category][source_stem] = []

            category_groups[category][source_stem].append(str(rel_path))

        except Exception as e:
            print(f"Warning: Skipping {exr_path}: {e}")
            continue

    # Prepare output dicts
    train_dict = {}
    val_dict = {}
    test_dict = {}

    for category, stem_dict in category_groups.items():
        stems = list(stem_dict.keys())
        random.shuffle(stems)

        n = len(stems)
        n_train = int(0.9 * n)
        n_val = int(0.05 * n)
        train_stems = stems[:n_train]
        val_stems = stems[n_train:n_train + n_val]
        test_stems = stems[n_train + n_val:]

        train_dict[category] = sorted([exr for stem in train_stems for exr in stem_dict[stem]])
        val_dict[category] = sorted([exr for stem in val_stems for exr in stem_dict[stem]])
        test_dict[category] = sorted([exr for stem in test_stems for exr in stem_dict[stem]])

        print(f"Category '{category}': {len(train_stems)} train, {len(val_stems)} val, {len(test_stems)} test sources")

    # Save JSONs to output_dir
    train_path = output_dir / "full_training_lighting.json"
    val_path = output_dir / "full_validation_lighting.json"
    test_path = output_dir / "full_testing_lighting.json"

    with open(train_path, "w") as f:
        json.dump(train_dict, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_dict, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_dict, f, indent=2)

    total_train = sum(len(v) for v in train_dict.values())
    total_val = sum(len(v) for v in val_dict.values())
    total_test = sum(len(v) for v in test_dict.values())

    print("\n✅ Split completed!")
    print(f"Training:   {total_train} files")
    print(f"Validation: {total_val} files")
    print(f"Test:       {total_test} files")
    print(f"\nOutput saved to: {output_dir.resolve()}")
    print(f"  - {train_path.name}")
    print(f"  - {val_path.name}")
    print(f"  - {test_path.name}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import json
import random
import argparse
from pathlib import Path

def split_json_list(input_file, output_prefix, num_splits, seed=None):
    """
    Split a JSON list into multiple randomly shuffled sublists.
    
    Args:
        input_file (str): Path to input JSON file containing a list
        output_prefix (str): Prefix for output filenames
        num_splits (int): Number of output files to create
        seed (int, optional): Random seed for reproducibility
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Read input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate input is a list
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a top-level list")
    
    if not data:
        raise ValueError("Input list is empty")
    
    if num_splits < 1:
        raise ValueError("Number of splits must be at least 1")
    
    # Shuffle the list
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    # Calculate split sizes
    total_items = len(shuffled)
    base_size = total_items // num_splits
    remainder = total_items % num_splits
    
    # Create splits
    splits = []
    start = 0
    for i in range(num_splits):
        # Distribute remainder items across first 'remainder' splits
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        splits.append(shuffled[start:end])
        start = end
    
    # Write output files
    output_files = []
    for i, split_data in enumerate(splits):
        output_path = f"{output_prefix}_{i+1}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        output_files.append(output_path)
        print(f"Created {output_path} with {len(split_data)} items")
    
    print(f"\nSplit {total_items} items into {num_splits} files")
    if seed is not None:
        print(f"Used random seed: {seed}")
    
    return output_files

def main():
    parser = argparse.ArgumentParser(
        description="Split a JSON list into multiple randomly shuffled sublists"
    )
    parser.add_argument(
        "input_file",
        help="Input JSON file containing a list"
    )
    parser.add_argument(
        "-o", "--output-prefix",
        default="split",
        help="Prefix for output filenames (default: split)"
    )
    parser.add_argument(
        "-n", "--num-splits",
        type=int,
        required=True,
        help="Number of output files to create"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    try:
        split_json_list(
            args.input_file,
            args.output_prefix,
            args.num_splits,
            args.seed
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
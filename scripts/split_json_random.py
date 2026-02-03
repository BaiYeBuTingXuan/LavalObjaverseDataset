#!/usr/bin/env python3
import json
import random
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Randomly split a JSON list into two parts.")
    parser.add_argument("input_json", type=Path, help="Input JSON file (must contain a list)")
    parser.add_argument("--count", type=int, help="Number of items to select (use either --count or --fraction)")
    parser.add_argument("--fraction", type=float, help="Fraction of items to select (between 0.0 and 1.0)")
    parser.add_argument("--output-selected", type=Path, required=True, help="Output JSON for selected items")
    parser.add_argument("--output-remaining", type=Path, required=True, help="Output JSON for remaining items")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    if (args.count is None) == (args.fraction is None):
        parser.error("Specify exactly one of --count or --fraction.")

    # Load input
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be a list. Got {type(data).__name__}.")

    n = len(data)
    if n == 0:
        raise ValueError("Input list is empty.")

    # Determine number to select
    if args.count is not None:
        k = args.count
        if k < 0:
            raise ValueError("--count must be non-negative.")
        if k > n:
            raise ValueError(f"--count ({k}) exceeds total items ({n}).")
    else:
        if not (0.0 <= args.fraction <= 1.0):
            raise ValueError("--fraction must be between 0.0 and 1.0.")
        k = int(round(args.fraction * n))
        k = max(0, min(k, n))  # clamp

    # Shuffle and split
    random.seed(args.seed)
    indices = list(range(n))
    random.shuffle(indices)
    selected_indices = set(indices[:k])
    
    selected = [data[i] for i in range(n) if i in selected_indices]
    remaining = [data[i] for i in range(n) if i not in selected_indices]

    # Save outputs
    with open(args.output_selected, 'w', encoding='utf-8') as f:
        json.dump(selected, f, indent=2)
    with open(args.output_remaining, 'w', encoding='utf-8') as f:
        json.dump(remaining, f, indent=2)

    print(f"Total items: {n}")
    print(f"Selected: {len(selected)} → saved to {args.output_selected}")
    print(f"Remaining: {len(remaining)} → saved to {args.output_remaining}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Optional imports — will error if not installed, but only when needed
try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE = True
except ImportError:
    OPENEXR_AVAILABLE = False

try:
    import imageio.v3 as iio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


def read_exr_pixels(filepath):
    """Read OpenEXR (.exr) file and return flattened pixel array, height, width."""
    if not OPENEXR_AVAILABLE:
        raise RuntimeError("OpenEXR not installed. Install with: pip install OpenEXR")

    exr_file = OpenEXR.InputFile(str(filepath))
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Prefer RGB(A); fallback to first 3 float channels
    channels = [ch for ch in ['R', 'G', 'B', 'A'] if ch in header['channels']]
    if not channels:
        float_channels = [
            ch for ch, info in header['channels'].items()
            if info.type == Imath.PixelType(Imath.PixelType.FLOAT)
        ]
        channels = float_channels[:3] if float_channels else list(header['channels'].keys())[:3]

    if not channels:
        raise ValueError(f"No valid channels found in {filepath}")

    pixel_arrays = []
    for ch in channels:
        try:
            data = exr_file.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
            arr = np.frombuffer(data, dtype=np.float32).reshape((height, width))
            pixel_arrays.append(arr)
        except Exception as e:
            print(f"Warning: Skipping channel {ch} in {filepath}: {e}", file=sys.stderr)
            continue

    if not pixel_arrays:
        raise ValueError(f"No readable channels in {filepath}")

    combined = np.stack(pixel_arrays, axis=-1)  # (H, W, C)
    return combined.flatten(), height, width


def read_hdr_pixels(filepath):
    """Read Radiance HDR (.hdr) file and return flattened pixel array, height, width."""
    if not IMAGEIO_AVAILABLE:
        raise RuntimeError("imageio not installed. Install with: pip install 'imageio[full]'")

    try:
        img = iio.imread(filepath, plugin='HDR-FI')
    except Exception:
        # Fallback: newer imageio may auto-detect
        img = iio.imread(filepath)

    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]  # Keep only first 3 channels

    height, width = img.shape[:2]
    return img.astype(np.float32).flatten(), height, width


def read_pixels_by_extension(filepath):
    """Dispatch to correct reader based on file extension."""
    suffix = filepath.suffix.lower()
    if suffix == '.exr':
        return read_exr_pixels(filepath)
    elif suffix == '.hdr':
        return read_hdr_pixels(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")


def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Compute statistics for .exr and .hdr files.")
    parser.add_argument("root_path", help="Root directory to scan")
    args = parser.parse_args()

    root = Path(args.root_path)
    if not root.exists():
        print(f"Error: Path '{root}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Collect both .exr and .hdr
    files = [f for ext in ('*.exr', '*.hdr') for f in root.rglob(ext)]
    if not files:
        print(f"No .exr or .hdr files found under '{root}'.")
        return

    print(f"Found {len(files)} files (.exr/.hdr). Processing...\n")

    all_means = []
    all_medians = []
    all_maxes = []
    all_sizes_mb = []
    all_heights = []
    all_widths = []
    processed_count = 0

    for i, fp in enumerate(files, 1):
        try:
            print(f"[{i}/{len(files)}] {fp.name}", end='\r')
            pixels, h, w = read_pixels_by_extension(fp)
            
            all_means.append(np.mean(pixels))
            all_medians.append(np.median(pixels))
            all_maxes.append(np.max(pixels))
            all_sizes_mb.append(get_file_size_mb(fp))
            all_heights.append(h)
            all_widths.append(w)
            processed_count += 1
        except Exception as e:
            print(f"\nError processing {fp}: {e}", file=sys.stderr)
            continue

    print("\n" + "="*60)
    print("EXR/HDR FILE STATISTICS")
    print("="*60)
    
    if processed_count == 0:
        print("No valid files were successfully processed.")
        return

    print(f"Total files processed: {processed_count} / {len(files)}")
    print()
    print(f"Average of mean values:   {np.mean(all_means):.6f}")
    print(f"Average of median values: {np.mean(all_medians):.6f}")
    print(f"Average of max values:    {np.mean(all_maxes):.6f}")
    print()
    print(f"Average file size:        {np.mean(all_sizes_mb):.2f} MB")
    print(f"Max file size:            {np.max(all_sizes_mb):.2f} MB")
    print()
    print(f"Resolution (W x H):")
    print(f"  Average width:          {np.mean(all_widths):.1f}")
    print(f"  Average height:         {np.mean(all_heights):.1f}")
    print(f"  Min resolution:         {min(all_widths)} x {min(all_heights)}")
    print(f"  Max resolution:         {max(all_widths)} x {max(all_heights)}")


if __name__ == "__main__":
    main()
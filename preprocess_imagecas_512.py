#!/usr/bin/env python3
"""preprocess_imagecas_512_fast.py – OPTIMIZED preprocessing for TransUNet

Performance optimizations:
- Parallel processing with multiprocessing
- Skip resize when not needed (512→512)
- Batch I/O operations
- Memory-efficient slice processing
- Optional: disable compression for faster writes

Usage:
python preprocess_imagecas_512_fast.py \
    --image-dir ~/data/train \
    --label-dir ~/data/label \
    --out-train ~/data/train_processed_512 \
    --out-test ~/data/val_processed_512 \
    --lists-dir ~/lists_512 \
    --workers 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------


def _expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def get_case_id(path: Path) -> str:
    """Extract clean case ID from various naming conventions."""
    name = path.name
    if '.img.nii.gz' in name:
        return name.replace('.img.nii.gz', '')
    if '.label.nii.gz' in name:
        return name.replace('.label.nii.gz', '')
    if name.endswith('.nii.gz'):
        return name[:-7]
    return path.stem


def resize_slice(arr: np.ndarray, target_size: tuple, is_label: bool):
    """Resize 2D slice - SKIPS if already correct size."""
    # Skip resize if already target size
    if arr.shape[0] == target_size[0] and arr.shape[1] == target_size[1]:
        if is_label:
            return arr.astype(np.uint8)
        else:
            return arr.astype(np.float32)
    
    zoom_factors = (target_size[0] / arr.shape[0], target_size[1] / arr.shape[1])
    
    if is_label:
        out = zoom(arr.astype(np.float32), zoom_factors, order=0)
        return out.astype(np.uint8)
    else:
        out = zoom(arr.astype(np.float32), zoom_factors, order=1)
        return out.astype(np.float32)


# ----------------------------------------------------------------------------
# Pair collection
# ----------------------------------------------------------------------------


def collect_pairs(img_dir: Path, lbl_dir: Path):
    """Collect matching image-label pairs from directories."""
    pairs = []
    
    for img in sorted(img_dir.glob("*.img.nii.gz")):
        lbl = lbl_dir / img.name.replace(".img.nii.gz", ".label.nii.gz")
        if lbl.exists():
            pairs.append((img, lbl))
    
    if not pairs:
        for img in sorted(img_dir.glob("*.nii.gz")):
            lbl = lbl_dir / img.name
            if lbl.exists():
                pairs.append((img, lbl))
    
    if not pairs:
        for img in sorted(img_dir.glob("*.mha")):
            lbl = lbl_dir / img.name
            if lbl.exists():
                pairs.append((img, lbl))
    
    if not pairs:
        raise RuntimeError(f"No matching pairs found in {img_dir} and {lbl_dir}")
    
    return list(zip(*pairs))


# ----------------------------------------------------------------------------
# Single volume processing (for parallel execution)
# ----------------------------------------------------------------------------


def process_single_volume(args_tuple):
    """Process a single volume - designed for parallel execution.
    
    Args:
        args_tuple: (img_path, lbl_path, out_dir, target_size, hu_min, hu_max, compress)
    
    Returns:
        Tuple of (case_id, num_slices, slice_names, success)
    """
    img_p, lbl_p, out_dir, target_size, hu_min, hu_max, compress = args_tuple
    
    try:
        case_id = get_case_id(img_p)
        
        # Load volumes
        vol = sitk.ReadImage(str(img_p))
        lbl = sitk.ReadImage(str(lbl_p))
        
        # Quick geometry validation
        if vol.GetSize() != lbl.GetSize():
            return (case_id, 0, [], False, f"Size mismatch")
        
        # Extract metadata
        sitk_size = vol.GetSize()
        sitk_spacing = vol.GetSpacing()
        sitk_origin = vol.GetOrigin()
        sitk_direction = vol.GetDirection()
        
        size_x, size_y, size_z = sitk_size
        spacing_x, spacing_y, spacing_z = sitk_spacing
        
        # Convert to numpy
        v_arr = sitk.GetArrayFromImage(vol)
        l_arr = sitk.GetArrayFromImage(lbl)
        
        num_slices, height, width = v_arr.shape
        target_H, target_W = target_size
        
        # Calculate effective spacing
        resize_factor_y = height / target_H
        resize_factor_x = width / target_W
        effective_spacing_x = spacing_x * resize_factor_x
        effective_spacing_y = spacing_y * resize_factor_y
        effective_spacing_z = spacing_z
        
        # Normalize intensities
        v_arr = np.clip(v_arr, hu_min, hu_max)
        v_arr = (v_arr - hu_min) / (hu_max - hu_min)
        
        # Save metadata
        meta_name = f"{case_id}_metadata.npz"
        np.savez(out_dir / meta_name,
                 sitk_size=np.array(sitk_size),
                 sitk_spacing=np.array(sitk_spacing),
                 sitk_origin=np.array(sitk_origin),
                 sitk_direction=np.array(sitk_direction),
                 numpy_shape=np.array(v_arr.shape),
                 target_size=np.array(target_size),
                 resize_factors=np.array([resize_factor_y, resize_factor_x]),
                 effective_spacing=np.array([effective_spacing_x, effective_spacing_y, effective_spacing_z]),
                 hu_min=hu_min,
                 hu_max=hu_max,
                 num_slices=num_slices,
                 case_id=case_id,
                 source_image=str(img_p),
                 source_label=str(lbl_p))
        
        # Process slices
        slice_names = []
        for z in range(num_slices):
            img_sl = resize_slice(v_arr[z], target_size, is_label=False)
            lbl_sl = resize_slice(l_arr[z], target_size, is_label=True)
            
            name = f"{case_id}_slice{z:04d}.npz"
            
            # Save with or without compression
            if compress:
                np.savez_compressed(out_dir / name, image=img_sl, label=lbl_sl)
            else:
                np.savez(out_dir / name, image=img_sl, label=lbl_sl)
            
            slice_names.append(name)
        
        return (case_id, num_slices, slice_names, True, None)
    
    except Exception as e:
        return (get_case_id(img_p), 0, [], False, str(e))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="FAST preprocessing for TransUNet (parallel processing)"
    )
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-test", required=True)
    ap.add_argument("--lists-dir", required=True)
    ap.add_argument("--split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--hu-min", type=int, default=0)
    ap.add_argument("--hu-max", type=int, default=1000)
    ap.add_argument("--max-volumes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of parallel workers (default: CPU count)")
    ap.add_argument("--no-compress", action="store_true",
                    help="Disable compression for faster writes (larger files)")
    args = ap.parse_args()

    img_dir = _expand(args.image_dir)
    lbl_dir = _expand(args.label_dir)
    out_tr = _expand(args.out_train)
    out_te = _expand(args.out_test)
    lists = _expand(args.lists_dir)
    target_size = (args.size, args.size)
    compress = not args.no_compress
    
    # Determine worker count
    if args.workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    else:
        n_workers = args.workers

    rng = np.random.default_rng(args.seed)
    imgs, lbls = collect_pairs(img_dir, lbl_dir)
    
    # Limit volumes if requested
    if args.max_volumes is not None and args.max_volumes < len(imgs):
        subset_idx = rng.choice(len(imgs), size=args.max_volumes, replace=False)
        subset_idx = sorted(subset_idx)
        imgs = [imgs[i] for i in subset_idx]
        lbls = [lbls[i] for i in subset_idx]
        print(f"🔬 Limited to {args.max_volumes} volumes")
    
    # Split into train/test
    order = rng.permutation(len(imgs))
    n_test = max(1, int(len(order) * args.split)) if len(order) else 0
    test_idx, train_idx = set(order[:n_test]), set(order[n_test:])

    out_tr.mkdir(parents=True, exist_ok=True)
    out_te.mkdir(parents=True, exist_ok=True)
    lists.mkdir(parents=True, exist_ok=True)

    print(f"📦 {len(imgs)} volumes → train {len(train_idx)}, val {len(test_idx)}")
    print(f"🔬 Target size: {target_size[0]}×{target_size[1]}")
    print(f"📊 HU range: [{args.hu_min}, {args.hu_max}]")
    print(f"⚡ Workers: {n_workers}")
    print(f"📁 Compression: {'ON' if compress else 'OFF (faster)'}")

    # Prepare tasks
    train_tasks = []
    test_tasks = []
    
    for i in range(len(imgs)):
        if i in train_idx:
            train_tasks.append((imgs[i], lbls[i], out_tr, target_size, 
                               args.hu_min, args.hu_max, compress))
        else:
            test_tasks.append((imgs[i], lbls[i], out_te, target_size,
                              args.hu_min, args.hu_max, compress))

    # Process in parallel
    train_slices = []
    test_slices = []
    
    print(f"\n⏳ Processing {len(train_tasks)} training volumes...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_volume, task): task 
                   for task in train_tasks}
        
        for future in as_completed(futures):
            case_id, num_slices, slice_names, success, error = future.result()
            if success:
                train_slices.extend(slice_names)
                print(f"  ✅ {case_id}: {num_slices} slices")
            else:
                print(f"  ❌ {case_id}: {error}")

    print(f"\n⏳ Processing {len(test_tasks)} test volumes...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_volume, task): task 
                   for task in test_tasks}
        
        for future in as_completed(futures):
            case_id, num_slices, slice_names, success, error = future.result()
            if success:
                test_slices.extend(slice_names)
                print(f"  ✅ {case_id}: {num_slices} slices")
            else:
                print(f"  ❌ {case_id}: {error}")

    # Write list files
    with open(lists / "train.txt", "w") as f:
        for name in sorted(train_slices):
            f.write(name + "\n")
    
    with open(lists / "test_vol.txt", "w") as f:
        for name in sorted(test_slices):
            f.write(name + "\n")
    
    with open(lists / "all.lst", "w") as f:
        for img in imgs:
            f.write(get_case_id(img) + "\n")

    print(f"\n✅ Done!")
    print(f"   Training slices: {len(train_slices)}")
    print(f"   Test slices: {len(test_slices)}")
    print(f"   Output: {out_tr}, {out_te}")


if __name__ == "__main__":
    main()
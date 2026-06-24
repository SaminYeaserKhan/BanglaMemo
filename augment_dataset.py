#!/usr/bin/env python3
"""
Bangla Receipt Dataset — Augmentation & Finalization
=====================================================
Takes the cropped dataset from prepare_dataset.py and produces a
PaddleOCR-ready training package with:
  1. Train / Val / Test splits (85 / 10 / 5%)
  2. Data augmentation on training set only (4× per image)
  3. PaddleOCR-compatible label files & character dictionary
  4. Integrity verification

Usage:
    python3 augment_dataset.py
"""

import csv
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import numpy as np
except ImportError:
    import subprocess
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "numpy"])
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "Dataset_Final"
OUTPUT_DIR = BASE_DIR / "Dataset_PaddleOCR"

# Split ratios
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05

# Augmentation
AUG_PER_IMAGE = 4  # number of augmented copies per training image

# ─── Augmentation Functions ──────────────────────────────────────────────────

def augment_rotation(img: Image.Image) -> Image.Image:
    """Slight random rotation (±3°) with white fill."""
    angle = random.uniform(-3, 3)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True,
                      fillcolor=(255, 255, 255))


def augment_brightness(img: Image.Image) -> Image.Image:
    """Random brightness adjustment (±20%)."""
    factor = random.uniform(0.8, 1.2)
    return ImageEnhance.Brightness(img).enhance(factor)


def augment_contrast(img: Image.Image) -> Image.Image:
    """Random contrast adjustment (±20%)."""
    factor = random.uniform(0.8, 1.2)
    return ImageEnhance.Contrast(img).enhance(factor)


def augment_gaussian_noise(img: Image.Image) -> Image.Image:
    """Add Gaussian noise with random sigma (5-15)."""
    arr = np.array(img, dtype=np.float32)
    sigma = random.uniform(5, 15)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def augment_blur(img: Image.Image) -> Image.Image:
    """Slight Gaussian blur (radius 0.5-1.5)."""
    radius = random.uniform(0.5, 1.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def augment_perspective(img: Image.Image) -> Image.Image:
    """Slight random perspective warp (±2%)."""
    w, h = img.size
    # Maximum displacement is 2% of dimensions
    max_dx = w * 0.02
    max_dy = h * 0.02

    # Four corners: original → displaced
    coeffs = _find_perspective_coeffs(
        [(random.uniform(0, max_dx), random.uniform(0, max_dy)),
         (w - random.uniform(0, max_dx), random.uniform(0, max_dy)),
         (w - random.uniform(0, max_dx), h - random.uniform(0, max_dy)),
         (random.uniform(0, max_dx), h - random.uniform(0, max_dy))],
        [(0, 0), (w, 0), (w, h), (0, h)]
    )
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def _find_perspective_coeffs(source, target):
    """Compute perspective transform coefficients."""
    matrix = []
    for s, t in zip(source, target):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = np.matrix(matrix, dtype=np.float64)
    B = np.array([s for pair in source for s in pair], dtype=np.float64)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return np.array(res).reshape(8).tolist()


def augment_jpeg_compression(img: Image.Image) -> Image.Image:
    """Simulate JPEG compression artifacts (quality 50-85)."""
    quality = random.randint(50, 85)
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def _bilinear_warp(arr, x_coords, y_coords, dx, dy, h, w):
    """Apply bilinear interpolation warp given displacement fields dx, dy."""
    map_x = np.clip(x_coords + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y_coords + dy, 0, h - 1).astype(np.float32)

    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    fx = (map_x - x0).astype(np.float32)
    fy = (map_y - y0).astype(np.float32)

    if len(arr.shape) == 3:
        fx = fx[:, :, np.newaxis]
        fy = fy[:, :, np.newaxis]

    result = (
        arr[y0, x0] * (1 - fx) * (1 - fy) +
        arr[y0, x1] * fx * (1 - fy) +
        arr[y1, x0] * (1 - fx) * fy +
        arr[y1, x1] * fx * fy
    )
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def augment_paper_crumple(img: Image.Image) -> Image.Image:
    """
    Simulate paper crumpling/folding via elastic deformation.
    Uses sine-wave grid displacement plus localized fold lines
    to create realistic creases like a receipt carried in a pocket.

    Intensity is randomized per-call:
      - Mild:   2-3 waves, amplitude 2-3px, 0 folds
      - Medium: 3-4 waves, amplitude 3-5px, 0-1 fold
      - Strong: 4-5 waves, amplitude 4-7px, 1 fold
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    y_coords, x_coords = np.mgrid[0:h, 0:w]

    # Randomize intensity level (favor mild/medium to keep text readable)
    intensity = random.choices(
        ['mild', 'medium', 'strong'],
        weights=[0.35, 0.40, 0.25]
    )[0]

    if intensity == 'mild':
        n_waves = random.randint(2, 3)
        amp_range = (2.0, 3.0)
        n_folds = 0
    elif intensity == 'medium':
        n_waves = random.randint(3, 4)
        amp_range = (3.0, 5.0)
        n_folds = random.choice([0, 1])
    else:  # strong
        n_waves = random.randint(4, 5)
        amp_range = (4.0, 7.0)
        n_folds = 1

    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)

    for _ in range(n_waves):
        amplitude = random.uniform(*amp_range)
        freq_x = random.uniform(1.0, 3.5)
        freq_y = random.uniform(1.0, 3.5)
        phase_x = random.uniform(0, 2 * math.pi)
        phase_y = random.uniform(0, 2 * math.pi)
        angle = random.uniform(0, math.pi)

        wave_x = np.sin(2 * math.pi * freq_y * y_coords / h + phase_y) * amplitude
        wave_y = np.sin(2 * math.pi * freq_x * x_coords / w + phase_x) * amplitude

        dx += wave_x * math.cos(angle)
        dy += wave_y * math.sin(angle)

    # Add localized fold lines (sharp creases)
    for _ in range(n_folds):
        fold_pos = random.uniform(0.25, 0.75)
        fold_width_frac = random.uniform(0.04, 0.08)
        fold_amplitude = random.uniform(4, 8)
        is_horizontal = random.random() > 0.5

        if is_horizontal:
            fold_center = int(fold_pos * h)
            fold_sigma = max(1, int(fold_width_frac * h))
            y_dist = np.abs(y_coords - fold_center).astype(np.float32)
            fold_mask = np.exp(-(y_dist ** 2) / (2 * fold_sigma ** 2))
            dx += fold_mask * fold_amplitude * np.sin(
                2 * math.pi * x_coords / w * random.uniform(1.5, 2.5))
        else:
            fold_center = int(fold_pos * w)
            fold_sigma = max(1, int(fold_width_frac * w))
            x_dist = np.abs(x_coords - fold_center).astype(np.float32)
            fold_mask = np.exp(-(x_dist ** 2) / (2 * fold_sigma ** 2))
            dy += fold_mask * fold_amplitude * np.sin(
                2 * math.pi * y_coords / h * random.uniform(1.5, 2.5))

    return _bilinear_warp(arr, x_coords, y_coords, dx, dy, h, w)


def augment_ink_smudge(img: Image.Image) -> Image.Image:
    """
    Simulate ink smudging from handling/rubbing.
    Applies directional motion blur selectively to darker (ink) regions
    while preserving the lighter paper background. Includes ink bleeding
    (spread/dilation) for realistic wet-ink smearing.

    Intensity is randomized per-call:
      - Mild:   kernel 5-7,  strength 0.5-0.7, no bleed
      - Medium: kernel 7-11, strength 0.65-0.85, light bleed
      - Strong: kernel 9-13, strength 0.75-0.95, heavy bleed
    """
    arr = np.array(img, dtype=np.float32)

    # Randomize intensity level
    intensity = random.choices(
        ['mild', 'medium', 'strong'],
        weights=[0.25, 0.45, 0.30]
    )[0]

    if intensity == 'mild':
        kernel_size = random.choice([5, 7])
        smudge_strength = random.uniform(0.5, 0.7)
        bleed_strength = 0.0
    elif intensity == 'medium':
        kernel_size = random.choice([7, 9, 11])
        smudge_strength = random.uniform(0.65, 0.85)
        bleed_strength = random.uniform(0.15, 0.3)
    else:  # strong
        kernel_size = random.choice([9, 11, 13])
        smudge_strength = random.uniform(0.75, 0.95)
        bleed_strength = random.uniform(0.25, 0.45)

    # Create ink mask: pixels darker than threshold are "ink"
    gray = np.mean(arr, axis=2) if len(arr.shape) == 3 else arr
    ink_threshold = random.uniform(130, 170)
    ink_mask = (gray < ink_threshold).astype(np.float32)

    # Smooth the mask edges for natural blending
    mask_img = Image.fromarray((ink_mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))
    ink_mask = np.array(mask_img, dtype=np.float32) / 255.0

    # Create directional motion blur kernel
    angle = random.uniform(0, 180)  # smudge direction
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))

    for i in range(kernel_size):
        offset = i - center
        x = int(round(center + offset * cos_a))
        y = int(round(center + offset * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum

    # Apply motion blur using numpy convolution (per channel)
    def convolve2d_simple(image_2d, kern):
        """Simple 2D convolution with edge-padding."""
        h, w = image_2d.shape
        kh, kw = kern.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(image_2d, ((ph, ph), (pw, pw)), mode='edge')
        result = np.zeros_like(image_2d)
        for ky in range(kh):
            for kx in range(kw):
                result += padded[ky:ky+h, kx:kx+w] * kern[ky, kx]
        return result

    blurred_arr = np.zeros_like(arr)
    if len(arr.shape) == 3:
        for c in range(arr.shape[2]):
            blurred_arr[:, :, c] = convolve2d_simple(arr[:, :, c], kernel)
    else:
        blurred_arr = convolve2d_simple(arr, kernel)

    # Blend: use motion-blurred version for ink areas, original for paper
    if len(arr.shape) == 3:
        ink_mask_3d = ink_mask[:, :, np.newaxis]
    else:
        ink_mask_3d = ink_mask

    result = arr * (1 - ink_mask_3d * smudge_strength) + blurred_arr * (ink_mask_3d * smudge_strength)

    # Ink bleeding / spreading: darken and spread ink edges outward
    if bleed_strength > 0:
        result_img = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
        # MinFilter dilates dark regions (makes ink spread outward)
        bleed_size = 3 if intensity == 'medium' else 5
        dark_spread = result_img.filter(ImageFilter.MinFilter(size=bleed_size))
        dark_arr = np.array(dark_spread, dtype=np.float32)
        # Blend spread effect only where there's ink
        result = result * (1 - ink_mask_3d * bleed_strength) + dark_arr * (ink_mask_3d * bleed_strength)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def augment_image(img: Image.Image, aug_idx: int) -> Image.Image:
    """
    Apply a randomized combination of augmentations.
    Each aug_idx gets a slightly different mix to maximize diversity.

    Key constraint: crumple and smudge are NEVER both applied to the
    same image (combined effect is too extreme). At most one is used.
    """
    # Always apply brightness + contrast (subtle, always helps)
    result = augment_brightness(img)
    result = augment_contrast(result)

    # Geometric / noise augmentations (safe to combine freely)
    standard_augs = [
        augment_rotation,
        augment_gaussian_noise,
        augment_blur,
        augment_perspective,
        augment_jpeg_compression,
    ]

    # Receipt-specific augmentations (pick AT MOST one)
    receipt_augs = [
        augment_paper_crumple,
        augment_ink_smudge,
    ]

    # Decide whether to include a receipt-specific augmentation (75% chance)
    chosen = []
    if random.random() < 0.75:
        chosen.append(random.choice(receipt_augs))

    # Pick 1-2 standard augmentations to complement
    n_standard = random.choice([1, 2])
    chosen.extend(random.sample(standard_augs, n_standard))

    # Shuffle order for variety
    random.shuffle(chosen)

    for aug_fn in chosen:
        result = aug_fn(result)

    return result


# ─── Utility Functions ───────────────────────────────────────────────────────

def load_label_file(path: Path) -> list:
    """Load PaddleOCR label file → list of (image_path, transcription)."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2 and parts[1].strip():
                entries.append((parts[0], parts[1]))
    return entries


def split_receipts(entries: list, train_r: float, val_r: float) -> tuple:
    """
    Split entries by receipt ID to prevent data leakage.
    Returns (train_entries, val_entries, test_entries).
    """
    # Group by receipt ID
    receipt_map = defaultdict(list)
    for img_path, trans in entries:
        # Extract receipt number from filename
        basename = Path(img_path).stem  # e.g., "receipt_001_items"
        parts = basename.split('_')
        receipt_id = f"{parts[0]}_{parts[1]}"  # "receipt_001"
        receipt_map[receipt_id].append((img_path, trans))

    # Shuffle receipt IDs
    receipt_ids = sorted(receipt_map.keys())
    random.shuffle(receipt_ids)

    n = len(receipt_ids)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    train_ids = set(receipt_ids[:n_train])
    val_ids = set(receipt_ids[n_train:n_train + n_val])
    test_ids = set(receipt_ids[n_train + n_val:])

    train_entries = []
    val_entries = []
    test_entries = []

    for rid in receipt_ids:
        for entry in receipt_map[rid]:
            if rid in train_ids:
                train_entries.append(entry)
            elif rid in val_ids:
                val_entries.append(entry)
            else:
                test_entries.append(entry)

    return train_entries, val_entries, test_entries, train_ids, val_ids, test_ids


def copy_images(entries: list, src_subdir: str, dst_dir: Path):
    """Copy original images to the destination directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path, _ in entries:
        src = INPUT_DIR / img_path
        dst = dst_dir / Path(img_path).name
        if src.exists():
            shutil.copy2(src, dst)


def augment_and_save(entries: list, src_subdir: str, dst_dir: Path,
                     n_aug: int) -> list:
    """
    Generate augmented copies of training images.
    Returns list of (new_img_path, transcription) for augmented images.
    """
    aug_entries = []
    dst_dir.mkdir(parents=True, exist_ok=True)

    for img_path, trans in entries:
        src = INPUT_DIR / img_path
        if not src.exists():
            continue

        img = Image.open(src).convert('RGB')
        basename = Path(img_path).stem
        ext = Path(img_path).suffix

        for aug_i in range(n_aug):
            aug_img = augment_image(img, aug_i)
            aug_name = f"{basename}_aug{aug_i+1}{ext}"
            aug_dst = dst_dir / aug_name
            aug_img.save(aug_dst, quality=95)

            # Relative path for label file
            rel_path = f"{dst_dir.name}/{aug_name}"
            aug_entries.append((rel_path, trans))

    return aug_entries


def write_label_file(entries: list, path: Path):
    """Write PaddleOCR-format label file."""
    with open(path, 'w', encoding='utf-8') as f:
        for img_path, trans in entries:
            f.write(f"{img_path}\t{trans}\n")


def build_char_dict(all_entries: list) -> list:
    """Build character dictionary from all transcriptions."""
    chars = set()
    for _, trans in all_entries:
        for ch in trans:
            chars.add(ch)
    return sorted(chars)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Bangla Receipt Dataset — Augmentation & Finalization")
    print("=" * 60)
    print()

    # ─── Load existing label files ───────────────────────────────────────

    items_labels_path = INPUT_DIR / "rec_items_labels.txt"
    total_labels_path = INPUT_DIR / "rec_total_labels.txt"

    if not items_labels_path.exists() or not total_labels_path.exists():
        print("ERROR: Run prepare_dataset.py first to generate the cropped dataset.")
        sys.exit(1)

    items_entries = load_label_file(items_labels_path)
    total_entries = load_label_file(total_labels_path)

    print(f"Loaded {len(items_entries)} items entries")
    print(f"Loaded {len(total_entries)} total entries")

    # ─── Clean up ────────────────────────────────────────────────────────

    # Remove entries with empty transcriptions
    items_clean = [(p, t) for p, t in items_entries if t.strip()]
    total_clean = [(p, t) for p, t in total_entries if t.strip()]

    removed_items = len(items_entries) - len(items_clean)
    removed_total = len(total_entries) - len(total_clean)
    if removed_items + removed_total > 0:
        print(f"  Removed {removed_items} items + {removed_total} total entries with empty transcriptions")

    # ─── Create output directory ─────────────────────────────────────────

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # ─── Split by receipt ────────────────────────────────────────────────

    print(f"\nSplitting by receipt (Train {TRAIN_RATIO:.0%} / Val {VAL_RATIO:.0%} / Test {TEST_RATIO:.0%})...")

    items_train, items_val, items_test, train_ids, val_ids, test_ids = \
        split_receipts(items_clean, TRAIN_RATIO, VAL_RATIO)

    # Use the SAME receipt split for totals
    total_train, total_val, total_test = [], [], []
    for img_path, trans in total_clean:
        basename = Path(img_path).stem
        parts = basename.split('_')
        receipt_id = f"{parts[0]}_{parts[1]}"
        if receipt_id in train_ids:
            total_train.append((img_path, trans))
        elif receipt_id in val_ids:
            total_val.append((img_path, trans))
        else:
            total_test.append((img_path, trans))

    print(f"  Items split: Train={len(items_train)}, Val={len(items_val)}, Test={len(items_test)}")
    print(f"  Total split: Train={len(total_train)}, Val={len(total_val)}, Test={len(total_test)}")
    print(f"  Receipts:    Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # ─── Copy originals to split dirs ────────────────────────────────────

    print("\nCopying original images to split directories...")

    for split_name, items_split, total_split in [
        ("train", items_train, total_train),
        ("val", items_val, total_val),
        ("test", items_test, total_test),
    ]:
        split_dir = OUTPUT_DIR / split_name
        copy_images(items_split, "items_crops", split_dir / "items_crops")
        copy_images(total_split, "total_crops", split_dir / "total_crops")
        print(f"  {split_name}: {len(items_split)} items + {len(total_split)} totals copied")

    # ─── Augment training set ────────────────────────────────────────────

    print(f"\nAugmenting training set ({AUG_PER_IMAGE}× per image)...")

    train_dir = OUTPUT_DIR / "train"
    items_aug = augment_and_save(
        items_train, "items_crops",
        train_dir / "items_crops", AUG_PER_IMAGE
    )
    total_aug = augment_and_save(
        total_train, "total_crops",
        train_dir / "total_crops", AUG_PER_IMAGE
    )

    print(f"  Generated {len(items_aug)} augmented items crops")
    print(f"  Generated {len(total_aug)} augmented total crops")

    # ─── Build label files ───────────────────────────────────────────────

    print("\nWriting PaddleOCR label files...")

    # Training: originals + augmented
    train_items_all = []
    for img_path, trans in items_train:
        # Update path to be relative within split
        new_path = f"items_crops/{Path(img_path).name}"
        train_items_all.append((new_path, trans))
    train_items_all.extend(items_aug)

    train_total_all = []
    for img_path, trans in total_train:
        new_path = f"total_crops/{Path(img_path).name}"
        train_total_all.append((new_path, trans))
    train_total_all.extend(total_aug)

    write_label_file(train_items_all, train_dir / "rec_items_train.txt")
    write_label_file(train_total_all, train_dir / "rec_total_train.txt")

    # Validation & Test: originals only
    for split_name, items_split, total_split in [
        ("val", items_val, total_val),
        ("test", items_test, total_test),
    ]:
        split_dir = OUTPUT_DIR / split_name
        items_relabeled = [(f"items_crops/{Path(p).name}", t) for p, t in items_split]
        total_relabeled = [(f"total_crops/{Path(p).name}", t) for p, t in total_split]
        write_label_file(items_relabeled, split_dir / f"rec_items_{split_name}.txt")
        write_label_file(total_relabeled, split_dir / f"rec_total_{split_name}.txt")

    # ─── Character dictionary ────────────────────────────────────────────

    all_entries = items_clean + total_clean
    char_list = build_char_dict(all_entries)
    dict_path = OUTPUT_DIR / "char_dictionary.txt"
    with open(dict_path, 'w', encoding='utf-8') as f:
        for ch in char_list:
            f.write(ch + '\n')

    print(f"  Character dictionary: {len(char_list)} unique characters")

    # ─── Verification ────────────────────────────────────────────────────

    print("\nRunning integrity checks...")
    errors = []

    # Check all image paths in label files exist
    for split_name in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split_name
        for label_file in split_dir.glob("rec_*.txt"):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split('\t', 1)
                    if len(parts) < 2:
                        continue
                    img_rel = parts[0]
                    img_abs = split_dir / img_rel
                    if not img_abs.exists():
                        errors.append(f"{label_file.name}:{line_num} — image not found: {img_rel}")

    # Check no data leakage
    if train_ids & val_ids:
        errors.append(f"DATA LEAKAGE: {len(train_ids & val_ids)} receipts in both train and val!")
    if train_ids & test_ids:
        errors.append(f"DATA LEAKAGE: {len(train_ids & test_ids)} receipts in both train and test!")
    if val_ids & test_ids:
        errors.append(f"DATA LEAKAGE: {len(val_ids & test_ids)} receipts in both val and test!")

    # Check no empty transcriptions
    for split_name in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split_name
        for label_file in split_dir.glob("rec_*.txt"):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2 and not parts[1].strip():
                        errors.append(f"{label_file.name}:{line_num} — empty transcription")

    if errors:
        print(f"\n  ❌ {len(errors)} errors found:")
        for e in errors[:20]:
            print(f"    • {e}")
    else:
        print("  ✅ All image paths verified")
        print("  ✅ No data leakage between splits")
        print("  ✅ No empty transcriptions")

    # ─── Summary Report ──────────────────────────────────────────────────

    # Count files per split
    split_counts = {}
    for split_name in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split_name
        items_count = len(list((split_dir / "items_crops").glob("*.jpg")))
        total_count = len(list((split_dir / "total_crops").glob("*.jpg")))
        split_counts[split_name] = (items_count, total_count)

    report_lines = [
        "=" * 60,
        "BANGLA RECEIPT OCR DATASET — FINAL REPORT",
        "=" * 60,
        "",
        "DATASET OVERVIEW",
        "-" * 40,
        f"Original labeled receipts:       {len(items_clean)}",
        f"Original total crops:            {len(total_clean)}",
        f"Empty transcriptions removed:    {removed_items + removed_total}",
        f"Augmentation factor:             {AUG_PER_IMAGE}× (training only)",
        "",
        "SPLIT DISTRIBUTION",
        "-" * 40,
        f"{'Split':<10} {'Receipts':>10} {'Items Crops':>14} {'Total Crops':>14}",
        f"{'Train':<10} {len(train_ids):>10} {split_counts['train'][0]:>14} {split_counts['train'][1]:>14}",
        f"{'Val':<10} {len(val_ids):>10} {split_counts['val'][0]:>14} {split_counts['val'][1]:>14}",
        f"{'Test':<10} {len(test_ids):>10} {split_counts['test'][0]:>14} {split_counts['test'][1]:>14}",
        f"{'TOTAL':<10} {len(train_ids)+len(val_ids)+len(test_ids):>10} "
        f"{sum(v[0] for v in split_counts.values()):>14} "
        f"{sum(v[1] for v in split_counts.values()):>14}",
        "",
        "  (Train includes original + augmented; Val/Test are originals only)",
        "",
        "AUGMENTATION TECHNIQUES",
        "-" * 40,
        "Applied to training set in random combinations:",
        "  • Brightness adjustment (±20%)",
        "  • Contrast adjustment (±20%)",
        "  • Slight rotation (±3°)",
        "  • Gaussian noise (σ = 5–15)",
        "  • Gaussian blur (radius 0.5–1.5)",
        "  • Perspective warp (±2%)",
        "  • JPEG compression artifacts (quality 50–85)",
        "  • Paper crumpling/folding (multi-level: 3–10px waves + fold lines)",
        "  • Ink smudging (kernel 5–13px, motion blur + ink bleeding/spread)",
        "",
        "  Note: Crumple and smudge are mutually exclusive per image to",
        "  prevent over-distortion while maintaining text readability.",
        "",
        "CHARACTER DICTIONARY",
        "-" * 40,
        f"Unique characters:               {len(char_list)}",
        "",
        "PADDLEOCR FORMAT",
        "-" * 40,
        "Label format: <relative_image_path>\\t<transcription>",
        "Label files per split:",
        "  train/rec_items_train.txt, train/rec_total_train.txt",
        "  val/rec_items_val.txt, val/rec_total_val.txt",
        "  test/rec_items_test.txt, test/rec_total_test.txt",
        "",
        "INTEGRITY",
        "-" * 40,
        f"Verification errors:             {len(errors)}",
        f"Data leakage:                    None",
        "",
        "DIRECTORY STRUCTURE",
        "-" * 40,
        "Dataset_PaddleOCR/",
        "├── train/",
        "│   ├── items_crops/          (originals + augmented)",
        "│   ├── total_crops/          (originals + augmented)",
        "│   ├── rec_items_train.txt",
        "│   └── rec_total_train.txt",
        "├── val/",
        "│   ├── items_crops/          (originals only)",
        "│   ├── total_crops/          (originals only)",
        "│   ├── rec_items_val.txt",
        "│   └── rec_total_val.txt",
        "├── test/",
        "│   ├── items_crops/          (originals only)",
        "│   ├── total_crops/          (originals only)",
        "│   ├── rec_items_test.txt",
        "│   └── rec_total_test.txt",
        "├── char_dictionary.txt",
        "└── dataset_report.txt",
        "",
        "=" * 60,
    ]

    report_path = OUTPUT_DIR / "dataset_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines) + '\n')

    print("\n" + '\n'.join(report_lines))


if __name__ == "__main__":
    main()

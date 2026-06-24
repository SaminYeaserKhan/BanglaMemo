"""
Quick visual test: compare mild vs strong crumple/smudge augmentations.
Generates a side-by-side comparison image.
"""
import random
import math
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path

random.seed(42)
np.random.seed(42)


def crumple_mild(img):
    """Current (too subtle) crumple."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)

    for _ in range(random.randint(2, 4)):
        amplitude = random.uniform(1.5, 4.0)
        freq_x = random.uniform(0.5, 3.0)
        freq_y = random.uniform(0.5, 3.0)
        phase_x = random.uniform(0, 2 * math.pi)
        phase_y = random.uniform(0, 2 * math.pi)
        angle = random.uniform(0, math.pi)
        wave_x = np.sin(2 * math.pi * freq_y * y_coords / h + phase_y) * amplitude
        wave_y = np.sin(2 * math.pi * freq_x * x_coords / w + phase_x) * amplitude
        dx += wave_x * math.cos(angle)
        dy += wave_y * math.sin(angle)

    return _apply_warp(arr, x_coords, y_coords, dx, dy, h, w)


def crumple_strong(img):
    """Stronger crumple — more waves, higher amplitude, added fold lines."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)

    # More waves (4-7) with higher amplitude (4-10 pixels)
    for _ in range(random.randint(4, 7)):
        amplitude = random.uniform(4.0, 10.0)
        freq_x = random.uniform(1.0, 5.0)
        freq_y = random.uniform(1.0, 5.0)
        phase_x = random.uniform(0, 2 * math.pi)
        phase_y = random.uniform(0, 2 * math.pi)
        angle = random.uniform(0, math.pi)
        wave_x = np.sin(2 * math.pi * freq_y * y_coords / h + phase_y) * amplitude
        wave_y = np.sin(2 * math.pi * freq_x * x_coords / w + phase_x) * amplitude
        dx += wave_x * math.cos(angle)
        dy += wave_y * math.sin(angle)

    # Add 1-2 sharp fold lines (localized high-displacement)
    n_folds = random.randint(1, 2)
    for _ in range(n_folds):
        fold_pos = random.uniform(0.2, 0.8)  # position along image
        fold_width = random.uniform(0.03, 0.08)  # width of fold region
        fold_amplitude = random.uniform(6, 15)  # strong local displacement
        is_horizontal = random.random() > 0.5

        if is_horizontal:
            fold_center = int(fold_pos * h)
            fold_w = int(fold_width * h)
            y_dist = np.abs(y_coords - fold_center).astype(np.float32)
            fold_mask = np.exp(-(y_dist ** 2) / (2 * (fold_w ** 2)))
            dx += fold_mask * fold_amplitude * np.sin(2 * math.pi * x_coords / w * 2)
        else:
            fold_center = int(fold_pos * w)
            fold_w = int(fold_width * w)
            x_dist = np.abs(x_coords - fold_center).astype(np.float32)
            fold_mask = np.exp(-(x_dist ** 2) / (2 * (fold_w ** 2)))
            dy += fold_mask * fold_amplitude * np.sin(2 * math.pi * y_coords / h * 2)

    return _apply_warp(arr, x_coords, y_coords, dx, dy, h, w)


def _apply_warp(arr, x_coords, y_coords, dx, dy, h, w):
    map_x = np.clip(x_coords + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y_coords + dy, 0, h - 1).astype(np.float32)
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    fx = (map_x - x0)[:, :, np.newaxis] if len(arr.shape) == 3 else (map_x - x0)
    fy = (map_y - y0)[:, :, np.newaxis] if len(arr.shape) == 3 else (map_y - y0)
    result = (arr[y0, x0] * (1 - fx) * (1 - fy) +
              arr[y0, x1] * fx * (1 - fy) +
              arr[y1, x0] * (1 - fx) * fy +
              arr[y1, x1] * fx * fy)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def convolve2d(image_2d, kern):
    h, w = image_2d.shape
    kh, kw = kern.shape
    padded = np.pad(image_2d, ((kh//2, kh//2), (kw//2, kw//2)), mode='edge')
    result = np.zeros_like(image_2d)
    for ky in range(kh):
        for kx in range(kw):
            result += padded[ky:ky+h, kx:kx+w] * kern[ky, kx]
    return result


def smudge_mild(img):
    """Current (barely visible) smudge."""
    arr = np.array(img, dtype=np.float32)
    gray = np.mean(arr, axis=2) if len(arr.shape) == 3 else arr
    ink_mask = (gray < 140).astype(np.float32)
    mask_img = Image.fromarray((ink_mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))
    ink_mask = np.array(mask_img, dtype=np.float32) / 255.0

    kernel_size = 5
    kernel = _make_motion_kernel(kernel_size, random.uniform(0, 180))
    blurred_arr = _apply_blur(arr, kernel, kernel_size)

    if len(arr.shape) == 3:
        ink_mask = ink_mask[:, :, np.newaxis]
    strength = 0.6
    result = arr * (1 - ink_mask * strength) + blurred_arr * (ink_mask * strength)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def smudge_strong(img):
    """Stronger smudge — larger kernel, higher strength, ink spread effect."""
    arr = np.array(img, dtype=np.float32)
    gray = np.mean(arr, axis=2) if len(arr.shape) == 3 else arr

    # More aggressive ink detection
    ink_mask = (gray < 160).astype(np.float32)
    mask_img = Image.fromarray((ink_mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))
    ink_mask = np.array(mask_img, dtype=np.float32) / 255.0

    # Larger motion blur kernel (9-15 pixels)
    kernel_size = random.choice([9, 11, 13])
    angle = random.uniform(0, 180)
    kernel = _make_motion_kernel(kernel_size, angle)
    blurred_arr = _apply_blur(arr, kernel, kernel_size)

    if len(arr.shape) == 3:
        ink_mask_3d = ink_mask[:, :, np.newaxis]
    else:
        ink_mask_3d = ink_mask

    # Higher blend strength
    strength = random.uniform(0.7, 0.95)
    result = arr * (1 - ink_mask_3d * strength) + blurred_arr * (ink_mask_3d * strength)

    # Add ink bleeding/spreading: dilate dark regions slightly
    result_img = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    # Apply a slight darkening spread on ink regions
    dark_spread = result_img.filter(ImageFilter.MinFilter(size=3))
    dark_arr = np.array(dark_spread, dtype=np.float32)
    # Blend the spread effect only where there's ink
    spread_strength = random.uniform(0.2, 0.4)
    result = result * (1 - ink_mask_3d * spread_strength) + dark_arr * (ink_mask_3d * spread_strength)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _make_motion_kernel(kernel_size, angle):
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
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return kernel


def _apply_blur(arr, kernel, kernel_size):
    blurred = np.zeros_like(arr)
    if len(arr.shape) == 3:
        for c in range(arr.shape[2]):
            blurred[:, :, c] = convolve2d(arr[:, :, c], kernel)
    else:
        blurred = convolve2d(arr, kernel)
    return blurred


def main():
    # Load a sample crop
    src = Path("/Users/test/ReceiptProject/Dataset_PaddleOCR/train/items_crops/receipt_001_items.jpg")
    img = Image.open(src)

    out_dir = Path("/Users/test/ReceiptProject/augment_comparison")
    out_dir.mkdir(exist_ok=True)

    # Generate comparisons
    random.seed(42); np.random.seed(42)
    crumple_m = crumple_mild(img)
    random.seed(42); np.random.seed(42)
    crumple_s = crumple_strong(img)

    random.seed(99); np.random.seed(99)
    smudge_m = smudge_mild(img)
    random.seed(99); np.random.seed(99)
    smudge_s = smudge_strong(img)

    # Also do combined strong
    random.seed(77); np.random.seed(77)
    combined = crumple_strong(img)
    random.seed(88); np.random.seed(88)
    combined = smudge_strong(combined)

    # Save individual files
    img.save(out_dir / "0_original.jpg")
    crumple_m.save(out_dir / "1_crumple_mild.jpg")
    crumple_s.save(out_dir / "2_crumple_strong.jpg")
    smudge_m.save(out_dir / "3_smudge_mild.jpg")
    smudge_s.save(out_dir / "4_smudge_strong.jpg")
    combined.save(out_dir / "5_combined_strong.jpg")

    print("✅ Saved comparison images to:", out_dir)
    print("Files:")
    for f in sorted(out_dir.glob("*.jpg")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

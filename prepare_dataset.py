#!/usr/bin/env python3
"""
Bangla Receipt Dataset Preparation Script
==========================================
Reads merged_labels.csv, cleans corrupted transcriptions, and produces:
1. Cropped images: items region (first Item_Name → last Item_Price)
2. Cropped images: total amount region (separate)
3. Cleaned merged CSV with fixed transcriptions
4. Bangla character dictionary
5. Dataset summary report

Supervisor instructions:
- Crop from first Item_Name to last Item_Price
- Crop Total_Amount separately
- Unlabeled images are intentionally discarded (illegible/printed)
"""

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    print("Pillow is required. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageOps


# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
RECEIPTS_DIR = BASE_DIR / "Receipts_Processed"
MERGED_CSV = BASE_DIR / "merged_labels.csv"

# Output directories
OUTPUT_DIR = BASE_DIR / "Dataset_Final"
ITEMS_CROP_DIR = OUTPUT_DIR / "items_crops"
TOTAL_CROP_DIR = OUTPUT_DIR / "total_crops"

# Output files
CLEANED_CSV = OUTPUT_DIR / "cleaned_labels.csv"
CHAR_DICT_FILE = OUTPUT_DIR / "char_dictionary.txt"
REPORT_FILE = OUTPUT_DIR / "dataset_report.txt"

# Padding (percentage of crop size to add around edges)
PADDING_PERCENT = 2.0

# Item-related label types (defines the "items region")
ITEM_LABELS = {"Item_Name", "Quantity", "Item_Price"}


# ─── Helper Functions ────────────────────────────────────────────────────────

def extract_receipt_name(image_path: str) -> str:
    """Extract receipt filename (e.g., 'receipt_001') from Label Studio path."""
    match = re.search(r'(receipt_\d+)', image_path)
    return match.group(1) if match else None


def clean_transcription(transcription_list: list) -> list:
    """Remove corrupted dict entries from transcription arrays.
    
    Some Label Studio exports contain leftover JSON dict objects like:
    {"x": 27.23, "y": 48.07, "width": 17.39, ...}
    mixed in with the actual text strings. We filter those out.
    """
    cleaned = []
    for item in transcription_list:
        if isinstance(item, str):
            cleaned.append(item)
        # Skip dict/list entries (corrupted data)
    return cleaned


def percent_to_pixels(x_pct, y_pct, w_pct, h_pct, img_width, img_height):
    """Convert Label Studio percentage coordinates to pixel coordinates."""
    x = int(x_pct / 100.0 * img_width)
    y = int(y_pct / 100.0 * img_height)
    w = int(w_pct / 100.0 * img_width)
    h = int(h_pct / 100.0 * img_height)
    return x, y, w, h


def get_bounding_box_union(bboxes: list, img_width: int, img_height: int, 
                           padding_pct: float = PADDING_PERCENT) -> tuple:
    """
    Compute the union bounding box of multiple Label Studio bboxes.
    Returns (left, top, right, bottom) in pixel coordinates with padding.
    Handles rotated bounding boxes by computing their axis-aligned envelope.
    """
    import math
    
    if not bboxes:
        return None

    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for bbox in bboxes:
        x_pct = bbox['x']
        y_pct = bbox['y']
        w_pct = bbox['width']
        h_pct = bbox['height']
        rotation = bbox.get('rotation', 0)

        # Convert to pixel coordinates
        cx_px = (x_pct + w_pct / 2) / 100.0 * img_width
        cy_px = (y_pct + h_pct / 2) / 100.0 * img_height
        half_w = (w_pct / 100.0 * img_width) / 2
        half_h = (h_pct / 100.0 * img_height) / 2

        if abs(rotation) > 0.01:
            # Compute 4 corners of rotated rectangle
            angle_rad = math.radians(-rotation)  # Label Studio uses clockwise
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            corners = [
                (-half_w, -half_h),
                (half_w, -half_h),
                (half_w, half_h),
                (-half_w, half_h),
            ]
            for dx, dy in corners:
                rx = cx_px + dx * cos_a - dy * sin_a
                ry = cy_px + dx * sin_a + dy * cos_a
                min_x = min(min_x, rx)
                min_y = min(min_y, ry)
                max_x = max(max_x, rx)
                max_y = max(max_y, ry)
        else:
            x_px = x_pct / 100.0 * img_width
            y_px = y_pct / 100.0 * img_height
            w_px = w_pct / 100.0 * img_width
            h_px = h_pct / 100.0 * img_height
            min_x = min(min_x, x_px)
            min_y = min(min_y, y_px)
            max_x = max(max_x, x_px + w_px)
            max_y = max(max_y, y_px + h_px)

    # Add padding
    pad_x = (max_x - min_x) * padding_pct / 100.0
    pad_y = (max_y - min_y) * padding_pct / 100.0

    left = max(0, int(min_x - pad_x))
    top = max(0, int(min_y - pad_y))
    right = min(img_width, int(max_x + pad_x))
    bottom = min(img_height, int(max_y + pad_y))

    return left, top, right, bottom


def filter_outlier_item_bboxes(item_bboxes: list, item_trans: list,
                                store_bboxes: list, store_trans: list) -> tuple:
    """
    Detect Item_Name bboxes that are likely mislabeled store names.
    
    Heuristic: If an item bbox is in the top 20% of the image AND has
    a y-gap > 20% to the nearest other item bbox, it's probably a store
    name that was mislabeled as Item_Name. Reclassify it to store.
    
    Returns: (filtered_item_bboxes, filtered_item_trans,
              updated_store_bboxes, updated_store_trans, num_reclassified)
    """
    if len(item_bboxes) < 2:
        return item_bboxes, item_trans, store_bboxes, store_trans, 0

    # Pair each bbox with its index so we can track transcription alignment
    indexed = [(i, bb) for i, bb in enumerate(item_bboxes)]
    # Sort by y-position (top → bottom)
    indexed.sort(key=lambda x: x[1]['y'])

    reclassified_indices = set()

    # Check the first (topmost) bbox for isolation
    first_idx, first_bb = indexed[0]
    second_bb = indexed[1][1]
    gap = second_bb['y'] - (first_bb['y'] + first_bb['height'])

    if first_bb['y'] < 20.0 and gap > 15.0:
        reclassified_indices.add(first_idx)

    if not reclassified_indices:
        return item_bboxes, item_trans, store_bboxes, store_trans, 0

    # Rebuild filtered lists
    new_item_bboxes = []
    new_item_trans = []
    trans_idx = 0
    for i, bb in enumerate(item_bboxes):
        rl = bb.get('rectanglelabels', bb.get('labels', ['']))
        if i in reclassified_indices:
            store_bboxes.append(bb)
            if trans_idx < len(item_trans):
                store_trans.append(item_trans[trans_idx])
                trans_idx += 1
        else:
            new_item_bboxes.append(bb)
            if trans_idx < len(item_trans):
                new_item_trans.append(item_trans[trans_idx])
                trans_idx += 1

    return new_item_bboxes, new_item_trans, store_bboxes, store_trans, len(reclassified_indices)


def build_char_dictionary(all_transcriptions: list) -> list:
    """Build a sorted list of unique characters from all transcriptions."""
    chars = set()
    for text in all_transcriptions:
        if isinstance(text, str):
            for ch in text:
                chars.add(ch)
    # Sort: Bangla chars first, then digits, then Latin, then symbols
    def sort_key(c):
        code = ord(c)
        if 0x0980 <= code <= 0x09FF:  # Bangla
            return (0, code)
        elif 0x09E6 <= code <= 0x09EF:  # Bangla digits
            return (1, code)
        elif c.isdigit():  # ASCII digits
            return (2, code)
        elif c.isalpha():  # Latin
            return (3, code)
        else:  # Symbols, spaces
            return (4, code)
    return sorted(chars, key=sort_key)


# ─── Main Processing ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Bangla Receipt Dataset Preparation")
    print("=" * 60)

    # Create output directories
    ITEMS_CROP_DIR.mkdir(parents=True, exist_ok=True)
    TOTAL_CROP_DIR.mkdir(parents=True, exist_ok=True)

    # Read merged CSV
    with open(MERGED_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\nLoaded {len(rows)} labeled receipts from merged_labels.csv")

    # ─── Processing ──────────────────────────────────────────────────────

    stats = {
        'total': len(rows),
        'items_cropped': 0,
        'total_cropped': 0,
        'no_items': 0,
        'no_total': 0,
        'image_missing': 0,
        'corrupted_fixed': 0,
        'crop_errors': 0,
        'outliers_reclassified': 0,
    }

    all_transcriptions = []
    cleaned_rows = []
    missing_images = []
    no_items_receipts = []
    no_total_receipts = []
    items_crop_labels = []   # For items crop label file
    total_crop_labels = []   # For total crop label file

    for row_idx, row in enumerate(rows):
        receipt_name = extract_receipt_name(row['image'])
        if not receipt_name:
            print(f"  [WARN] Could not parse receipt name from: {row['image']}")
            continue

        image_path = RECEIPTS_DIR / f"{receipt_name}.jpg"
        if not image_path.exists():
            # Some receipts use .jpeg extension
            image_path = RECEIPTS_DIR / f"{receipt_name}.jpeg"
            if not image_path.exists():
                stats['image_missing'] += 1
                missing_images.append(receipt_name)
                continue

        # Parse label and transcription
        try:
            labels = json.loads(row['label'])
            raw_trans = json.loads(row['transcription'])
        except json.JSONDecodeError as e:
            print(f"  [ERROR] JSON parse error for {receipt_name}: {e}")
            continue

        # Clean corrupted transcriptions
        cleaned_trans = clean_transcription(raw_trans)
        if len(cleaned_trans) < len(raw_trans):
            stats['corrupted_fixed'] += 1

        all_transcriptions.extend(cleaned_trans)

        # Open image and apply EXIF rotation
        # Label Studio annotates images in their display orientation (after EXIF rotation)
        # so we must apply the same rotation before cropping
        try:
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            img_width, img_height = img.size
        except Exception as e:
            print(f"  [ERROR] Could not open {receipt_name}: {e}")
            stats['crop_errors'] += 1
            continue

        # Use original_width/height from labels for coordinate conversion
        # (these match what Label Studio saw when the annotations were made)
        label_width = labels[0].get('original_width', img_width)
        label_height = labels[0].get('original_height', img_height)

        # ─── Separate bboxes by category ─────────────────────────────

        item_bboxes = []     # Item_Name, Quantity, Item_Price
        total_bboxes = []    # Total_Amount
        store_bboxes = []    # Store_Name

        # Also collect transcriptions per category
        item_trans = []
        total_trans = []
        store_trans = []

        # Build a clean mapping: pair each label with its corresponding
        # transcription from the *original* (uncleaned) list, then skip
        # any non-string entries to stay in sync.
        for lbl_idx, lbl in enumerate(labels):
            rect_labels = lbl.get('rectanglelabels', lbl.get('labels', []))
            category = rect_labels[0] if rect_labels else 'Unknown'

            # Get the corresponding transcription from the raw list
            trans_text = None
            if lbl_idx < len(raw_trans) and isinstance(raw_trans[lbl_idx], str):
                trans_text = raw_trans[lbl_idx]

            if category in ITEM_LABELS:
                item_bboxes.append(lbl)
                if trans_text is not None:
                    item_trans.append(trans_text)
            elif category == 'Total_Amount':
                total_bboxes.append(lbl)
                if trans_text is not None:
                    total_trans.append(trans_text)
            elif category == 'Store_Name':
                store_bboxes.append(lbl)
                if trans_text is not None:
                    store_trans.append(trans_text)

        # ─── Filter mislabeled store names from items ─────────────

        item_bboxes, item_trans, store_bboxes, store_trans, n_reclass = \
            filter_outlier_item_bboxes(item_bboxes, item_trans,
                                       store_bboxes, store_trans)
        if n_reclass > 0:
            stats['outliers_reclassified'] += n_reclass

        # ─── Crop items region ───────────────────────────────────────

        if item_bboxes:
            union_bbox = get_bounding_box_union(item_bboxes, label_width, label_height)
            if union_bbox:
                left, top, right, bottom = union_bbox
                # Scale to actual image size if different from label dimensions
                sx = img_width / label_width
                sy = img_height / label_height
                left_px = max(0, int(left * sx))
                top_px = max(0, int(top * sy))
                right_px = min(img_width, int(right * sx))
                bottom_px = min(img_height, int(bottom * sy))
                # Validate crop dimensions
                if right_px > left_px and bottom_px > top_px:
                    try:
                        items_crop = img.crop((left_px, top_px, right_px, bottom_px))
                        crop_filename = f"{receipt_name}_items.jpg"
                        items_crop.save(ITEMS_CROP_DIR / crop_filename, quality=95)
                        stats['items_cropped'] += 1

                        # Record label info for this crop
                        items_crop_labels.append({
                            'receipt': receipt_name,
                            'crop_file': crop_filename,
                            'crop_bbox': [left_px, top_px, right_px, bottom_px],
                            'transcriptions': item_trans,
                            'num_items': len([b for b in item_bboxes 
                                            if b.get('rectanglelabels', b.get('labels', ['']))[0] == 'Item_Name']),
                        })
                    except Exception as e:
                        print(f"  [ERROR] Crop failed for {receipt_name} items: {e}")
                        stats['crop_errors'] += 1
        else:
            stats['no_items'] += 1
            no_items_receipts.append(receipt_name)

        # ─── Crop total amount ───────────────────────────────────────

        if total_bboxes:
            union_bbox = get_bounding_box_union(total_bboxes, label_width, label_height,
                                                  padding_pct=5.0)  # More padding for total
            if union_bbox:
                left, top, right, bottom = union_bbox
                sx = img_width / label_width
                sy = img_height / label_height
                left_px = max(0, int(left * sx))
                top_px = max(0, int(top * sy))
                right_px = min(img_width, int(right * sx))
                bottom_px = min(img_height, int(bottom * sy))
                if right_px > left_px and bottom_px > top_px:
                    try:
                        total_crop = img.crop((left_px, top_px, right_px, bottom_px))
                        crop_filename = f"{receipt_name}_total.jpg"
                        total_crop.save(TOTAL_CROP_DIR / crop_filename, quality=95)
                        stats['total_cropped'] += 1

                        total_crop_labels.append({
                            'receipt': receipt_name,
                            'crop_file': crop_filename,
                            'crop_bbox': [left_px, top_px, right_px, bottom_px],
                            'transcription': total_trans[0] if total_trans else '',
                        })
                    except Exception as e:
                        print(f"  [ERROR] Crop failed for {receipt_name} total: {e}")
                        stats['crop_errors'] += 1
        else:
            stats['no_total'] += 1
            no_total_receipts.append(receipt_name)

        img.close()

        # Build cleaned row for output CSV
        cleaned_row = dict(row)
        cleaned_row['transcription'] = json.dumps(cleaned_trans, ensure_ascii=False)
        cleaned_rows.append(cleaned_row)

        # Progress
        if (row_idx + 1) % 100 == 0:
            print(f"  Processed {row_idx + 1}/{len(rows)} receipts...")

    # ─── Write cleaned CSV ───────────────────────────────────────────────

    if cleaned_rows:
        fieldnames = cleaned_rows[0].keys()
        with open(CLEANED_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)
        print(f"\n✅ Cleaned CSV written: {CLEANED_CSV}")

    # ─── Write items crop labels ─────────────────────────────────────────

    items_label_file = OUTPUT_DIR / "items_crop_labels.json"
    with open(items_label_file, 'w', encoding='utf-8') as f:
        json.dump(items_crop_labels, f, ensure_ascii=False, indent=2)
    print(f"✅ Items crop labels written: {items_label_file}")

    # ─── Write total crop labels ─────────────────────────────────────────

    total_label_file = OUTPUT_DIR / "total_crop_labels.json"
    with open(total_label_file, 'w', encoding='utf-8') as f:
        json.dump(total_crop_labels, f, ensure_ascii=False, indent=2)
    print(f"✅ Total crop labels written: {total_label_file}")

    # ─── Build character dictionary ──────────────────────────────────────

    char_list = build_char_dictionary(all_transcriptions)
    with open(CHAR_DICT_FILE, 'w', encoding='utf-8') as f:
        for ch in char_list:
            f.write(ch + '\n')
    print(f"✅ Character dictionary written: {CHAR_DICT_FILE} ({len(char_list)} unique chars)")

    # ─── Write PaddleOCR-format recognition labels ───────────────────────

    # Items crops: each crop path + all item text joined
    rec_items_file = OUTPUT_DIR / "rec_items_labels.txt"
    with open(rec_items_file, 'w', encoding='utf-8') as f:
        for entry in items_crop_labels:
            crop_path = f"items_crops/{entry['crop_file']}"
            # Join all transcriptions with space separator
            text = ' '.join(entry['transcriptions'])
            f.write(f"{crop_path}\t{text}\n")
    print(f"✅ Recognition labels (items) written: {rec_items_file}")

    # Total crops: each crop path + total text
    rec_total_file = OUTPUT_DIR / "rec_total_labels.txt"
    with open(rec_total_file, 'w', encoding='utf-8') as f:
        for entry in total_crop_labels:
            crop_path = f"total_crops/{entry['crop_file']}"
            text = entry['transcription']
            f.write(f"{crop_path}\t{text}\n")
    print(f"✅ Recognition labels (total) written: {rec_total_file}")

    # ─── Generate Report ─────────────────────────────────────────────────

    report_lines = [
        "=" * 60,
        "BANGLA RECEIPT DATASET - PREPARATION REPORT",
        "=" * 60,
        "",
        "SUMMARY",
        "-" * 40,
        f"Total labeled receipts:          {stats['total']}",
        f"Images found and processed:      {stats['total'] - stats['image_missing']}",
        f"Images not found:                {stats['image_missing']}",
        f"Corrupted transcriptions fixed:  {stats['corrupted_fixed']}",
        f"Mislabeled store names fixed:   {stats['outliers_reclassified']}",
        "",
        "CROPS GENERATED",
        "-" * 40,
        f"Items region crops:              {stats['items_cropped']}",
        f"Total amount crops:              {stats['total_cropped']}",
        f"Receipts with NO items:          {stats['no_items']}",
        f"Receipts with NO total:          {stats['no_total']}",
        f"Crop errors:                     {stats['crop_errors']}",
        "",
        "CHARACTER DICTIONARY",
        "-" * 40,
        f"Unique characters:               {len(char_list)}",
        f"  Bangla chars:                  {sum(1 for c in char_list if 0x0980 <= ord(c) <= 0x09FF)}",
        f"  ASCII digits:                  {sum(1 for c in char_list if c.isascii() and c.isdigit())}",
        f"  Latin chars:                   {sum(1 for c in char_list if c.isascii() and c.isalpha())}",
        f"  Symbols/spaces:                {sum(1 for c in char_list if not c.isalnum())}",
        "",
        "OUTPUT FILES",
        "-" * 40,
        f"  {CLEANED_CSV.relative_to(BASE_DIR)}",
        f"  {items_label_file.relative_to(BASE_DIR)}",
        f"  {total_label_file.relative_to(BASE_DIR)}",
        f"  {CHAR_DICT_FILE.relative_to(BASE_DIR)}",
        f"  {rec_items_file.relative_to(BASE_DIR)}",
        f"  {rec_total_file.relative_to(BASE_DIR)}",
        f"  {ITEMS_CROP_DIR.relative_to(BASE_DIR)}/  ({stats['items_cropped']} images)",
        f"  {TOTAL_CROP_DIR.relative_to(BASE_DIR)}/  ({stats['total_cropped']} images)",
        "",
    ]

    if no_total_receipts:
        report_lines.extend([
            "RECEIPTS MISSING TOTAL_AMOUNT LABEL",
            "-" * 40,
        ])
        for r in sorted(no_total_receipts):
            report_lines.append(f"  {r}")
        report_lines.append("")

    if missing_images:
        report_lines.extend([
            "LABELED RECEIPTS WITH MISSING IMAGE FILE",
            "-" * 40,
        ])
        for r in sorted(missing_images):
            report_lines.append(f"  {r}")
        report_lines.append("")

    # Note about intentionally excluded images
    report_lines.extend([
        "NOTES",
        "-" * 40,
        f"76 images were intentionally not labeled:",
        f"  - Images that were not readable/understandable",
        f"  - Images at the end of the dataset that were printed (not handwritten)",
        f"These are excluded from the final dataset by design.",
        "",
        "=" * 60,
    ])

    report_text = '\n'.join(report_lines)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✅ Report written: {REPORT_FILE}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()

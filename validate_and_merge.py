#!/usr/bin/env python3
"""
Validate Label Studio CSV exports for Bangla receipt OCR dataset
and merge them into a single CSV file.

Handles the column mismatch in '501 - 600.csv' which uses a different
Label Studio export template (bbox/ocr columns instead of image).
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict

CSV_DIR = "Labeled_CSV"
IMAGE_DIR = "Receipts_Processed"
OUTPUT_FILE = "merged_labels.csv"

# Standard columns from most CSV files
STANDARD_COLUMNS = [
    "annotation_id", "annotator", "created_at", "id", "image",
    "label", "lead_time", "transcription", "updated_at"
]

# Alternate columns from '501 - 600.csv'
# Has: annotation_id, annotator, bbox, created_at, id, label, lead_time, ocr, transcription, updated_at
# 'ocr' column contains the image path, 'bbox' contains unlabeled bbox data

VALID_LABELS = {"Store_Name", "Item_Name", "Quantity", "Item_Price", "Total_Amount"}

def extract_receipt_number(image_path):
    """Extract the receipt number from the image path."""
    if not image_path:
        return None
    match = re.search(r'receipt_(\d+)', str(image_path))
    if match:
        return int(match.group(1))
    return None

def normalize_row(row, fieldnames):
    """
    Normalize a row to the standard column format.
    Handles the alternate column layout from '501 - 600.csv'.
    """
    if 'image' in fieldnames:
        # Standard format
        return row
    
    # Alternate format: 'ocr' contains image path, 'bbox' is extra
    normalized = {}
    for col in STANDARD_COLUMNS:
        if col == 'image':
            # Image path is in the 'ocr' column
            normalized['image'] = row.get('ocr', '')
        else:
            normalized[col] = row.get(col, '')
    
    # Fix label format: alternate uses 'labels' key instead of 'rectanglelabels'
    label_str = normalized.get('label', '')
    if label_str:
        try:
            labels = json.loads(label_str)
            if isinstance(labels, list):
                fixed = False
                for lbl in labels:
                    if 'labels' in lbl and 'rectanglelabels' not in lbl:
                        lbl['rectanglelabels'] = lbl.pop('labels')
                        fixed = True
                if fixed:
                    normalized['label'] = json.dumps(labels)
        except json.JSONDecodeError:
            pass
    
    return normalized

def validate_row(row, row_idx, filename):
    """Validate a single row and return issues found."""
    issues = []
    receipt_num = extract_receipt_number(row.get('image', ''))
    row_id = f"Row {row_idx} (receipt_{receipt_num:03d})" if receipt_num else f"Row {row_idx}"
    
    # 1. Check if image reference is valid
    image_path = row.get('image', '')
    if not image_path:
        issues.append(f"  ❌ {row_id}: Empty image path")
    
    # 2. Check label JSON
    label_str = row.get('label', '')
    if not label_str:
        issues.append(f"  ❌ {row_id}: Empty label")
    else:
        try:
            labels = json.loads(label_str)
            if not isinstance(labels, list):
                issues.append(f"  ❌ {row_id}: Label is not a JSON array")
            else:
                label_types = defaultdict(int)
                for lbl in labels:
                    rect_labels = lbl.get('rectanglelabels', [])
                    for rl in rect_labels:
                        label_types[rl] += 1
                        if rl not in VALID_LABELS:
                            issues.append(f"  ❌ {row_id}: Invalid label type '{rl}'")
                    
                    # Check for required bbox fields
                    for field in ['x', 'y', 'width', 'height']:
                        if field not in lbl:
                            issues.append(f"  ❌ {row_id}: Missing '{field}' in bounding box")
                        elif not isinstance(lbl[field], (int, float)):
                            issues.append(f"  ⚠️  {row_id}: '{field}' is not numeric: {lbl[field]}")
                    
                    # Check for negative or out-of-bounds coords
                    x = lbl.get('x', 0)
                    y = lbl.get('y', 0)
                    w = lbl.get('width', 0)
                    h = lbl.get('height', 0)
                    if x < -1 or y < -1:
                        issues.append(f"  ⚠️  {row_id}: Negative coordinate x={x:.2f}, y={y:.2f}")
                    if w <= 0 or h <= 0:
                        issues.append(f"  ❌ {row_id}: Zero/negative dimension w={w:.2f}, h={h:.2f}")
                    if x + w > 105:
                        issues.append(f"  ⚠️  {row_id}: Bbox extends beyond image width (x+w={x+w:.2f})")
                    if y + h > 105:
                        issues.append(f"  ⚠️  {row_id}: Bbox extends beyond image height (y+h={y+h:.2f})")
                
                # Check that required label types are present
                if 'Total_Amount' not in label_types:
                    issues.append(f"  ⚠️  {row_id}: Missing 'Total_Amount' label")
                if 'Item_Name' not in label_types and 'Store_Name' not in label_types:
                    issues.append(f"  ⚠️  {row_id}: Missing both 'Item_Name' and 'Store_Name' labels")
        
        except json.JSONDecodeError as e:
            issues.append(f"  ❌ {row_id}: Invalid label JSON - {e}")
    
    # 3. Check transcription JSON
    transcription_str = row.get('transcription', '')
    if not transcription_str:
        issues.append(f"  ❌ {row_id}: Empty transcription")
    else:
        try:
            transcriptions = json.loads(transcription_str)
            if not isinstance(transcriptions, list):
                issues.append(f"  ❌ {row_id}: Transcription is not a JSON array")
            else:
                dict_items = []
                for idx, t in enumerate(transcriptions):
                    if isinstance(t, dict):
                        dict_items.append(idx)
                    elif not isinstance(t, str):
                        issues.append(f"  ⚠️  {row_id}: Transcription item {idx} is type {type(t).__name__}")
                
                if dict_items:
                    issues.append(f"  ⚠️  {row_id}: {len(dict_items)} transcription items are dicts (leftover bbox data) at indices {dict_items}")
                
                # Count transcription items vs label bboxes
                try:
                    labels = json.loads(label_str)
                    num_bboxes = len(labels) if isinstance(labels, list) else 0
                    num_str_trans = sum(1 for t in transcriptions if isinstance(t, str))
                    if num_str_trans != num_bboxes:
                        issues.append(f"  ⚠️  {row_id}: Mismatch - {num_bboxes} bboxes vs {num_str_trans} text transcriptions")
                except:
                    pass
        
        except json.JSONDecodeError as e:
            issues.append(f"  ❌ {row_id}: Invalid transcription JSON - {e}")
    
    # 4. Check lead_time
    lead_time = row.get('lead_time', '')
    if lead_time:
        try:
            lt = float(lead_time)
            if lt > 10000:
                issues.append(f"  ⚠️  {row_id}: Very high lead_time ({lt:.1f}s = {lt/60:.1f}min)")
        except ValueError:
            issues.append(f"  ⚠️  {row_id}: Invalid lead_time: {lead_time}")
    
    return issues

def validate_csv_file(filepath):
    """Validate a single CSV file and return normalized rows + issues."""
    issues = []
    rows = []
    filename = os.path.basename(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        
        # Check columns
        has_image = 'image' in fieldnames
        has_ocr = 'ocr' in fieldnames
        
        if not has_image and not has_ocr:
            issues.append(f"  ❌ Missing both 'image' and 'ocr' columns - cannot determine image paths")
        elif not has_image and has_ocr:
            issues.append(f"  ⚠️  Non-standard column layout: uses 'ocr' for image path and has extra 'bbox' column")
            issues.append(f"       Will be normalized during merge.")
        
        for row_idx, row in enumerate(reader, start=2):
            normalized = normalize_row(row, fieldnames)
            row_issues = validate_row(normalized, row_idx, filename)
            issues.extend(row_issues)
            rows.append(normalized)
    
    return rows, issues, fieldnames

def check_image_coverage(all_receipt_numbers):
    """Check which receipt images are covered by the labels."""
    image_files = set()
    for f in os.listdir(IMAGE_DIR):
        if f.startswith('receipt_') and (f.endswith('.jpg') or f.endswith('.jpeg')):
            num = extract_receipt_number(f)
            if num:
                image_files.add(num)
    
    labeled = set(all_receipt_numbers)
    unlabeled = image_files - labeled
    no_image = labeled - image_files
    
    return image_files, labeled, unlabeled, no_image

def main():
    print("=" * 80)
    print("BANGLA RECEIPT OCR DATASET - LABEL VALIDATION REPORT")
    print("=" * 80)
    
    csv_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith('.csv')])
    
    all_rows = []
    all_issues = defaultdict(list)
    all_receipt_numbers = []
    duplicate_receipts = defaultdict(list)
    
    print(f"\n📁 Found {len(csv_files)} CSV files:\n")
    
    for csv_file in csv_files:
        filepath = os.path.join(CSV_DIR, csv_file)
        file_size = os.path.getsize(filepath) / 1024
        
        rows, issues, fieldnames = validate_csv_file(filepath)
        
        file_receipts = []
        for row in rows:
            receipt_num = extract_receipt_number(row.get('image', ''))
            if receipt_num:
                file_receipts.append(receipt_num)
                all_receipt_numbers.append(receipt_num)
                duplicate_receipts[receipt_num].append(csv_file)
        
        file_receipts_sorted = sorted(set(file_receipts))
        range_str = f"{min(file_receipts_sorted)}-{max(file_receipts_sorted)}" if file_receipts_sorted else "N/A"
        
        errors = sum(1 for i in issues if '❌' in i)
        warnings = sum(1 for i in issues if '⚠️' in i)
        
        print(f"  📄 {csv_file}")
        print(f"     Size: {file_size:.1f} KB | Rows: {len(rows)} | Receipt range: {range_str}")
        print(f"     Columns: {fieldnames}")
        
        if issues:
            print(f"     Issues: {errors} errors, {warnings} warnings")
            all_issues[csv_file] = issues
        else:
            print(f"     ✅ No issues found")
        
        all_rows.extend(rows)
    
    # Duplicate check
    print(f"\n{'=' * 80}")
    print("DUPLICATE CHECK")
    print("=" * 80)
    
    duplicates = {k: v for k, v in duplicate_receipts.items() if len(v) > 1}
    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} receipts labeled in multiple files:")
        for receipt_num in sorted(duplicates.keys()):
            files = duplicates[receipt_num]
            print(f"  receipt_{receipt_num:03d} → {', '.join(files)}")
    else:
        print("\n✅ No duplicate labels found")
    
    # Image coverage
    print(f"\n{'=' * 80}")
    print("IMAGE COVERAGE CHECK")
    print("=" * 80)
    
    image_files, labeled, unlabeled, no_image = check_image_coverage(all_receipt_numbers)
    
    print(f"\n  Total images in folder: {len(image_files)}")
    print(f"  Total labeled (unique): {len(labeled)}")
    print(f"  Coverage: {len(labeled)}/{len(image_files)} ({100*len(labeled)/len(image_files):.1f}%)")
    
    if unlabeled:
        # Group unlabeled by range for readability
        unlabeled_sorted = sorted(unlabeled)
        ranges = []
        start = unlabeled_sorted[0]
        end = start
        for num in unlabeled_sorted[1:]:
            if num == end + 1:
                end = num
            else:
                ranges.append((start, end))
                start = num
                end = num
        ranges.append((start, end))
        
        print(f"\n  ⚠️  {len(unlabeled)} images WITHOUT labels:")
        for s, e in ranges:
            if s == e:
                print(f"    receipt_{s:03d}")
            else:
                print(f"    receipt_{s:03d} to receipt_{e:03d} ({e - s + 1} images)")
    else:
        print(f"\n  ✅ All images have labels")
    
    if no_image:
        print(f"\n  ❌ {len(no_image)} labels WITHOUT matching images:")
        for num in sorted(no_image):
            print(f"    receipt_{num:03d}")
    
    # Detailed issues summary (condensed)
    print(f"\n{'=' * 80}")
    print("ISSUE SUMMARY BY FILE")
    print("=" * 80)
    
    total_errors = 0
    total_warnings = 0
    
    # Categorize issues
    issue_categories = {
        'schema': [],
        'missing_total': [],
        'dict_in_transcription': [],
        'bbox_transcription_mismatch': [],
        'high_lead_time': [],
        'other': []
    }
    
    for csv_file, issues in all_issues.items():
        errors = sum(1 for i in issues if '❌' in i)
        warnings = sum(1 for i in issues if '⚠️' in i)
        total_errors += errors
        total_warnings += warnings
        
        for issue in issues:
            if 'Non-standard column' in issue or 'normalized' in issue:
                issue_categories['schema'].append((csv_file, issue))
            elif "Missing 'Total_Amount'" in issue:
                # Extract receipt num
                m = re.search(r'receipt_(\d+)', issue)
                if m:
                    issue_categories['missing_total'].append(int(m.group(1)))
            elif 'dicts' in issue or 'leftover bbox' in issue:
                m = re.search(r'receipt_(\d+)', issue)
                if m:
                    issue_categories['dict_in_transcription'].append(int(m.group(1)))
            elif 'Mismatch' in issue:
                m = re.search(r'receipt_(\d+)', issue)
                if m:
                    issue_categories['bbox_transcription_mismatch'].append(int(m.group(1)))
            elif 'lead_time' in issue:
                m = re.search(r'receipt_(\d+)', issue)
                if m:
                    issue_categories['high_lead_time'].append(int(m.group(1)))
    
    # Print categorized
    if issue_categories['schema']:
        print(f"\n📋 SCHEMA ISSUES:")
        print(f"  '501 - 600.csv' uses non-standard columns (bbox/ocr instead of image)")
        print(f"  → Automatically normalized during merge")
    
    if issue_categories['missing_total']:
        receipts = sorted(set(issue_categories['missing_total']))
        print(f"\n📋 MISSING 'Total_Amount' LABEL ({len(receipts)} receipts):")
        print(f"  {', '.join(f'receipt_{n:03d}' for n in receipts)}")
    
    if issue_categories['dict_in_transcription']:
        receipts = sorted(set(issue_categories['dict_in_transcription']))
        print(f"\n📋 LEFTOVER DICT IN TRANSCRIPTION ({len(receipts)} receipts):")
        print(f"  These have bbox JSON objects mixed into the transcription text array")
        print(f"  {', '.join(f'receipt_{n:03d}' for n in receipts)}")
    
    if issue_categories['bbox_transcription_mismatch']:
        receipts = sorted(set(issue_categories['bbox_transcription_mismatch']))
        print(f"\n📋 BBOX vs TRANSCRIPTION COUNT MISMATCH ({len(receipts)} receipts):")
        print(f"  Number of bounding boxes doesn't match number of text transcriptions")
        print(f"  {', '.join(f'receipt_{n:03d}' for n in receipts)}")
    
    if issue_categories['high_lead_time']:
        receipts = sorted(set(issue_categories['high_lead_time']))
        print(f"\n📋 VERY HIGH LEAD TIME ({len(receipts)} receipts):")
        print(f"  Labeling took >10 min (likely left tab open, not an actual error)")
        print(f"  {', '.join(f'receipt_{n:03d}' for n in receipts)}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n  Total CSV files: {len(csv_files)}")
    print(f"  Total rows (annotations): {len(all_rows)}")
    print(f"  Unique receipts labeled: {len(labeled)}")
    print(f"  Duplicate entries: {len(duplicates)}")
    print(f"  Unlabeled images: {len(unlabeled)}")
    print(f"  Missing 'Total_Amount': {len(set(issue_categories['missing_total']))}")
    print(f"  Leftover dict in transcription: {len(set(issue_categories['dict_in_transcription']))}")
    print(f"  Bbox/transcription mismatch: {len(set(issue_categories['bbox_transcription_mismatch']))}")
    print(f"  High lead time (informational): {len(set(issue_categories['high_lead_time']))}")
    
    # Merge
    print(f"\n{'=' * 80}")
    print("MERGING CSV FILES")
    print("=" * 80)
    
    # Deduplicate: keep the latest annotation per receipt
    seen_receipts = {}
    for row in all_rows:
        receipt_num = extract_receipt_number(row.get('image', ''))
        if receipt_num:
            if receipt_num in seen_receipts:
                existing_time = seen_receipts[receipt_num].get('updated_at', '')
                new_time = row.get('updated_at', '')
                if new_time > existing_time:
                    seen_receipts[receipt_num] = row
            else:
                seen_receipts[receipt_num] = row
    
    sorted_receipts = sorted(seen_receipts.items(), key=lambda x: x[0])
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=STANDARD_COLUMNS)
        writer.writeheader()
        for receipt_num, row in sorted_receipts:
            # Only write standard columns
            clean_row = {k: row.get(k, '') for k in STANDARD_COLUMNS}
            writer.writerow(clean_row)
    
    merged_size = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\n  ✅ Merged {len(sorted_receipts)} unique annotations into '{OUTPUT_FILE}'")
    print(f"  Output file size: {merged_size:.1f} KB")
    print(f"  All rows normalized to standard column format")
    
    if duplicates:
        print(f"  ℹ️  {len(duplicates)} duplicates resolved (kept latest annotation)")
    
    print(f"\n{'=' * 80}")
    print("DONE")
    print("=" * 80)

if __name__ == '__main__':
    main()

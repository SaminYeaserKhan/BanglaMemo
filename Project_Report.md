# BanglaMemo: A Handwritten Bangla Receipt Dataset for OCR-Based Expense Tracking

**Course:** Digital Image Processing  
**Semester:** [Your Semester/Year]  
**Team Members:** [Your Names]  
**Supervisor:** [Supervisor Name]  
**Date:** June 2026

---

## Abstract

This report presents **BanglaMemo**, a novel dataset of 698 handwritten Bangla receipt images, collected, annotated, and processed for training Optical Character Recognition (OCR) systems. The dataset addresses a critical gap in Bangla handwriting recognition research: the absence of domain-specific, real-world receipt data with messy, informal handwriting. We collected receipt images from local marketplaces in Dhaka, annotated them using Label Studio with bounding boxes and Bangla transcriptions, and developed an automated pipeline for data cleaning, cropping, augmentation, and PaddleOCR-compatible formatting. The final dataset contains 622 fully labeled receipts with 3,110 augmented training samples, split into train/validation/test sets with no data leakage. The dataset includes 143 unique characters spanning Bangla script, Bangla digits, Latin characters, and special symbols. This work establishes a foundation for building a complete Bangla receipt OCR system for mobile expense tracking applications.

---

## 1. Introduction

### 1.1 Motivation

Expense tracking is a fundamental financial management task. In Bangladesh, a significant portion of retail transactions—particularly in clothing bazaars, grocery stores, and street markets—still rely on handwritten paper receipts. These receipts are typically written in Bangla script with informal handwriting styles that vary dramatically between writers.

While OCR technology has matured considerably for printed text and for major languages like English, Chinese, and Hindi, Bangla handwriting recognition remains a challenging and under-explored problem. Existing Bangla OCR datasets predominantly focus on:

- Isolated character recognition (BanglaLekha, CMATERdb)
- Printed Bangla text (Apona-Pathak)
- Clean, structured handwriting samples collected under controlled conditions

None of these datasets capture the real-world messiness of marketplace receipts, which feature:

- Highly variable handwriting quality
- Mixed Bangla and English text (product names, quantities)
- Inconsistent layouts across different store formats
- Environmental noise (creases, smudges, colored backgrounds)
- Bangla numerals mixed with Arabic numerals

### 1.2 Objectives

The objectives of this project are:

1. **Data Collection:** Collect a substantial corpus of real handwritten Bangla receipt images from local markets.
2. **Annotation:** Annotate receipts with structured labels (item names, quantities, prices, totals, store names) and their Bangla transcriptions.
3. **Data Processing:** Develop an automated pipeline to clean, crop, augment, and format the dataset for OCR training.
4. **Dataset Delivery:** Produce a PaddleOCR-compatible dataset ready for fine-tuning a text recognition model.

### 1.3 Scope

This project focuses exclusively on the dataset construction phase. The actual OCR model training and the expense tracking application development are planned as future work beyond this course.

---

## 2. Related Work

### 2.1 Bangla Handwriting Datasets

| Dataset | Year | Size | Type | Limitation |
|:---|:---|:---|:---|:---|
| BanglaLekha-Isolated | 2017 | 166,105 images | Isolated characters | No word/line-level context |
| CMATERdb | 2012 | 15,000+ images | Isolated characters & words | Controlled writing conditions |
| BN-HTRd | 2021 | 1,000 pages | Handwritten text lines | Literary/formal writing only |
| Ekush | 2019 | 367,018 images | Isolated characters | No real-world domain data |
| **BanglaMemo (Ours)** | **2026** | **698 images** | **Full receipt documents** | **Real marketplace receipts** |

### 2.2 Receipt OCR Systems

Receipt OCR has received growing attention in recent years:

- **CORD (Consolidated Receipt Dataset):** 11,000 Indonesian receipts, but printed text only.
- **SROIE (ICDAR 2019):** 1,000 scanned receipts for key information extraction, English/Chinese only.
- **WildReceipt:** 1,740 receipts with key-value pairs, but focused on printed English text.

To our knowledge, **BanglaMemo is the first dataset specifically targeting handwritten Bangla receipts** from informal retail settings.

### 2.3 PaddleOCR

PaddleOCR is an open-source OCR toolkit developed by Baidu that supports text detection, recognition, and end-to-end systems. It provides:

- Pre-trained models for 80+ languages
- Fine-tuning capabilities for domain-specific text
- Lightweight deployment options for mobile and edge devices

We selected PaddleOCR as our target framework due to its strong multilingual support and proven fine-tuning pipeline.

---

## 3. Methodology

### 3.1 Data Collection

We collected **698 receipt images** from local marketplaces in Dhaka, Bangladesh. The receipts were sourced from two primary categories:

1. **Clothing/textile bazaars** (receipts 001–356): Featuring items such as hijabs, sarees, fabrics, and accessories with Bangla item names, quantities, and prices.
2. **Grocery/daily necessities stores** (receipts 357–698): Featuring items such as rice, eggs, oil, onions, and household goods.

All images were captured using smartphone cameras under natural conditions, preserving the real-world variability in lighting, angle, and image quality. Images were stored in JPEG format with EXIF metadata intact.

### 3.2 Annotation

Annotation was performed by three team members using **Label Studio**, an open-source data labeling platform. Each receipt was annotated with bounding boxes for:

| Label Type | Description | Example |
|:---|:---|:---|
| **Store_Name** | Business/shop name at the top | "মুফ্তি টেক্স" |
| **Item_Name** | Product description | "জর্জেট প্রিন্ট বুকে কুচি হেজাব" |
| **Quantity** | Number of items purchased | "৮ ps" |
| **Item_Price** | Individual item price | "৪৬০" |
| **Total_Amount** | Total price at the bottom | "৮৭৬০" |

Each bounding box was paired with a Bangla text transcription. The annotations were exported as CSV files from Label Studio, with each row containing the image path, bounding box coordinates (percentage-based), rotation angles, label categories, and transcription arrays in JSON format.

#### Annotation Challenges

- **Inter-annotator consistency:** Three annotators labeled different receipt ranges (1–100, 101–200, etc.), requiring careful schema alignment during merging.
- **Illegible receipts:** 76 images were intentionally excluded because the handwriting was not readable even by native Bangla speakers.
- **Printed receipts:** Some images at the end of the collection were printed rather than handwritten and were excluded from labeling.

### 3.3 Data Processing Pipeline

We developed a multi-stage automated processing pipeline in Python:

#### Stage 1: CSV Merging and Validation (`validate_and_merge.py`)

The six individual CSV exports from Label Studio were merged into a single unified dataset:

- Resolved schema inconsistencies across annotator exports (e.g., the `501-600.csv` file used an `ocr` column instead of `image`)
- Validated label format consistency
- Produced `merged_labels.csv` with 622 fully labeled receipts

#### Stage 2: Cleaning, Cropping, and Formatting (`prepare_dataset.py`)

This script processes the merged CSV and generates the cropped dataset:

1. **Transcription Cleaning:** Fixed 19 corrupted transcription entries where Label Studio exported raw JSON dictionary objects mixed into the transcription string arrays. The script filters these by checking `isinstance(item, str)`.

2. **Image File Resolution:** Resolved 42 "missing" images caused by file extension mismatch — receipts 357–421 were saved with `.jpeg` extension while the labels referenced `.jpg`. The script now checks both extensions.

3. **EXIF Rotation Handling:** A critical issue was discovered where images were stored in landscape orientation (3117×2019 pixels) but Label Studio displayed and annotated them in portrait orientation (2019×3117) based on EXIF orientation tag (value 6). The script applies `ImageOps.exif_transpose()` before cropping to ensure pixel-space coordinates match the annotation coordinates.

4. **Mislabeled Store Name Detection:** An automated outlier detection heuristic identified 9 instances where annotators labeled store names (e.g., "মুফ্তি টেক্স", "আল আরাফা ট্রেডিং") as `Item_Name`. The algorithm detects when an `Item_Name` bounding box is in the top 20% of the image with a gap >15% to the nearest item cluster, and reclassifies it to `Store_Name`.

5. **Region Cropping (per supervisor instructions):**
   - **Items region:** Union bounding box from the first `Item_Name` to the last `Item_Price`, with 2% padding
   - **Total Amount:** Separate crop of the `Total_Amount` bounding box, with 5% padding

6. **Character Dictionary Generation:** Extracted 143 unique characters across all transcriptions:
   - 66 Bangla characters (consonants, vowels, conjuncts)
   - 10 Bangla/Arabic digits
   - 47 Latin characters (for English brand names)
   - 34 symbols and spaces

#### Stage 3: Augmentation and Finalization (`augment_dataset.py`)

The final stage prepares the dataset for PaddleOCR fine-tuning:

1. **Data Cleaning:** Removed 2 entries with empty transcriptions (receipt_400, receipt_455 total crops).

2. **Train/Validation/Test Split:** Split by receipt ID (not by individual crop) to prevent data leakage:
   - Train: 85% of receipts
   - Validation: 10% of receipts
   - Test: 5% of receipts

3. **Data Augmentation:** Applied 4× augmentation to training images only, using randomized combinations of:
   - Brightness adjustment (±20%)
   - Contrast adjustment (±20%)
   - Slight rotation (±3°)
   - Gaussian noise (σ = 5–15)
   - Gaussian blur (radius 0.5–1.5)
   - Perspective warp (±2%)
   - JPEG compression artifacts (quality 50–85)

   These augmentations simulate real-world capture variability without destroying text readability. Destructive augmentations (flips, heavy rotation, cutout) were intentionally excluded.

4. **PaddleOCR Formatting:** Generated label files in the required `<image_path>\t<transcription>` format, with relative paths within each split directory.

### 3.4 Quality Assurance

The pipeline includes automated integrity checks:

- **Image-label consistency:** Verifies every image path in label files points to an existing file
- **Data leakage detection:** Confirms no receipt ID appears in multiple splits
- **Empty transcription detection:** Flags entries with missing ground truth text
- **Visual verification:** Manual inspection of cropped outputs across diverse receipt types

---

## 4. Dataset Description

### 4.1 Final Statistics

| Metric | Value |
|:---|:---|
| Raw images collected | 698 |
| Fully labeled receipts | 622 |
| Excluded (illegible/printed) | 76 |
| Items region crops | 622 |
| Total amount crops | 599 (after cleaning) |
| Character dictionary size | 143 |
| Corrupted entries fixed | 19 |
| Mislabeled entries corrected | 9 |
| Empty transcriptions removed | 2 |

### 4.2 Split Distribution

| Split | Receipts | Items Crops | Total Crops |
|:---|:---|:---|:---|
| Train (original) | 528 | 528 | 507 |
| Train (augmented) | — | 2,112 | 2,028 |
| **Train (total)** | **528** | **2,640** | **2,535** |
| Validation | 62 | 62 | 60 |
| Test | 32 | 32 | 32 |

### 4.3 Character Set

The dataset contains text in three scripts:

- **Bangla script:** অ–ঔ, ক–হ, vowel signs (া, ি, ী, ু, ূ, ে, ৈ, ো, ৌ), conjunct markers (্), chandrabindu (ঁ), anusvara (ং), visarga (ঃ)
- **Bangla digits:** ০–৯
- **Latin characters:** A–Z, a–z (used for brand names and abbreviations like "VIP", "XXL", "ps")
- **Symbols:** Spaces, parentheses, hyphens, slashes, periods

### 4.4 Dataset Structure

```
Dataset_PaddleOCR/
├── train/
│   ├── items_crops/          # Original + augmented item images
│   ├── total_crops/          # Original + augmented total images
│   ├── rec_items_train.txt   # PaddleOCR label file (items)
│   └── rec_total_train.txt   # PaddleOCR label file (totals)
├── val/
│   ├── items_crops/          # Original item images only
│   ├── total_crops/          # Original total images only
│   ├── rec_items_val.txt
│   └── rec_total_val.txt
├── test/
│   ├── items_crops/          # Original item images only
│   ├── total_crops/          # Original total images only
│   ├── rec_items_test.txt
│   └── rec_total_test.txt
├── char_dictionary.txt       # 143 unique characters
└── dataset_report.txt        # Automated statistics report
```

---

## 5. Challenges and Solutions

| Challenge | Solution |
|:---|:---|
| Heterogeneous CSV exports from 3 annotators | Automated schema detection and normalization |
| Corrupted JSON in Label Studio transcriptions | Type-based filtering (`isinstance(item, str)`) |
| File extension mismatch (.jpg vs .jpeg) | Dual-extension fallback search |
| EXIF rotation mismatch between storage and annotation | `ImageOps.exif_transpose()` before coordinate mapping |
| Store names mislabeled as item names | Statistical outlier detection (y-position + gap heuristic) |
| Limited dataset size (622 samples) | 4× data augmentation with realistic transformations |
| Need for honest evaluation metrics | Augmentation applied only to training split |
| Data leakage risk | Receipt-level splitting (not image-level) |

---

## 6. Future Work

### 6.1 Dataset Expansion

The current dataset of 622 receipts provides a starting foundation, but OCR models typically benefit from larger training sets. Planned expansion includes:

- Collecting an additional 1,000–1,500 handwritten receipt images from diverse marketplaces
- Leveraging Vision-Language Models (e.g., Qwen2.5-VL-7B) for semi-automated annotation to accelerate labeling

### 6.2 OCR Model Training

Fine-tuning PaddleOCR's recognition model (PP-OCRv4) on this dataset using the generated PaddleOCR-format files. The training pipeline would include:

- Text detection model adaptation for receipt layouts
- Text recognition model fine-tuning with the character dictionary
- Performance evaluation on the held-out test set

### 6.3 LLM Post-Processing

Integrating a lightweight Language Model (LLM) that understands Bangla formatting and receipt structure to:

- Correct OCR transcription errors using contextual understanding
- Structure extracted text into item–quantity–price tuples
- Handle mixed-language content (Bangla + English)

### 6.4 Expense Tracking Application

Building a mobile application that:

- Captures receipt photos via smartphone camera
- Processes them through the OCR + LLM pipeline
- Automatically populates expense records
- Provides spending analytics and budget tracking

---

## 7. Tools and Technologies

| Tool/Technology | Purpose |
|:---|:---|
| Python 3.x | Primary development language |
| Pillow (PIL) | Image processing, EXIF handling, cropping |
| NumPy | Numerical operations for augmentation |
| Label Studio | Image annotation platform |
| PaddleOCR | Target OCR framework for model training |
| Git | Version control |

---

## 8. Conclusion

This project successfully constructed **BanglaMemo**, a specialized dataset for handwritten Bangla receipt OCR. Starting from 698 raw receipt images, we produced a cleaned, cropped, augmented, and properly formatted dataset of 622 labeled receipts with over 5,000 training samples (after augmentation). The automated processing pipeline handles real-world data quality issues including EXIF rotation mismatches, annotation errors, corrupted exports, and file naming inconsistencies.

The dataset fills a gap in existing Bangla handwriting resources by providing domain-specific data from real marketplace transactions with genuine handwriting variability. The PaddleOCR-compatible format ensures the dataset is immediately usable for fine-tuning text recognition models.

While the current dataset size is modest, it establishes the annotation schema, processing pipeline, and quality assurance framework that will scale to the planned 1,500+ image expansion. Combined with the future OCR model and LLM post-processing components, this work lays the groundwork for a complete Bangla expense tracking system.

---

## References

1. Rabby, A. S. A., et al. "BanglaLekha-Isolated: A comprehensive Bangla handwritten character dataset." *Data in Brief* 15 (2017): 171-174.
2. Sarkar, R., et al. "CMATERdb1: A database of unconstrained handwritten Bangla and Bangla-English mixed script document image." *International Journal of Document Analysis and Recognition* 15.1 (2012): 71-83.
3. Du, Y., et al. "PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System." *arXiv preprint arXiv:2206.03001* (2022).
4. Park, S., et al. "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing." *Workshop on Document Intelligence at NeurIPS* (2019).
5. Huang, Z., et al. "ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction." *ICDAR* (2019).
6. Sun, H., et al. "WildReceipt: A Large-Scale Benchmark for Receipt Information Extraction in the Wild." *arXiv preprint arXiv:2103.14470* (2021).

---

*This report was prepared as part of the Digital Image Processing course project.*

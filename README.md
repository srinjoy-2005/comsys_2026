# Handwritten Multi-Script Character Segmentation and Recognition
## A Two-Stage Pipeline for YOLO-Based Detection and Fine-tuned EfficientNet Classification

### Technical Documentation

#### Abstract
This document provides a comprehensive technical description of a multi-stage machine learning pipeline developed for the **Comsys 7 Hackathon**.
The system addresses the task of handwritten character segmentation and recognition across multiple scripts, specifically **Bengali and English**.
The pipeline comprises five principal phases: (1) synthetic training data generation via programmatic page rendering, (2) exploratory data analysis (EDA) of the competition dataset, (3) a two-stage detection and classification architecture combining YOLOv8 with EfficientNet-B0, (4) a Bengali character specialist model trained with focused data augmentation on a constrained real-world dataset, and (5) data and submission visualisation tooling developed for inspection and debugging.
Each phase is implemented as a standalone Kaggle-compatible notebook or Python script.
The final inference pipeline detects character bounding boxes using a fine-tuned YOLOv8 detector and classifies each crop using a fine-tuned EfficientNet-B0 model, producing structured output compatible with the competition submission format.

---

### Table of Contents
1. [Repository Structure and Component Overview](#1-repository-structure-and-component-overview)
2. [Phase I: Synthetic Dataset Generation (kaggle_text_classifier_data_generation.py)](#2-phase-i-synthetic-dataset-generation)
3. [Phase II: Exploratory Data Analysis (EDA.ipynb)](#3-phase-ii-exploratory-data-analysis)
4. [Phase III: Bengali Character Specialist Classifier (bengali.ipynb)](#4-phase-iii-bengali-character-specialist-classifier)
5. [Phase IV: YOLO Training — Synthetic Pre-training and Real Data Fine-tuning (YOLO_training.ipynb)](#5-phase-iv-yolo-training)
6. [Phase V: Text Classification Pipeline (text-classification.ipynb)](#6-phase-v-text-classification-pipeline)
7. [Two-Stage Inference and Submission Generation](#7-two-stage-inference-and-submission-generation)
8. [Viewer Utility (viewer.html)](#8-viewer-utility)
9. [Design Rationale and Key Engineering Decisions](#9-design-rationale-and-key-engineering-decisions)
10. [Environment and Dependencies](#10-environment-and-dependencies)

---

### How to Run the Pipeline

To reproduce the submission, execute the components in the following sequence:

1. **Synthetic Data Generation:** Run `kaggle_text_classifier_data_generation.py`. This script outputs a directory named `dataset_production_test1` containing synthetic page images and YOLO label files, and extracts character crops into `classifier_dataset`. The Discord webhook URL is entirely optional — if not provided, progress is printed to the console instead.
2. **YOLO Detection Training:** Run `YOLO_training.ipynb`.
   - *Note on paths:* The notebook assumes the synthetic dataset (`dataset_production_test1`) is mounted as an input dataset on Kaggle. If the dataset has not been copied to the working directory or if you are not running on Kaggle, **uncomment the dataset copy block** (Step 4 in the notebook) and adjust `INPUT_DIR` and `WORKING_DIR` paths to match your local filesystem.
   - This notebook outputs **`intermediate_boxes_expanded.json`**, which contains the candidate character bounding boxes for all test images.
3. **Character Classification:** Run `text-classification.ipynb`.
   - Provide `intermediate_boxes_expanded.json` as input.
   - *Required datasets:* This notebook requires the `emnist` dataset and the [basicfinal dataset](https://www.kaggle.com/datasets/mostafiz53/basicfinal) (`/kaggle/input/datasets/mostafiz53/basicfinal`).
   - *Note on paths:* Several intermediate files generated during this script's execution are fed back into it in specific folders. Ensure all file paths are correctly configured for your Kaggle or local environment.
   - This outputs the primary submission file: `submission_extended.csv`.
4. **Bengali Specialist Model:** Run `bengali.ipynb` **separately**. This notebook trains a specialist EfficientNet-B0 model focused specifically on Bengali character recognition, using focused data augmentation to handle the small real-world Bengali dataset. It outputs a CSV of improved Bengali predictions.
5. **Merge Predictions:** Run `merge_bengali_predictions.py`. This utility merges the improved Bengali outputs with the primary `submission_extended.csv` to produce the final competition submission.
6. **Inspect Results:** Open `viewer.html` in a browser to visually inspect predictions against the original images or review the submission CSV for any anomalies.

---

### 1. Repository Structure and Component Overview
The codebase is organised into interdependent components. Each addresses a distinct phase of the end-to-end pipeline, from raw data generation through to final competition submission.
The table below maps each artefact to its functional role.

| Artefact | Type | Purpose |
| :--- | :--- | :--- |
| `kaggle_text_classifier_data_generation.py` | Python Script | Generates 10,000 synthetic multi-script handwriting page images with per-character YOLO labels and classifier crops |
| `EDA.ipynb` | Jupyter Notebook | Performs corpus-level statistical analysis and bounding box visualisation on the JU_CMATER training set |
| `bengali.ipynb` | Jupyter Notebook | Trains a specialist EfficientNet-B0 model on Bengali character crops with focused augmentation; exports classifier weights |
| `YOLO_training.ipynb` | Jupyter Notebook | Two-stage YOLO pipeline: synthetic pre-training on generated data followed by fine-tuning on real annotated images |
| `text-classification.ipynb` | Jupyter Notebook | Classifies detected crops using an EfficientNet-B0 model trained on synthetic crops, EMNIST, and the basicfinal dataset |
| `merge_bengali_predictions.py` | Python Script | Merges the Bengali specialist model's output with the primary submission CSV |
| `viewer.html` | HTML Utility | Interactive browser-based viewer for inspecting training/test images with bounding box overlays and reviewing submission predictions |

---

### 2. Phase I: Synthetic Dataset Generation
**File:** `kaggle_text_classifier_data_generation.py`

#### 2.1 Motivation
Competition datasets for handwritten OCR tasks are frequently limited in size relative to the high number of character classes involved.
To mitigate this, a synthetic data generation pipeline was developed that programmatically renders multi-script text onto white A4-sized page canvases, producing paired image–label files suitable for both YOLO detector training and character-level classifier training.
The pipeline is designed to run on Kaggle's infrastructure and optionally sends real-time progress notifications via a Discord webhook.

#### 2.2 Configuration Parameters
The generator is governed by a centralised configuration block. Key parameters are summarised below.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `NUM_TEST_IMAGES` | 10,000 | Total synthetic page images to generate |
| `PAGE_WIDTH / PAGE_HEIGHT` | 800 × 1131 px | Canvas dimensions approximating A4 proportions at 96 dpi |
| `MAX_CROPS_PER_CLASS` | 1,000 | Safety cap on per-character classifier crops to avoid Kaggle inode exhaustion |
| Font size range | 32–46 pt | Randomly sampled per page to simulate natural writing size variability |
| Line spacing step | font_size + 8–18 px | Randomised inter-line gap to prevent uniform grid artefacts |

#### 2.3 Script Distribution
Each synthetic page is assigned one of four execution categories drawn from a weighted pool.
This distribution reflects the expected class imbalance in the competition data and prioritises the dominant English cursive script.

| Category | Weight (%) | Font Examples |
| :--- | :--- | :--- |
| english_cursive | 60% | Caveat, Dancing Script, Great Vibes, Pacifico, Zeyada (15 fonts) |
| english_plain | 15% | Lato, Montserrat, Roboto Flex, Coming Soon (4 fonts) |
| bengali | 13% | Mina, Baloo Da2 (3 fonts) |
| hindi | 12% | Amita, Hind, Kalam, Poppins (4 fonts) |

#### 2.4 Page Rendering Engine
Each synthetic page is generated by a grapheme-aware rendering pipeline summarised in three points:
- Canvas & font selection: initialise an 800 × 1131 white RGB canvas; pick a category-specific TrueType font at random (32–46 pt); sample up to 300 tokens and split them into Unicode grapheme clusters for correct complex-script handling.
- Rendering mechanics: draw text grapheme-by-grapheme with sampled near-black/dark-blue ink, apply a script-dependent overlap tightness (English cursive 1–3 px; Bengali/Hindi 2–4 px), and manage cursor wrapping and page overflow.
- Outputs: emit per-grapheme bounding boxes suitable for YOLO labels and save padded, resized classifier crops using filesystem-safe names.

#### 2.5 Label Generation
Each generated page produces two output files:
- **YOLO label file (.txt):** One line per grapheme cluster, encoded in normalised centre-format: `class_id cx cy w h`. All characters are assigned class 0, making this a single-class detection dataset.
- **Classifier crop directory:** Each grapheme cluster is cropped from the rendered canvas (with 2 px padding), resized, and saved to a directory named by the UTF-8 hexadecimal encoding of the character to avoid filesystem-forbidden character collisions. The per-class crop count is capped at 1,000 images.

---

### 3. Phase II: Exploratory Data Analysis
**File:** `EDA.ipynb`

```text
=============================================
         DATASET STATISTICS — TRAIN SET
=============================================
  Annotated images       :       17
  Total characters       :    6,679
  Unique characters      :      134
  Avg chars / image      :    392.9
  Min chars / image      :       38
  Max chars / image      :      725
  Avg bbox width  (px)   :     38.4
  Avg bbox height (px)   :     52.5
=============================================

Top 20 Most Frequent Characters:
  U+0065 'e' :    633  ███████████████████████████████████████
  U+0074 't' :    452  ████████████████████████████
  U+006E 'n' :    421  ██████████████████████████
  U+0069 'i' :    414  █████████████████████████
  U+0061 'a' :    405  █████████████████████████
  U+006F 'o' :    394  ████████████████████████
  U+0072 'r' :    332  ████████████████████
  U+0073 's' :    331  ████████████████████
  U+0075 'u' :    205  ████████████
  U+006C 'l' :    200  ████████████
  U+0063 'c' :    199  ████████████
  U+0064 'd' :    192  ████████████
  U+0070 'p' :    163  ██████████
  U+006D 'm' :    162  ██████████
  U+0068 'h' :    131  ████████
  U+0067 'g' :    104  ██████
  U+09BE 'া' :    101  ██████
  U+0066 'f' :     97  ██████
  U+0079 'y' :     75  ████
  U+0076 'v' :     60  ███
```
#### 3.1 Class Imbalance Analysis
A dedicated class imbalance cell computes the frequency distribution across all unique characters and flags those with fewer than *10 samples* (configurable via the `RARE_THRESHOLD` parameter).
The character frequency is plotted on a logarithmic scale to visualise the Zipfian distribution typical of natural language corpora.
Rare characters are enumerated with their Unicode code points to guide downstream augmentation strategies.

#### 3.2 Bounding Box Visualisation
Ten training images are sampled at random and displayed with overlaid bounding boxes.
Each box is rendered in green, with the decoded Unicode character displayed as a label above the box.
A text reconstruction function sorts annotations in reading order (top-to-bottom, left-to-right) using a quantised row-grouping strategy to handle natural handwriting skew, and produces a string representation of the page content for visual verification.

#### 3.3 Image Dimension Analysis
The resolution statistics of up to 500 sampled training images are computed, including width, height, and aspect ratio distributions.
These statistics inform the choice of input resolution for the YOLO detector, which was subsequently set to 1024 pixels to preserve the small character structures present in high-resolution document scans.

---

### 4. Phase III: Bengali Character Specialist Classifier
**File:** `bengali.ipynb`

> **Note:** This notebook was run **separately** from the main pipeline. It trains a specialist model focused on Bengali character recognition. The improved Bengali prediction rows are subsequently merged into the primary submission using `merge_bengali_predictions.py`.

#### 4.1 Strategy and Rationale
For the Bengali character classification component, a focused training strategy was adopted.
Given the small and highly imbalanced nature of the real Bengali character dataset, EfficientNet-B0 is trained with aggressive class-conditional data augmentation and a weighted Focal Loss.
This approach leverages the observation that the classification stage operates on YOLO-detected crops geometrically close to the training distribution, where a focused, high-accuracy model outperforms a more generalised one.

#### 4.2 Character Crop Extraction
Prior to training, Bengali character crops are extracted from the competition's training images. The extraction procedure:

- Iterate over all training annotation JSON files and parse annotations with `load_annotations()`.
- Filter annotations to Bengali Unicode block characters via `is_bengali_or_punctuation()`.
- Load corresponding images with OpenCV and compute pixel coordinates from normalised boxes (clamped to image bounds).
- Crop each character region with 2 px padding, resize to the classifier input size, and save under a filesystem-safe folder named by the Unicode label (e.g., `U+0985`).

A comprehensive class size report is generated at completion, listing each class, its sample count, and the total number of extracted images.

#### 4.3 Dataset and Augmentation
Two custom PyTorch Dataset classes handle the imbalanced data:

##### 4.3.1 ImbalancedCharDataset
The base dataset class wraps PyTorch's `ImageFolder` loader and applies *conditional augmentation* based on class frequency.
Classes with fewer than a configurable `minority_threshold` samples receive an aggressive `RandomSlice` augmentation that randomly crops 60–70% of the image's width and height from a random anchor point.
Majority classes receive only standard resizing, greyscale conversion, and normalisation.

##### 4.3.2 EfficientNetCharDataset
A subclass that overrides the transforms to produce *three-channel RGB output* at 224 × 224 pixels, as required by EfficientNet-B0's ImageNet pre-trained weights.
Normalisation uses the standard ImageNet statistics (mean [0.485, 0.456, 0.406], standard deviation [0.229, 0.224, 0.225]).

#### 4.4 Model Architecture
The classifier is built on **EfficientNet-B0** with ImageNet pre-trained weights.
The standard final linear layer (1,280 input features) is replaced with a custom head comprising a dropout layer (p = 0.1) followed by a linear projection to the number of target classes.
The remainder of the backbone is left unfrozen, permitting full end-to-end fine-tuning.

#### 4.5 Loss Function: Weighted Focal Loss
A custom Focal Loss implementation is used to simultaneously address **class** imbalance and **hard example mining**.
The focal loss formula is:
$$FL(p_t) = -(1 - p_t)^\gamma \cdot CE(p_t), \quad \text{where } \gamma = 2.0$$

Class weights from the inverse-frequency computation are passed to the cross-entropy base of the focal loss, providing a compound weighting mechanism that penalises both incorrect predictions on rare classes and easy examples on majority classes.

#### 4.6 Training Configuration

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| Optimiser | AdamW | Adaptive learning rates per parameter group with decoupled weight decay |
| Learning rate | 1 × 10⁻⁴ | Conservative initial rate appropriate for fine-tuning pre-trained weights |
| Weight decay | 0.0 | Disabled to allow the model to fully leverage the limited real-data distribution |
| Scheduler | CosineAnnealingLR | Smoothly reduces learning rate to η_min = 1 × 10⁻⁶ over 80 epochs |
| Epochs | 80 (max) | Training terminates early upon reaching 99.9% training accuracy |
| Batch size | 16 | Selected to balance GPU utilisation and stable gradient estimates on small crops |
| Focal loss γ | 2.0 | Standard value; down-weights well-classified easy examples |


---

### 5. Phase IV: YOLO Training
**File:** `YOLO_training.ipynb`

#### 5.1 Overview
The character detection component of the pipeline is built on YOLOv8-Small (YOLOv8s).
Training is conducted in two stages: an initial pre-training phase on the synthetic dataset, followed by a fine-tuning phase on real annotated images from the competition dataset.
This curriculum learning approach allows the detector to develop general character localisation capabilities from abundant synthetic data before specialising to the statistics of real handwritten documents.

#### 5.2 Dataset Preparation
The synthetic dataset generated in Phase I is restructured into the YOLO directory convention.
An 80/20 train–validation split is applied after random shuffling of the image list.
Both the image file and its paired label text file are copied to the appropriate subdirectory.
A YAML configuration file is generated specifying the dataset root, training and validation paths, and the single class name ('Character').

#### 5.3 Stage 1: Synthetic Data Pre-training
YOLOv8-Small is initialised from its ImageNet-pre-trained weights and trained on the synthetic dataset with the following configuration:

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| Epochs | 10 | Short pre-training to develop feature representations without fitting to synthetic style |
| Image size (imgsz) | 1024 px | Preserves small character detail in A4-scale document images |
| Batch size | 8 | Balanced for dual-GPU (T4 × 2 or P100) Kaggle configuration |
| Devices | `[0, 1]` | Distributed Data Parallel across both available Kaggle GPUs |
| Rotation (degrees) | 2.0° | Small rotation simulates natural handwriting tilt |
| Shear | 2.0° | Mild shear augmentation for style generalisation |
| Perspective | 0.001 | Minimal perspective warp; excessive values distort character shape |
| Mosaic | 1.0 | Full mosaic augmentation; beneficial for dense, multi-character scenes |
| Erasing | 0.2 | Random rectangular erasure to improve occlusion robustness |


#### 5.4 Stage 2: Fine-tuning on Real Annotated Data
The real competition training dataset consists of a small number of annotated handwriting images.
All annotation JSON files are converted from LabelMe format to YOLO normalised bounding box format.
A class-agnostic single-class formulation is maintained (all characters mapped to class 0).
The bounding box coordinates are clamped to [0, 1] to prevent out-of-bounds annotation errors.
A 14/3 train–validation split is used given the limited image count.
Fine-tuning is initialised from the Stage 1 checkpoint and trained with a conservative learning rate:

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| Epochs | 100 | Extended training to allow the model to adapt to real handwriting statistics |
| Image size | 1024 px | Consistent with Stage 1 for weight transfer compatibility |
| Batch size | 4 | Reduced to 4 for stability on small real-data batches; single GPU |
| Device | 0 (single GPU) | DDP disabled for tiny dataset; single GPU avoids synchronisation overhead |
| AMP | False | Disabled to prevent learning rate override by YOLO's automatic mixed precision logic |
| Optimiser | AdamW | Explicitly set to prevent YOLO's auto-selection from overriding the micro-learning rate |
| Initial LR (lr0) | 1 × 10⁻⁴ | Micro-learning rate to prevent catastrophic forgetting of Stage 1 representations |
| LR final ratio (lrf) | 0.01 | Final learning rate = lr0 × lrf = 1 × 10⁻⁶ |
| Warmup epochs | 0 | No warmup; fine-tuning begins from the calibrated micro-rate immediately |
| Box loss weight | Not specified | YOLO default; Stage 2 inherits Stage 1's box weight priorities |
| Mosaic | 0.5 | Reduced from Stage 1 to preserve the spatial statistics of real documents |

---

### 6. Phase V: Text Classification Pipeline
**File:** `text-classification.ipynb`

This stage processes the candidate boxes generated by the YOLO detector. To maximise accuracy, the pipeline incorporates extensive external datasets:
- The standard **EMNIST** dataset.
- The **[basicfinal](https://www.kaggle.com/datasets/mostafiz53/basicfinal)** dataset (`/kaggle/input/datasets/mostafiz53/basicfinal`).

The character bounding box data from `intermediate_boxes_expanded.json` are passed through an EfficientNet-B0 classifier trained on the aggregated dataset to produce the primary submission CSV (`submission_extended.csv`).

---

### 7. Two-Stage Inference and Submission Generation
#### 7.1 Pipeline Architecture
The final inference pipeline implements a two-stage detect-then-classify architecture. For each test image, the pipeline:
- **Stage 1 (Detection):** The fine-tuned YOLOv8 model processes the full-resolution test image and returns a set of candidate character bounding boxes. A low confidence threshold of 0.08 is used to maximise recall, accepting some false positives in preference to missed characters. An IoU threshold of 0.45 and a maximum detection count of 2,000 are applied.
- **Stage 2 (Classification):** Each bounding box crop is resized to 224 × 224 pixels, normalised with ImageNet statistics, and passed through the EfficientNet-B0 classifier. The argmax of the classifier's output logits is mapped to the corresponding Unicode label via the class index maintained during training.

#### 7.2 Output Format
For each test image, the predictions are serialised as a JSON string conforming to the competition's submission schema.
Each predicted character entry contains:
- `script` : Integer script identifier (1 for Bengali in the demonstrated case).
- `unicode_value` : The Unicode label string for the predicted character class.
- `bbox` : A list of four floating-point coordinates `[x1, y1, x2, y2]` in absolute pixel units, rounded to four decimal places.

The per-page prediction lists are assembled into a `submission_extended.csv` file with columns `page_id` and `predictions`, where `predictions` contains the JSON-serialised list of character dictionaries. Bengali rows are subsequently replaced with improved outputs from `bengali.ipynb` using `merge_bengali_predictions.py`.

---

### 8. Viewer Utility
**File:** `viewer.html`

A standalone browser-based viewer was developed to facilitate inspection of both training data and submission predictions without requiring a running Jupyter server.

**Features:**
- **Dataset viewer:** Load any training or test image and overlay the corresponding ground-truth or predicted bounding boxes with Unicode character labels rendered at each box location.
- **Submission viewer:** Load `submission_extended.csv` or the final merged submission CSV and step through predictions page by page, enabling rapid visual verification of classification outputs.
- **No server required:** The viewer is a single self-contained HTML file that runs entirely in the browser. Open it locally with any modern browser.

---

### 9. Design Rationale and Key Engineering Decisions
#### 9.1 Grapheme Cluster Segmentation
Standard string iteration in Python splits text at Unicode code-point boundaries, which incorrectly fragments complex script characters that span multiple code points (e.g., Bengali conjunct characters).
The `regex` library's `\X` pattern correctly segments text into **extended grapheme clusters** — the minimal units of human-perceived characters — ensuring that bounding boxes correspond to visually coherent character units rather than partial code points.

#### 9.2 UTF-8 Hexadecimal Directory Naming
Saving character crop directories under the UTF-8 hexadecimal encoding of the character (rather than the raw character itself) avoids filesystem failures arising from characters that are forbidden in file paths on certain operating systems (e.g., forward slashes in some Unicode characters, null bytes, or OS-reserved characters).
This encoding is reversible, allowing the original character to be recovered at inference time.

#### 9.3 Box Loss Weighting in the Bengali YOLO Run
In `bengali.ipynb`'s YOLO training cell, the box loss coefficient is set to 7.5 (relative to the classification loss coefficient of 0.5).
Because the task is single-class detection, accurate bounding box regression (localisation) is substantially more important than class discrimination.
The elevated box weight forces the detector to minimise IoU error aggressively, producing tighter character crops that improve downstream classifier accuracy.

#### 9.4 Single-GPU Fine-tuning for Small Datasets
Distributed Data Parallel training across multiple GPUs introduces gradient synchronisation overhead and can destabilise training when the effective batch size (physical batch × number of GPUs) is small relative to the dataset.
For Stage 2 fine-tuning on the small real-image subset, training is restricted to a single GPU with automatic mixed precision disabled, ensuring that the explicitly specified micro-learning rate is respected rather than being overridden by YOLO's automatic optimiser selection logic.

#### 9.5 Low Confidence Threshold at Inference
The YOLO detector is queried with a confidence threshold of 0.08 during inference, deliberately accepting false positive detections in order to maximise character recall.
In OCR tasks, missed characters (false negatives) typically incur a higher penalty in evaluation metrics than spurious detections, particularly when the downstream classifier can reject implausible crops based on low classifier confidence.
The resulting high-recall detection set is then filtered through the EfficientNet classifier, which provides a second stage of verification.

---

### 10. Environment and Dependencies
#### 10.1 Hardware
- Kaggle Notebook environment with GPU acceleration (tested on T4 × 2 and P100 configurations).
- CUDA-enabled PyTorch for GPU-accelerated training.
- Dual-GPU support utilised for Stage 1 YOLO pre-training via Distributed Data Parallel.

#### 10.2 Python Libraries

| Library | Version | Role |
| :--- | :--- | :--- |
| `ultralytics` | Latest | YOLOv8 model training and inference |
| `torch / torchvision` | ≥ 2.0 | Deep learning framework; EfficientNet-B0 backbone and training loop |
| `Pillow (PIL)` | Any recent | Image canvas creation, font rendering, and crop saving |
| `OpenCV (cv2)` | Any recent | Image loading and crop extraction from competition annotations |
| `regex` | Any recent | Unicode grapheme cluster segmentation via \X pattern — critical dependency |
| `numpy` | Any recent | Array operations for bounding box statistics and image data |
| `matplotlib` | Any recent | Distribution plots and bounding box visualisations in EDA phase |
| `tqdm` | Any recent | Progress bars for long-running extraction and training loops |
| `requests` | Any recent | Optional Discord webhook notifications for remote progress monitoring |
| `pandas` | Any recent | Submission CSV construction and display |
| `PyYAML` | Any recent | YOLO dataset configuration file generation |

# ---------------------------------x--------------------------------

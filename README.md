# End-to-End Object Detection with YOLO26

An end-to-end object detection project using YOLO26n, covering dataset preparation, model training, evaluation, and deployment through a Streamlit application.

## 1. Project Overview

This project implements an object detection pipeline using the YOLO26 model. It includes dataset inspection and preparation, duplicate detection, model training, performance evaluation, and an interactive application for image and video inference.

The trained model supports detection across 20 object categories.

### Main Objectives

- Inspect and validate the object detection dataset.
- Identify duplicate and near-duplicate images.
- Convert Pascal VOC annotations into YOLO format.
- Prepare training, validation, and testing splits.
- Train and compare two YOLO26n experiments.
- Evaluate the final model using standard detection metrics.
- Develop an interactive Streamlit application.
- Document the complete project workflow.

## 2. Dataset Description

The original dataset contained 5,012 JPG images and 5,012 corresponding Pascal VOC XML annotation files.

After dataset inspection and duplicate auditing, the prepared dataset contained 5,007 image-annotation pairs.

### Dataset Split

| Split | Images | Annotated Objects |
|---|---:|---:|
| Training | 4,009 | 12,522 |
| Validation | 499 | 1,648 |
| Testing | 499 | 1,484 |
| **Total** | **5,007** | **15,654** |

### Dataset Classes

The dataset contains 20 object categories:

1. Aeroplane
2. Bicycle
3. Bird
4. Boat
5. Bottle
6. Bus
7. Car
8. Cat
9. Chair
10. Cow
11. Dining Table
12. Dog
13. Horse
14. Motorbike
15. Person
16. Potted Plant
17. Sheep
18. Sofa
19. Train
20. TV Monitor

### Dataset Preparation

The preparation workflow included:

- Checking image and annotation file pairs.
- Inspecting XML annotation files.
- Validating bounding-box coordinates.
- Identifying duplicate and near-duplicate images.
- Removing identified duplicate pairs.
- Splitting the dataset into training, validation, and testing sets.
- Converting Pascal VOC annotations into YOLO format.
- Creating the `data.yaml` configuration file.
- Verifying the generated YOLO dataset.

The final cross-split duplicate audit reported zero exact-duplicate and zero near-duplicate candidates.

**Dataset availability:** The dataset is not included in this project package. The original dataset files must be obtained separately before running dataset preparation or training scripts.

## 3. Model Training

The project uses the YOLO26n model initialized with pretrained weights.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Model | YOLO26n |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Maximum Epochs | 50 |
| Optimizer | SGD |
| Random Seed | 42 |
| Pretrained Weights | Yes |
| Training Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |

### Experiment 1: Baseline YOLO26n

The first experiment established baseline performance using the initial training configuration.

| Parameter | Value |
|---|---|
| Learning Rate | 0.01 |
| Patience | 15 |
| Maximum Epochs | 50 |
| Actual Training Epochs | 16 |
| Training Time | 19 minutes 38 seconds |

### Experiment 2: Modified YOLO26n Configuration

The second experiment used a modified training configuration to investigate potential performance improvements.

| Parameter | Value |
|---|---|
| Learning Rate | 0.001 |
| Patience | 20 |
| Maximum Epochs | 50 |
| Actual Training Epochs | 50 |
| Training Time | 60 minutes 44 seconds |

Additional augmentation settings included mosaic, scaling, flipping, and HSV adjustments.

### Validation Results

| Metric | Baseline | Experiment 2 |
|---|---:|---:|
| Precision | 78.827% | 86.276% |
| Recall | 64.244% | 71.997% |
| mAP@0.50 | 74.916% | 82.622% |
| mAP@0.50:0.95 | 52.283% | 61.058% |

Experiment 2 achieved better validation performance across all four reported metrics.

The experiments involved changes beyond the learning rate, so the improvement cannot be attributed to the learning-rate change alone.

## 4. Final Model Evaluation

The best checkpoint from Experiment 2 was evaluated on the held-out test dataset.

### Test Results

| Metric | Performance |
|---|---:|
| Precision | 83.791% |
| Recall | 74.109% |
| mAP@0.50 | 83.033% |
| mAP@0.50:0.95 | 63.317% |
| GPU Inference Speed | 3.4 ms/image |

The test dataset contained 499 images and 1,484 annotated objects.

### Evaluation Visualizations

The `evaluation/experiment2_test/` directory contains:

- F1-confidence curve.
- Precision-recall curve.
- Precision-confidence curve.
- Recall-confidence curve.
- Confusion matrix.
- Normalized confusion matrix.
- Validation batch ground-truth visualizations.
- Validation batch prediction visualizations.

These visualizations help examine class-wise performance, confidence thresholds, detection errors, and model predictions.

### Checkpoint Verification

The deployed checkpoint was verified using SHA-256 hashing.

```text
BCC0DB4F31487968ED3072EA7B003C55686920B70C69E2C70B75173EADED118B
```

This hash matches the documented Experiment 2 best checkpoint.

## 5. Streamlit Application

The project includes an interactive Streamlit application that uses the trained YOLO26 model.

### Main Features

**Image Detection**
- Upload JPG, JPEG, and PNG images.
- Display the uploaded image and annotated predictions.
- View detected object classes and confidence scores.
- Inspect prediction summaries and bounding-box information.
- Download annotated image results.

**Video Detection**
- Upload MP4 videos.
- Perform frame-by-frame object detection.
- Display annotated video results.
- Download processed video output.

**Confidence Threshold**
- Adjust the confidence threshold.
- Update displayed detections according to the selected threshold.

**Prediction Summary**
- View the total number of detections.
- Inspect per-class detection counts.
- View confidence scores and bounding-box coordinates.

The application also includes input validation and error handling for unsupported or unreadable files.

YOLO26 uses a native NMS-free detection design, which is reflected in the inference workflow.

## 6. Installation and Execution

### Prerequisites

- A compatible Python installation.
- pip package manager.
- Project files.
- Trained model checkpoint.

### Step 1: Extract the Project

Extract the project ZIP file and open a terminal in the project root directory.

### Step 2: Create a Virtual Environment

```bash
python -m venv yolo_env
```

### Step 3: Activate the Environment

**Windows PowerShell**

```powershell
.\yolo_env\Scripts\Activate.ps1
```

**Windows Command Prompt**

```cmd
yolo_env\Scripts\activate.bat
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the Application

From the project root directory:

```bash
streamlit run app/app.py
```

The application will open in the browser. If it does not open automatically, use the local URL displayed in the terminal.

## 7. Project Directory Structure

```text
YOLO26-Object-Detection/
│
├── app/
│   └── app.py
│
├── evaluation/
│   └── experiment2_test/
│       ├── BoxF1_curve.png
│       ├── BoxPR_curve.png
│       ├── BoxP_curve.png
│       ├── BoxR_curve.png
│       ├── confusion_matrix.png
│       ├── confusion_matrix_normalized.png
│       ├── val_batch0_labels.jpg
│       ├── val_batch0_pred.jpg
│       ├── val_batch1_labels.jpg
│       ├── val_batch1_pred.jpg
│       ├── val_batch2_labels.jpg
│       └── val_batch2_pred.jpg
│
├── models/
│   └── best.pt
│
├── report/
│   └── YOLO26_Technical_Report.docx
│
├── scripts/
│   ├── check_duplicates.py
│   ├── convert_and_split.py
│   ├── diagnose_coordinates.py
│   ├── inspect_dataset.py
│   ├── inspect_duplicate_labels.py
│   ├── verify_yolo_dataset.py
│   ├── visualize_dataset.py
│   ├── visualize_duplicate_annotations.py
│   └── visualize_near_duplicates.py
│
├── training/
│   └── train.ipynb
│
├── data.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

The dataset and generated training outputs are not included in this project package.

## 8. Important Files

| File | Description |
|---|---|
| `app/app.py` | Streamlit application |
| `requirements.txt` | Python dependencies |
| `data.yaml` | Dataset configuration |
| `models/best.pt` | Final trained model checkpoint |
| `training/train.ipynb` | Model training notebook |
| `scripts/` | Dataset preparation, verification, and visualization scripts |
| `evaluation/` | Final evaluation visualizations |
| `report/YOLO26_Technical_Report.docx` | Editable technical report |
| `.gitignore` | Files and directories excluded from Git tracking |

## 9. Group Contributions

All group members collaborated throughout the project and shared responsibility for its development and completion.

Collective contributions included:

- Understanding the assignment requirements and planning the workflow.
- Inspecting, cleaning, validating, and preparing the dataset.
- Configuring and conducting YOLO26 training experiments.
- Evaluating trained models and interpreting performance metrics.
- Developing and testing the Streamlit application.
- Reviewing model predictions and evaluation visualizations.
- Preparing the technical report and project documentation.
- Organizing and reviewing the final submission package.

All members participated in the overall project development, discussions, review, and completion.

## 10. Conclusion

This project demonstrates an end-to-end object detection workflow using YOLO26, covering dataset preparation, model training, performance evaluation, and application development.

Two training experiments were conducted and compared. The second experiment achieved improved validation results, and its best checkpoint was selected for final test evaluation and deployment.

The Streamlit application provides an interface for image and video inference, confidence-threshold adjustment, prediction summaries, and annotated outputs.

Overall, the project brings together the major stages of an object detection system, from dataset preparation to model evaluation and interactive deployment.
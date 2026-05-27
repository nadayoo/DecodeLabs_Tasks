# 🤖 DecodeLabs — Artificial Intelligence Training Kit
**Batch: 2026 | Powered by DecodeLabs**

A hands-on industrial training series covering the core pillars of Artificial Intelligence — from rule-based chatbot logic, to supervised learning classification, to intelligent recommendation systems, to computer vision and OCR.

---

## 📁 Project Structure

```
DECODELABS_TASKS/
│
├── Task1/
│   └── Task1_chatbot.py              ← Task 1: Hello AI (Rule-Based Chatbot)
│
├── Task2/
│   ├── Task2_classification.py       ← Task 2: Data Classification Using AI
│   └── Task2_results.png             ← Auto-generated visualization
│
├── Task3/
│   ├── Task3_reclogic.py             ← Task 3: AI Recommendation Logic
│   └── project3_recommendations.png  ← Auto-generated visualization
│
├── Task4/
│   ├── Path1/
│   │   ├── ocr_recognition.py        ← Task 4 Path 1: OCR Text Recognition
│   │   ├── sample_text.jpg           ← OCR test image
│   │   └── ocr_preprocessed.jpg      ← Auto-generated pre-processed output
│   ├── Path2/
│   │   ├── object_detection.py       ← Task 4 Path 2: Object Detection
│   │   ├── sample_image.jpg          ← Object detection test image
│   │   ├── detection_output.jpg      ← Auto-generated annotated output
│   │   ├── frozen_inference_graph.pb ← MobileNet-SSD v3 weights
│   │   └── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt ← Model config
│   └── requirements.txt              ← Computer vision dependencies
│
└── README.md                         ← You are here
```

---

## ⚙️ Requirements

**Python Version:** 3.8+

Install all dependencies with one command:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn opencv-python pytesseract Pillow
```

| Library | Used In |
|---|---|
| `numpy` | Task 2, Task 3, Task 4 |
| `pandas` | Task 2, Task 3 |
| `scikit-learn` | Task 2, Task 3 |
| `matplotlib` | Task 2, Task 3 |
| `seaborn` | Task 2, Task 3 |
| `opencv-python` | Task 4 |
| `pytesseract` | Task 4 |
| `Pillow` | Task 4 |

---

## 🚀 Tasks Overview

---

### Task 1 — Hello AI (Rule-Based Chatbot)
> **Track:** Foundations | **Skill:** Logic Building, Python Basics

**Goal:** Build a foundational AI chatbot using rule-based decision making — the starting point before machine learning.

**Key Concepts:**
- If/else decision trees
- Basic input/output logic
- Introduction to algorithmic thinking
- Pattern matching with keywords

**Run:**
```bash
cd Task1
python Task1_chatbot.py
```

---

### Task 2 — Data Classification Using AI
> **Track:** Supervised Learning | **Skill:** Model Training, Pattern Recognition

**Goal:** Build a complete classification pipeline using the Iris benchmark dataset and the K-Nearest Neighbors (KNN) algorithm.

**Pipeline:**
```
Raw Data → Feature Scaling → Train/Test Split → KNN Model → Evaluation
```

**Key Concepts:**
- Supervised learning fundamentals
- Feature scaling with StandardScaler (Mean=0, Variance=1)
- Train-test split (80% train / 20% test) with shuffle
- K-Nearest Neighbors algorithm
- Elbow Method to find optimal K
- Confusion Matrix, Accuracy & F1 Score

**Dataset:** Iris — 150 samples, 4 features, 3 classes (Setosa, Versicolor, Virginica)

**Run:**
```bash
cd Task2
python Task2_classification.py
```

**Expected Output:**
```
Accuracy : 100.00%
F1 Score : 1.0000
Visualization saved → Task2_results.png
```

**Visualizations Generated (`Task2_results.png`):**
- Elbow Curve — Optimal K selection
- Confusion Matrix heatmap
- Feature Space scatter plot (Sepal vs Petal Length)

---

### Task 3 — AI Recommendation Logic
> **Track:** Personalization | **Skill:** Content-Based Filtering, Similarity Math

**Goal:** Build a Tech Stack Recommender that maps a user's skills to the most relevant career paths using TF-IDF vectorization and Cosine Similarity.

**Pipeline:**
```
User Input (3+ skills) → TF-IDF Vectors → Cosine Similarity → Sorted Top-5 Output
```

**Key Concepts:**
- Content-based filtering (vs collaborative filtering)
- TF-IDF weighting — penalizes generic terms, rewards specific ones
- Cosine Similarity — measures angular alignment between vectors
- IPO Framework: Input → Process → Output (Top-N List)
- Cold Start problem awareness

**Dataset:** 15 tech job roles with skill tags (Data Scientist, ML Engineer, DevOps Engineer, etc.)

**Run:**
```bash
cd Task3
python Task3_reclogic.py
```

**Interaction Example:**
```
Skill 1: python
Skill 2: machine_learning
Skill 3: tensorflow
Skill 4: done

#1  Machine Learning Engineer    Match: 82.4%
#2  AI Research Scientist        Match: 74.1%
#3  Data Scientist               Match: 61.3%
```

**Visualizations Generated (`project3_recommendations.png`):**
- Horizontal bar chart — Top 5 recommendations
- Full heatmap — All 15 roles ranked by similarity score

---

### Task 4 — Image & Text Recognition (Optional Mastery)
> **Track:** Computer Vision | **Skill:** Model Implementation, Visual Intelligence

**Goal:** Implement a basic recognition pipeline using pre-trained AI libraries — proving the ability to integrate real-world AI models into a functional workflow.

**Two Paths Available (complete one or both):**

---

#### Path 1 — Optical Character Recognition (OCR)
> Extract machine-readable text from raw images using Google's Tesseract engine.

**Pipeline:**
```
Raw Image → Grayscale → Gaussian Blur → Otsu Thresholding → Deskew → Tesseract OCR → Text Output
```

**Key Concepts:**
- Image pre-processing pipeline (grayscale, blur, thresholding, deskewing)
- Otsu's Method — automatic binary threshold calculation
- PSM (Page Segmentation Mode) tuning for different document types
- Confidence scoring per detected word

**PSM Modes:**
| Mode | Use Case |
|---|---|
| `--psm 3` | Fully automatic (varied layouts) |
| `--psm 6` | Single uniform block of text |
| `--psm 7` | Single text line (number plates) |
| `--psm 11` | Sparse scattered text (invoices) |

**Setup:**
```bash
# Install Tesseract (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/releases

pip install opencv-python pytesseract Pillow numpy
```

**Run:**
```bash
cd Task4/Path1
python ocr_recognition.py
```

**Expected Output:**
```
Image loaded — Shape: (1030, 736, 3)
Pre-processed image saved → ocr_preprocessed.jpg

--- Extracted Text ---
Discipline is one such trait that requires a set of rules...

Confidence Score: 87.3%
PASS — Meets the 80% threshold.
```

---

#### Path 2 — Object Detection (MobileNet-SSD)
> Identify and locate physical objects in images using a deep learning model with bounding box output.

**Pipeline:**
```
Raw Image → 4D Blob Construction → MobileNet-SSD Inference → NMS Filtering → Annotated Output
```

**Key Concepts:**
- Transfer Learning — leveraging ImageNet pre-trained weights
- Blob construction with mean subtraction and channel scaling
- MobileNet v3 depthwise separable convolutions
- Single Shot Detector (SSD) — single pass inference
- Non-Max Suppression (NMS) — removes duplicate bounding boxes
- Softmax confidence scoring (80% gate)
- Bounding box coordinate scaling (normalized → pixel space)

**Required Model Files:**
| File | Source |
|---|---|
| `frozen_inference_graph.pb` | MobileNet-SSD v3 weights (2020_01_14) |
| `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` | Model config |

Download both from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

**Setup:**
```bash
pip install opencv-python numpy
```

**Run:**
```bash
cd Task4/Path2
python object_detection.py
```


## 📊 Skills Progression

| Task | Concept | Algorithm | Output |
|---|---|---|---|
| Task 1 | Rule-Based Logic | If/Else + Keywords | Chatbot Responses |
| Task 2 | Supervised Learning | KNN | Classification + Metrics |
| Task 3 | Recommendation Logic | TF-IDF + Cosine Similarity | Ranked Career List |
| Task 4 | Computer Vision | Tesseract OCR / MobileNet-SSD | Extracted Text / Bounding Boxes |

---

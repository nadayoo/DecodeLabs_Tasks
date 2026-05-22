# 🤖 DecodeLabs — Artificial Intelligence Training Kit
**Batch: 2026 | Powered by DecodeLabs**

A hands-on industrial training series covering the core pillars of Artificial Intelligence — from rule-based chatbot logic, to supervised learning classification, to intelligent recommendation systems.

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
└── README.md                         ← You are here
```

---

## ⚙️ Requirements

**Python Version:** 3.8+

Install all dependencies with one command:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

| Library | Used In |
|---|---|
| `numpy` | Task 2, Task 3 |
| `pandas` | Task 2, Task 3 |
| `scikit-learn` | Task 2, Task 3 |
| `matplotlib` | Task 2, Task 3 |
| `seaborn` | Task 2, Task 3 |

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

## 📊 Skills Progression

| Task | Concept | Algorithm | Output |
|---|---|---|---|
| Task 1 | Rule-Based Logic | If/Else + Keywords | Chatbot Responses |
| Task 2 | Supervised Learning | KNN | Classification + Metrics |
| Task 3 | Recommendation Logic | TF-IDF + Cosine Similarity | Ranked Career List |

---


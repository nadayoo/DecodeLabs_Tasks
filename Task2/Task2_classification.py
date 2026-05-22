# ============================================================
# Project 2: Data Classification Using AI
# DecodeLabs Industrial Training Kit - Batch 2026
# Algorithm: K-Nearest Neighbors (KNN)
# Dataset: Iris Benchmark
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)

# ============================================================
# STEP 1: LOAD & UNDERSTAND THE DATASET
# ============================================================

print("=" * 55)
print("  PROJECT 2 — DATA CLASSIFICATION USING AI")
print("  DecodeLabs | Batch 2026")
print("=" * 55)

iris = load_iris()
X = iris.data          # Features: sepal/petal length & width
y = iris.target        # Labels: 0=Setosa, 1=Versicolor, 2=Virginica

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

print("\nDataset Overview:")
print(f"   Samples    : {X.shape[0]}")
print(f"   Features   : {X.shape[1]}")
print(f"   Classes    : {list(iris.target_names)}")
print("\n   First 5 rows:")
print(df.head())
print("\n   Class distribution:")
print(df['species'].value_counts())

# ============================================================
# STEP 2: FEATURE SCALING (The Gatekeeper Rule)
# StandardScaler → Mean=0, Variance=1
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling applied (StandardScaler)")
print(f"   Before scaling — mean: {X.mean(axis=0).round(2)}")
print(f"   After scaling  — mean: {X_scaled.mean(axis=0).round(2)}")

# ============================================================
# STEP 3: TRAIN-TEST SPLIT (80% Train / 20% Test)
# shuffle=True removes order bias
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"\n Data Split:")
print(f"   Training samples : {X_train.shape[0]} (80%)")
print(f"   Testing samples  : {X_test.shape[0]}  (20%)")

# ============================================================
# STEP 4: FIND OPTIMAL K (The Elbow Method)
# ============================================================

print("\nFinding Optimal K...")

error_rates = []
k_range = range(1, 31)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))

optimal_k = error_rates.index(min(error_rates)) + 1
print(f"   Optimal K = {optimal_k} (lowest error rate: {min(error_rates):.4f})")

# ============================================================
# STEP 5: TRAIN THE KNN MODEL
# ============================================================

model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)          
predictions = model.predict(X_test)  

print(f"\nModel trained with K={optimal_k}")

# ============================================================
# STEP 6: OUTPUT VALIDATION
# ============================================================

acc = accuracy_score(y_test, predictions)
f1  = f1_score(y_test, predictions, average='weighted')
cm  = confusion_matrix(y_test, predictions)

print("\n" + "=" * 55)
print("  OUTPUT VALIDATION")
print("=" * 55)
print(f"\n  Accuracy : {acc * 100:.2f}%")
print(f"  F1 Score : {f1:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, predictions,
                             target_names=iris.target_names))

# ============================================================
# STEP 7: VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Project 2 — Data Classification Using AI | DecodeLabs",
             fontsize=13, fontweight='bold')

# --- Plot 1: Elbow Curve (Optimal K) ---
axes[0].plot(k_range, error_rates, marker='o',
             color='navy', linewidth=2, markersize=5)
axes[0].scatter(optimal_k, error_rates[optimal_k - 1],
                color='orangered', zorder=5, s=120,
                label=f'Optimal K={optimal_k}')
axes[0].set_title('Tuning the Engine: Choosing K')
axes[0].set_xlabel('K Value')
axes[0].set_ylabel('Error Rate')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Confusion Matrix ---
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=axes[1], linewidths=0.5)
axes[1].set_title('Diagnostic Tool: Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

# --- Plot 3: Feature Distribution by Class ---
df_scaled = pd.DataFrame(X_scaled, columns=iris.feature_names)
df_scaled['species'] = pd.Categorical.from_codes(y, iris.target_names)
for species in iris.target_names:
    subset = df_scaled[df_scaled['species'] == species]
    axes[2].scatter(subset.iloc[:, 0], subset.iloc[:, 2],
                    label=species, alpha=0.7, s=50)
axes[2].set_title('Feature Space: Sepal vs Petal Length')
axes[2].set_xlabel('Sepal Length (scaled)')
axes[2].set_ylabel('Petal Length (scaled)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Task2_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nVisualization saved → project2_results.png")

# ============================================================
# STEP 8: TEST WITH NEW DATA POINT
# ============================================================

print("\nPredicting a new flower sample...")
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # typical Setosa
sample_scaled = scaler.transform(sample)
result = model.predict(sample_scaled)
proba  = model.predict_proba(sample_scaled)

print(f"   Input    : {sample[0]}")
print(f"   Predicted: {iris.target_names[result[0]].upper()}")
print(f"   Confidence: {proba.max() * 100:.1f}%")

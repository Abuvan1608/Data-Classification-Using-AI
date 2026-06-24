# ============================================================
#  DecodeLabs AI Internship | Batch 2026
#  Project 2: Data Classification Using AI (KNN)
#  Author   : ABUVAN
# ============================================================

# ----- Library Imports --------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets          import load_iris
from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.metrics           import (accuracy_score,
                                        classification_report,
                                        confusion_matrix,
                                        f1_score)

# ============================================================
# PHASE 1 — INPUT: Load & Explore the Iris Dataset
# ============================================================
print("=" * 60)
print("  DecodeLabs AI Internship | Project 2: KNN Classifier")
print("=" * 60)

iris      = load_iris()
X         = iris.data          # Features: sepal/petal length & width
y         = iris.target        # Labels : 0=Setosa, 1=Versicolor, 2=Virginica
features  = iris.feature_names
classes   = iris.target_names

# Convert to DataFrame for easy exploration
df = pd.DataFrame(X, columns=features)
df['species'] = pd.Categorical.from_codes(y, classes)

print("\n📊 Dataset Overview:")
print(f"   Samples    : {X.shape[0]}")
print(f"   Features   : {X.shape[1]}  → {list(features)}")
print(f"   Classes    : {list(classes)}")
print(f"   Class dist :\n{df['species'].value_counts().to_string()}")

print("\n🔍 First 5 rows:")
print(df.head().to_string(index=False))

# ============================================================
# PHASE 2 — PROCESS: Scaling → Split → Train
# ============================================================

# --- Step 1: Feature Scaling (Gatekeeper Rule) --------------
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)   # Mean=0, Variance=1

print("\n⚙️  Feature scaling applied (StandardScaler)")
print(f"   Mean  before scaling: {X.mean(axis=0).round(2)}")
print(f"   Mean  after  scaling: {X_scaled.mean(axis=0).round(2)}")

# --- Step 2: Train-Test Split (80/20) -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y          # Preserves class distribution
)

print(f"\n✂️  Train-Test Split (80/20, stratified):")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")

# --- Step 3: KNN Model — Instantiate → Fit → Predict --------
K     = 5
model = KNeighborsClassifier(n_neighbors=K)
model.fit(X_train, y_train)               # FIT  (memorize the map)
y_pred = model.predict(X_test)            # PREDICT (apply logic)

print(f"\n🤖 KNN Model trained  (K = {K})")

# ============================================================
# PHASE 3 — OUTPUT: Evaluate Performance
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred, average='weighted')
cm       = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("  MODEL PERFORMANCE REPORT")
print("=" * 60)
print(f"\n✅ Accuracy Score : {accuracy * 100:.2f}%")
print(f"✅ F1 Score       : {f1:.4f}  (weighted average)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

print("🔢 Confusion Matrix:")
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
print(cm_df.to_string())
print("\n   Rows = Actual | Columns = Predicted")

# ============================================================
# ELBOW METHOD — Find Optimal K
# ============================================================
print("\n🔍 Finding optimal K (Elbow Method)...")
error_rates = []
k_range     = range(1, 21)

for k in k_range:
    knn  = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, pred))

optimal_k = k_range[error_rates.index(min(error_rates))]
print(f"   Optimal K found : {optimal_k}  (lowest error rate)")

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("DecodeLabs Project 2 — KNN Iris Classification",
             fontsize=14, fontweight='bold')

# --- Plot 1: Elbow Curve ------------------------------------
axes[0].plot(list(k_range), error_rates, 'bo-', markersize=6)
axes[0].axvline(x=optimal_k, color='red', linestyle='--',
                label=f'Optimal K={optimal_k}')
axes[0].set_title("Elbow Curve: Error Rate vs K")
axes[0].set_xlabel("K Value")
axes[0].set_ylabel("Error Rate")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Confusion Matrix Heatmap -----------------------
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_title("Confusion Matrix")
axes[1].set_xticks(range(3)); axes[1].set_yticks(range(3))
axes[1].set_xticklabels(classes, rotation=20)
axes[1].set_yticklabels(classes)
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black',
                     fontsize=14, fontweight='bold')
plt.colorbar(im, ax=axes[1])

# --- Plot 3: Scatter (Petal Length vs Width) ----------------
colors = ['#e41a1c', '#377eb8', '#4daf4a']
for idx, (cls, col) in enumerate(zip(classes, colors)):
    mask = y == idx
    axes[2].scatter(X[mask, 2], X[mask, 3],
                    c=col, label=cls, alpha=0.7, edgecolors='k', linewidths=0.4)
axes[2].set_title("Feature Space: Petal Length vs Width")
axes[2].set_xlabel("Petal Length (cm)")
axes[2].set_ylabel("Petal Width (cm)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("knn_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n📈 Visualization saved → knn_results.png")

# ============================================================
# LIVE PREDICTION DEMO
# ============================================================
print("\n" + "=" * 60)
print("  LIVE PREDICTION DEMO")
print("=" * 60)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # Known Setosa
sample_scaled = scaler.transform(sample)
prediction    = model.predict(sample_scaled)
print(f"  Input  : sepal_len=5.1, sepal_w=3.5, petal_len=1.4, petal_w=0.2")
print(f"  Predicted class → {classes[prediction[0]].upper()}")
print("=" * 60)

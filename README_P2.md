# 🌸 Data Classification Using AI (KNN)

**DecodeLabs AI Internship | Batch 2026 | Project 2**

---

## 📌 Description

A supervised machine learning project that classifies Iris flower species using the **K-Nearest Neighbors (KNN)** algorithm. Built as part of the **DecodeLabs AI Internship** to demonstrate the complete ML pipeline — from raw data to intelligent predictions.

The project implements the full **IPO Framework**:
- **Input** → Iris dataset with feature scaling
- **Process** → Train-test split + KNN training
- **Output** → Confusion matrix + F1 score + visualizations

---

## ✨ Features

- 📊 Dataset exploration using pandas
- ⚙️ Feature scaling with `StandardScaler` (Mean=0, Variance=1)
- ✂️ Stratified 80/20 train-test split
- 🤖 KNN classification with K=5
- 🔍 Elbow method to find optimal K
- 📋 Full classification report (Precision, Recall, F1 Score)
- 🔢 Confusion matrix (TP, FP, FN, TN)
- 📈 3-panel visualization saved as `knn_results.png`
- 🎯 Live prediction demo

---

## 🛠️ Tech Stack

| Library      | Purpose                          |
|--------------|----------------------------------|
| Python 3.x   | Core language                    |
| scikit-learn | KNN, scaling, metrics, split     |
| pandas       | Data exploration & display       |
| numpy        | Array operations                 |
| matplotlib   | Visualizations                   |

---

## ▶️ How to Run

### Prerequisites

```bash
pip install scikit-learn pandas numpy matplotlib
```

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/data-classification-knn.git

# 2. Navigate to the folder
cd data-classification-knn

# 3. Run the classifier
python classification.py
```

### Expected Output

```
============================================================
  DecodeLabs AI Internship | Project 2: KNN Classifier
============================================================

📊 Dataset Overview:
   Samples    : 150
   Features   : 4
   Classes    : ['setosa', 'versicolor', 'virginica']

✅ Accuracy Score : 100.00%
✅ F1 Score       : 1.0000

📋 Classification Report:
              precision    recall  f1-score
    setosa       1.00      1.00      1.00
versicolor       1.00      1.00      1.00
 virginica       1.00      1.00      1.00
```

> A `knn_results.png` file is also generated with 3 plots: Elbow Curve, Confusion Matrix, and Feature Scatter Plot.

---

## 🏗️ ML Pipeline Architecture

```
INPUT
  └── Load Iris Dataset (150 samples, 4 features, 3 classes)
  └── Feature Scaling → StandardScaler (Mean=0, Variance=1)
        │
PROCESS
  └── Train-Test Split → 80% train / 20% test (stratified)
  └── KNN Model → fit(X_train, y_train)
  └── Predict  → model.predict(X_test)
        │
OUTPUT
  └── Accuracy Score + F1 Score
  └── Confusion Matrix (TP / FP / FN / TN)
  └── Classification Report
  └── Visualization → knn_results.png
```

---

## 📂 File Structure

```
data-classification-knn/
│
├── classification.py   # Main ML pipeline script
├── knn_results.png     # Auto-generated visualization (after run)
└── README.md           # Project documentation
```

---

## 🧠 Key Concepts Demonstrated

- **Supervised Learning** — machine learns from labeled data
- **Feature Scaling** — prevents distance bias in KNN
- **Train-Test Split** — validates model on unseen data
- **KNN Algorithm** — classifies by majority vote of K neighbors
- **Elbow Method** — finds optimal K value
- **F1 Score** — balanced metric beyond simple accuracy
- **Confusion Matrix** — full TP/FP/FN/TN breakdown

---

## 👨‍💻 Author

**ABUVAN**  
ECE Student | AI Intern @ DecodeLabs (Batch 2026)

---

*Built with ❤️ at DecodeLabs — "We do not write the rules. We provide history, and the machine derives the logic."*

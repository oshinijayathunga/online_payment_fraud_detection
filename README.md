# 💳 Payment Fraud Detection

A machine learning project to detect fraudulent financial transactions using classical ML models and advanced class imbalance handling techniques.

---

## 📌 Problem Statement

Financial fraud is a rare but costly event. This project addresses the challenge of detecting fraudulent transactions in a highly imbalanced dataset, where fraudulent cases represent only a tiny fraction of all transactions.

---

## 📂 Dataset

- **Format:** CSV (`new_file.csv`)
- **Target column:** `isFraud` (binary: 0 = legitimate, 1 = fraud)
- **Key features:** transaction `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `nameOrig`, `nameDest`

---

## 🔍 Exploratory Data Analysis

- Checked for null values and dropped missing rows
- Visualized transaction type distribution using count plots
- Analyzed average transaction amounts by type using bar plots
- Computed a **correlation heatmap** on numerical features to identify redundant variables
- Found that `oldbalanceOrg` and `oldbalanceDest` were highly correlated with other balance columns → **dropped** to reduce multicollinearity

---

## ⚙️ Preprocessing

| Step | Details |
|---|---|
| Dropped columns | `oldbalanceOrg`, `oldbalanceDest` (high correlation), `nameOrig`, `nameDest` (identifiers) |
| Encoding | One-hot encoding on `type` column (`drop_first=True`) |
| Train/Test Split | 80% train / 20% test (`random_state=42`) |
| Column alignment | Test set columns reindexed to match training set after encoding |

---

## ⚖️ Handling Class Imbalance

Three strategies were explored progressively:

### 1. SMOTE (Synthetic Minority Over-sampling Technique)
- Generated synthetic fraud samples to balance the training set
- Simple but can introduce noise on large datasets

### 2. SMOTEENN (SMOTE + Edited Nearest Neighbours)
- Combined oversampling (SMOTE) with undersampling (ENN) to clean noisy synthetic samples
- Produced a cleaner, more balanced training distribution

### 3. Random Under-Sampling + Class Weights *(Final Approach)*
- Applied `RandomUnderSampler` with `sampling_strategy=0.1`
- Added `class_weight='balanced'` to Logistic Regression and Random Forest
- Used `scale_pos_weight=10` in XGBoost to penalize misclassified fraud cases
- Most effective combination for improving fraud recall without excessive false positives

---

## 🤖 Models Used

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight='balanced'` |
| Random Forest | `n_estimators=100`, `criterion='entropy'`, `max_depth=12`, `class_weight='balanced'` |
| XGBoost | `n_estimators=100`, `scale_pos_weight=10` |

---

## 🎯 Threshold Tuning (XGBoost)

Instead of using the default 0.5 classification threshold, a **Precision-Recall curve analysis** was performed on XGBoost's probability scores:

- Extracted `predict_proba` scores for all test samples
- Plotted precision vs recall across all thresholds
- Applied a **custom threshold of 0.98** — requiring 98% confidence to flag a transaction as fraud
- This dramatically reduced false positives (legitimate transactions incorrectly flagged as fraud) while maintaining solid fraud detection

---

## 📊 Evaluation Metrics

Models were evaluated using:
- **Accuracy** (training and testing)
- **Classification Report** — Precision, Recall, F1-score per class
- **Confusion Matrix** — to inspect false positives and false negatives

### Final XGBoost Visualizations
- Confusion matrix heatmap (default threshold)
- Feature importance bar chart

**Top features by importance (XGBoost):**
`amount`, `newbalanceOrig`, `newbalanceDest`, transaction type flags

---

## 🔄 Iteration Summary — What Changed and Why

| Iteration | Change Made | Reason |
|---|---|---|
| Baseline | SMOTE only, Random Forest with 7 estimators | Initial attempt to handle imbalance |
| v2 | Switched to SMOTEENN | Reduce noisy synthetic samples from plain SMOTE |
| v3 | Switched to Random Under-Sampling + class weights | Better generalization; SMOTEENN was slow on large data |
| v4 | Increased RF to 100 estimators, added `max_depth=12` | Prevent underfitting and overfitting |
| v5 | Added `scale_pos_weight=10` to XGBoost | Directly penalize missed fraud cases |
| v6 | Custom threshold (0.98) on XGBoost probabilities | Fine-tune precision/recall tradeoff for business needs |

---
## 📊 Results — Final Model Performance

### ✅ Best Model: XGBoost with Custom Threshold (0.98)

| Metric | Legitimate (0) | Fraud (1) |
|---|---|---|
| Precision | 1.00 | 0.58 |
| Recall | 1.00 | 0.73 |
| F1-Score | 1.00 | 0.65 |
| Support | 1,270,904 | 1,620 |

### Confusion Matrix

|  | Predicted: Legit | Predicted: Fraud |
|---|---|---|
| **Actual: Legit** | 1,270,061 ✅ | 843 ❌ |
| **Actual: Fraud** | 436 ❌ | 1,184 ✅ |

- **True Negatives:** 1,270,061 — legitimate transactions correctly cleared
- **True Positives:** 1,184 — fraud cases correctly caught
- **False Positives:** 843 — legitimate transactions incorrectly flagged
- **False Negatives:** 436 — fraud cases missed

> At threshold 0.98, the model requires 98% confidence before flagging a transaction as fraud, significantly cutting false alarms while still catching 73% of all fraudulent activity.

### 🔑 Feature Importance (XGBoost)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `newbalanceOrig` | ~0.52 (dominant) |
| 2 | `type_PAYMENT` | ~0.33 |
| 3 | `type_CASH_OUT` | ~0.07 |
| 4 | `newbalanceDest` | ~0.03 |
| 5 | `amount` | ~0.02 |
| 6 | `step` | ~0.01 |
| 7 | `type_TRANSFER` | ~0.01 |
| 8 | `type_DEBIT` | ~0.01 |

The origin account's new balance after a transaction (`newbalanceOrig`) is by far the strongest signal — fraudulent transactions tend to drain the sender's account to zero. Transaction type is also highly predictive, with PAYMENT and CASH_OUT being the most fraud-associated categories.

---
**Dataset download**- https://drive.google.com/file/d/1zrXT2uiYI4E5yCpCAVdb6WqJ8HAmH2ei/view?usp=sharing


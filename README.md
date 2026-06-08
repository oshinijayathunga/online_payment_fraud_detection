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




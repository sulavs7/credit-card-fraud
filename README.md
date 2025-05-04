# Credit Card Fraud Detection System


A comprehensive comparison of machine learning models for fraud detection, focusing on handling class imbalance and optimizing operational metrics.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Modeling Approach](#modeling-approach)
- [Results Analysis](#results-analysis)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [What I Learned](#what-i-learned)
- [Future Work](#future-work)

## Project Overview
This project compares four machine learning models for fraud detection in highly imbalanced credit card transactions (1:492 fraud ratio). Focus areas include:
- Handling class imbalance through different techniques
- Threshold optimization for operational metrics
- Preventing model overfitting
- Business impact analysis of false positives/negatives

## Dataset
**Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
**Characteristics:**
- 284,807 transactions (492 frauds)
- 30 numerical features (PCA transformed)
- Class distribution: 0.172% fraudulent

## Preprocessing
**Two Distinct Pipelines:**  
1. **SMOTE Pipeline (RF/Logistic):**
   - Applied SMOTE oversampling
   - StandardScaler normalization
   - Generated `.pkl` files for reproducibility

2. **Native Imbalance Pipeline (XGBoost/CatBoost):**
   - Preserved original distribution
   - Used class weighting
   - Separate `.pkl` files for processing

**Common Steps:**
- Train-test split (80-20)
- Stratified sampling
- Feature scaling (excluding 'Amount')

## Modeling Approach

### 1. Random Forest Classifier
- **Imbalance Handling:** SMOTE oversampling
- **Key Configuration:**
  - 100 estimators
  - Max depth 10
  - Class weight balanced
- **Result Highlight:** Perfect training metrics but potential overfitting

### 2. XGBoost Classifier
- **Imbalance Handling:**
  - Scale_pos_weight = 580 (1:580 class ratio)
  - Threshold tuning (0.30)
- **Key Configuration:**
  - Learning rate 0.01
  - Max depth 5
  - Early stopping rounds 50

### 3. CatBoost Classifier
- **Imbalance Handling:**
  - class_weights = [0.1, 0.9]
  - Threshold tuning (0.60)
- **Key Configuration:**
  - 500 iterations
  - learning_rate 0.05
  - depth 7

### 4. Logistic Regression (Baseline)
- **Imbalance Handling:** SMOTE oversampling
- **Key Configuration:**
  - L2 regularization
  - Max iterations 1000

## Results Analysis

### Performance Comparison
| Model          | Test Recall | Test Precision | F1 Score | ROC-AUC | PR-AUC | False Positives | False Negatives |
|----------------|-------------|----------------|----------|---------|--------|-----------------|-----------------|
| Random Forest  | 91.32%      | 93.53%         | 92.40%   | 91.32%  | -      | 12              | 17              |
| XGBoost (0.30) | 88.91%      | 83.02%         | 85.72%   | -       | 75.36% | 38              | 21              |
| CatBoost (0.60)| 88.40%      | 89.65%         | 89.02%   | -       | 72.44% | 19              | 22              |

### Critical Observations
1. **Overfitting Spectrum:**
   - Random Forest: 100% training accuracy → 91.32% test recall
   - XGBoost: 2.8% F1 drop (train 89.19% → test 86.31%)
   - CatBoost: Only 1.3% F1 difference

2. **Threshold Impact:**
   - **XGBoost (0.30 → 0.50):**
     - Recall ↓2.6% (88.91% → 86.31%)
     - Precision ↑9.5% (83.02% → 92.57%)
   - **CatBoost (0.50 → 0.60):**
     - Precision ↑5.6% (84.09% → 89.65%)
     - Recall ↔88.40%

3. **Cost Analysis (Per 100k Transactions):**
   | Model          | Fraud Caught | Legit Blocked | Fraud Missed |
   |----------------|--------------|---------------|--------------|
   | Random Forest  | 913          | 21            | 87           |
   | XGBoost        | 889          | 67            | 111          |
   | CatBoost       | 884          | 33            | 116          |


## Installation
1. Clone repository:
2. Install requirements:
3. Create a folder Named Data:
4. Run preprocessing:
    python scripts/preprocess_smote.py
    python scripts/preprocess_native.py
6. 





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
  - Scale_pos_weight =100 (1:100 class ratio)
  - Threshold tuning (0.30)
- **Best Configuration:**
- 'subsample': 0.8
-  'reg_lambda': 0.5
-  'reg_alpha': 0.5
-  'n_estimators': 200
-  'min_child_weight': 5
-  'max_depth': 4
-  'learning_rate': 0.05
-  , 'gamma': 0.2
-  'colsample_bytree': 0.6

### 3. CatBoost Classifier
- **Imbalance Handling:**
  - Scale_pos_weight =100 (1:100 class ratio)
  - Threshold tuning (0.60)
- **Key Configuration:**
  - 'subsample': 0.6
  -  'learning_rate': 0.1
  -  'l2_leaf_reg': 5
  -  'iterations': 200
  -  'grow_policy': 'Depthwise'
  -  'depth': 6

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
   `git clone https://github.com/sulavs7/credit-card-fraud.git`
3. Install requirements:
   `pip install -r requirements.txt`
5. Create a folder Named Data:
6. 4. Download the dataset
   Visit Kaggle Credit Card Fraud Detection Dataset

   Download the dataset 

   Place it inside the data/ directory
7. Run preprocessing:
    Run: `data_preprocessing.ipynb`
    Run:`without_smote_processing.ipynb`

## What I Learned

### Imbalance Handling
- **SMOTE Double-Edged Sword**: While effective for generating synthetic minority samples, SMOTE can amplify noise and lead to overfitting - evident in Random Forest's perfect training scores that didn't fully translate to test data
- **scale_pos_weight**: Class weighting (especially in XGBoost/CatBoost) proved superior for preserving data integrity, avoiding synthetic sample generation while maintaining 88-91% recall
- **Technique Selection**: SMOTE remains valuable for linear models like Logistic Regression, but tree-based models achieved better natural imbalance handling without oversampling

### Model Behavior
- **Tree Resilience**: Random Forest/XGBoost/CatBoost showed inherent robustness to imbalance, needing only class weights (no SMOTE) to achieve >88% recall
- **Linear Model Limitations**: Logistic Regression struggled despite SMOTE, confirming tree-based models' superiority for skewed financial data
- **Threshold Dynamics**: 
  - 0.30 threshold boosted XGBoost's recall by 2.6% (cost: 9.5% precision loss)
  - 0.60 threshold increased CatBoost's precision by 5.6% with negligible recall impact


### Validation Strategy
- **PR-AUC Superiority**: Used PR curves instead of ROC-AUC for evaluation (0.75 vs 0.91 ROC), better reflecting real-world fraud detection needs
- **Stratification Necessity**: Preserved 0.172% fraud rate in test sets through careful splitting - crucial for valid performance measurement
- **Threshold-Specific Metrics**: Reported precision/recall at operational thresholds rather than default 0.5, increasing business relevance




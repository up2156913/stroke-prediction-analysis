# Stroke Prediction Analysis — Machine Learning Pipeline

A comprehensive machine learning analysis of a healthcare dataset (5,110 patient records) for stroke prediction, BMI regression, and patient clustering. Built as part of an MSc in AI and Machine Learning at the University of Portsmouth.

## Project Overview

This project applies a full ML pipeline across three analytical tasks using a real-world stroke dataset:

| Task | Algorithms | Key Result |
|------|-----------|------------|
| **Classification** | Logistic Regression, Random Forest, XGBoost | 72% stroke recall with Random Forest (63.6% improvement over baseline) |
| **Regression** | Linear Regression, Random Forest | ~25% BMI variance explained (R² ≈ 0.25) |
| **Clustering** | K-Means, Hierarchical (Ward's) | 8 clinically meaningful patient segments identified |

## Dataset

The [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) contains 5,110 patient records with 12 attributes covering demographics, lifestyle factors, and clinical measurements. The target variable (`stroke`) exhibits severe class imbalance at a 19.5:1 ratio (only 4.87% positive cases).

## Key Techniques

### Data Preparation
- **MICE Imputation** — Multivariate imputation for 201 missing BMI values, preserving correlational structure (98.34% variance retention)
- **SMOTE** — Synthetic minority oversampling to address 19.5:1 class imbalance, applied only to training data
- **Feature Engineering** — Clinical discretisation of age, BMI, and glucose into WHO/ADA-aligned categories; composite cardiovascular risk score
- **Encoding Strategy** — Label encoding for binary variables, one-hot encoding (k−1) for nominal variables to prevent multicollinearity

### Classification
- Threshold optimisation via precision-recall curve analysis for each algorithm
- Random Forest with `balanced_subsample` weighting and aggressive threshold (0.15) achieved best stroke detection
- Comprehensive evaluation: confusion matrices, ROC-AUC, per-class metrics, and radar chart comparison

### Regression
- BMI prediction using 15 features (stroke excluded due to temporal causality concerns)
- Stratified train-test split by BMI quartiles
- Both models converge at ~R² = 0.25, indicating a data-driven performance ceiling rather than algorithmic limitation

### Clustering
- Optimal k=8 selected via elbow method, silhouette score, Calinski-Harabasz, and Davies-Bouldin indices
- Both algorithms independently identified the same high-risk elderly subgroup (17% stroke rate, 100% heart disease prevalence)
- PCA projection and silhouette analysis for cluster validation

## Project Structure

```
├── Stroke.ipynb                         # Full analysis notebook (165 cells)
├── healthcaredatasetstrokedata.csv      # Source dataset
└── README.md
```

## Results Summary

### Classification — Stroke Detection

| Model | Stroke Recall | Accuracy | ROC-AUC | False Negatives |
|-------|:------------:|:--------:|:-------:|:---------------:|
| Random Forest | **72%** | 75.8% | 0.786 | 14 |
| Logistic Regression | 44% | 90.1% | **0.802** | 28 |
| XGBoost | 26% | 93.3% | 0.797 | 37 |

Random Forest is recommended for clinical deployment — its 72% sensitivity minimises missed strokes, where preventing fatalities outweighs the cost of additional follow-up investigations.

### Regression — BMI Prediction

| Model | Test RMSE | Test R² | Overfitting (R² gap) |
|-------|:---------:|:-------:|:-------------------:|
| Linear Regression | **6.73** | **0.255** | −0.018 |
| Random Forest | 6.75 | 0.251 | 0.360 |

Both models plateau at ~25% variance explained. The modest performance reflects missing genetic, dietary, and lifestyle factors rather than algorithmic weakness.

### Clustering — Patient Segmentation

| Metric | K-Means | Hierarchical |
|--------|:-------:|:------------:|
| Silhouette Score | **0.227** | 0.218 |
| Calinski-Harabasz | **735.5** | 693.1 |
| Highest Cluster Stroke Rate | 17.0% | 17.0% |

Both methods converge on the same high-risk subgroup (Cluster 5: mean age 68.2, 100% heart disease, 17% stroke rate), validating it as genuine data structure.

## Tech Stack

- **Python** — pandas, NumPy, matplotlib, seaborn
- **ML** — scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Imputation** — IterativeImputer (MICE)
- **Evaluation** — ROC-AUC, confusion matrices, silhouette analysis, PCA

## How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/stroke-prediction-analysis.git
cd stroke-prediction-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn

# Launch the notebook
jupyter notebook Stroke.ipynb
```

## Licence

This project is for educational and portfolio purposes. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) under its original terms.

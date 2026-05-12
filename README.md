# Predicting Type 2 Diabetes Using Machine Learning

> **COMP 3608 - Intelligent Systems | B-Rank Project**  
> A comparative machine-learning study of Type 2 diabetes prediction across three datasets, three algorithms, multiple imbalance-handling strategies, feature-importance analysis, top-k feature testing, and cross-dataset generalisation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Research Questions](#research-questions)
- [Datasets](#datasets)
- [Algorithms](#algorithms)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Notebook Run Order](#notebook-run-order)
- [Experimental Design](#experimental-design)
- [Results Summary](#results-summary)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Project Management](#project-management)

---

## Project Overview

Type 2 diabetes is a major public-health problem where earlier risk detection can support faster screening, clinical follow-up, and better resource planning. This project investigates whether machine-learning models can predict diabetes using clinical, demographic, lifestyle, and survey-based features.

The core research question is:

> **Which machine-learning models predict Type 2 diabetes reliably across different populations, and which predictors matter most?**

Instead of optimising one model on one dataset, this project compares:

- **3 datasets**
- **3 algorithms**
- **3 imbalance-handling strategies**
- **feature importance**
- **top-k feature subsets**
- **cross-dataset generalisation**

The project treats machine learning as a **screening-support tool**, not as a standalone diagnostic system. The goal is to identify useful predictive patterns while recognising that clinical confirmation is still required.

---

## Team Members

| Name | Student ID | Primary Responsibilities |
|---|---:|---|
| Tamika Andrews | 816039421 | EDA, preprocessing, and experiments for the Diabetes Prediction dataset |
| Dylan Corbett | 816039487 | EDA, preprocessing, and experiments for the CDC BRFSS dataset |
| Kieron Seepersad | 816041436 | EDA, preprocessing, and experiments for the Pima Indians dataset |

---

## Research Questions

1. **Does dataset type affect model performance and generalisability?**  
   We compare a small clinical benchmark dataset, a large clinical-style dataset, and a large public-health survey dataset.

2. **Which predictors are most useful for diabetes prediction?**  
   We compare direct clinical predictors such as glucose and HbA1c with indirect survey/lifestyle predictors such as BMI, age, blood pressure, cholesterol, physical activity, and general health.

3. **Which algorithm performs best under different data conditions?**  
   Logistic Regression, LightGBM, and XGBoost are compared across raw, balanced, and tuned settings.

4. **Can models transfer across datasets?**  
   Cross-dataset testing checks whether models trained on one population can generalise to another without retraining.

---

## Datasets

All raw datasets are stored in `data/raw/`.

| Dataset | File | Shape | Target | Main Feature Types | Notes |
|---|---|---:|---|---|---|
| **Pima Indians Diabetes** | `diabetes.csv` | 768 rows × 9 columns | `Outcome` | Glucose, BMI, blood pressure, age, pregnancies, insulin | Small clinical benchmark; contains zero/missing-value concerns |
| **Diabetes Prediction Dataset** | `diabetes_prediction_dataset.csv` | 100,000 rows × 9 columns | `diabetes` | HbA1c, blood glucose, BMI, age, hypertension, heart disease, smoking, gender | Strongest internal model performance due to direct clinical predictors |
| **CDC BRFSS 2015 Diabetes Health Indicators** | `diabetes_binary_health_indicators_BRFSS2015.csv` | 253,680 rows × 22 columns | `Diabetes_binary` | BMI, general health, blood pressure, cholesterol, age, activity, healthcare access | Largest dataset; strongly imbalanced survey-based dataset |

### Dataset Class Balance

| Dataset | Negative Class | Positive Class | Main Concern |
|---|---:|---:|---|
| Pima Indians | ~65.1% | ~34.9% | Small sample size and missing/zero medical values |
| Diabetes Prediction | ~91.5% | ~8.5% | Strong class imbalance |
| CDC BRFSS | ~86.1% | ~13.9% | Strong class imbalance and no direct glucose/HbA1c lab features |

---

## Algorithms

| Algorithm | Why Used | Outlier Handling |
|---|---|---|
| **Logistic Regression** | Interpretable baseline model; useful for comparing against more complex models | Most sensitive to outliers because extreme values can influence the linear decision boundary; scaling is used |
| **LightGBM** | Efficient gradient-boosted tree model for tabular data | More robust than Logistic Regression because trees split by thresholds |
| **XGBoost** | Strong boosted-tree baseline with regularisation and good tabular performance | Generally robust to outliers because unusual values mostly affect local tree splits |

---

## Repository Structure

```text
Predicting-Type-2-Diabetes/
│
├── data/
│   ├── raw/
│   │   ├── diabetes.csv
│   │   ├── diabetes_prediction_dataset.csv
│   │   └── diabetes_binary_health_indicators_BRFSS2015.csv
│   │
│   └── preprocessed/
│       ├── pima_preprocessed/
│       │   ├── pima_X_train_scaled.csv
│       │   ├── pima_X_test_scaled.csv
│       │   ├── pima_y_train.csv
│       │   └── pima_y_test.csv
│       │
│       ├── diabetes_prediction_dataset_preprocessed/
│       │   ├── X_train_scaled.parquet
│       │   ├── X_train_scaled_smote.parquet
│       │   ├── X_test_scaled.parquet
│       │   ├── y_train.parquet
│       │   ├── y_train_smote.parquet
│       │   └── y_test.parquet
│       │
│       └── brfss_preprocessed/
│           ├── brfss_train_original.parquet
│           ├── brfss_train_smote.parquet
│           └── brfss_test.parquet
│
├── notebooks/
│   ├── preprocessing/
│   │   ├── pima_eda_preprocessing.ipynb
│   │   ├── diabetes_prediction_eda_preprocessing.ipynb
│   │   └── brfss_eda_preprocessing.ipynb
│   │
│   └── experiments/
│       ├── pima_experiments.ipynb
│       ├── dataset_2_diabetes_prediction_experiments.ipynb
│       └── brfss_eda_experiments.ipynb
│
└── README.md

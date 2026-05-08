# Predicting Type 2 Diabetes Using Machine Learning

> **COMP 3608 - Intelligent Systems | B-Rank Project**
> A three-dataset, three-algorithm comparative study on the generalisability of diabetes prediction models across diverse real-world populations.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Research Questions](#research-questions)
- [Datasets](#datasets)
- [Algorithms](#algorithms)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebooks](#running-the-notebooks)
- [Experimental Design](#experimental-design)
- [Results Summary](#results-summary)
- [Project Management](#project-management)

---

## Project Overview

Type 2 diabetes affects over **537 million adults worldwide** and accounts for more than 90% of all diabetes cases. Despite being largely preventable, it remains critically underdiagnosed, particularly in low-to-middle income and marginalised communities, leading to severe downstream complications including cardiovascular disease, chronic kidney disease, and retinopathy.

This project investigates the following core question:

> *How effectively can we predict Type 2 diabetes using medical and lifestyle data, and which algorithms remain the most accurate and reliable across different datasets representing different populations?*

Rather than optimising a single model on a single dataset, this study takes a **comparative approach**, benchmarking three state-of-the-art algorithms across three datasets that span clinical, demographic, and population health domains. The emphasis is on understanding the **tradeoffs** between approaches and how well models generalise beyond the population they were trained on.

---

## Team Members

| Name | Student ID | Primary Responsibilities |
|---|---|---|
| Tamika Andrews | 816039421 | EDA & Preprocessing (Diabetes Prediction Dataset), LightGBM experiments |
| Dylan Corbett | 816039487 | EDA & Preprocessing (BRFSS), XGBoost experiments |
| Kieron Seepersad | 816041436 | EDA & Preprocessing (Pima), Logistic Regression experiments |

---

## Research Questions

1. **Does dataset size affect model generalisability?**
   Compare performance across small (Pima, 768 rows), medium (Diabetes Prediction, 100,000 rows), and large (BRFSS, 253,680 rows) datasets.

2. **Which features are most predictive - clinical or lifestyle?**
   Clinical datasets contain lab values (glucose, HbA1c, insulin). The BRFSS dataset contains survey-based lifestyle indicators (exercise, diet, income). Feature importance analysis across datasets reveals which factors matter most.

3. **Which algorithm performs best under different data conditions?**
   LightGBM, XGBoost, and Logistic Regression are each expected to have relative strengths depending on dataset size, feature type, and class imbalance.

---

## Datasets

All datasets are sourced from [Kaggle](https://www.kaggle.com/).

| Dataset | Source | Size | Key Features | Lens |
|---|---|---|---|---|
| **Diabetes Prediction Dataset** | [iammustafatz](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) | 100,000 rows, 9 features | HbA1c, blood glucose, BMI, age, hypertension, heart disease, smoking history, gender | Modern, large-scale clinical lab values |
| **Pima Indians Diabetes** | [mathchi](https://www.kaggle.com/datasets/mathchi/diabetes-data-set) | 768 rows, 8 features | Pregnancies, insulin, BMI, glucose, blood pressure, skin thickness, age, diabetes pedigree function | Classic benchmark; females only, smaller dataset with missing values |
| **Diabetes Health Indicators (CDC BRFSS 2015)** | [alexteboul](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) | 253,680 rows, 22 features | BMI, general health, physical activity, diet, cholesterol, blood pressure, income, education, healthcare access | Population health survey; largest dataset, real-world, class imbalanced (~85/15) |

---

## Algorithms

| Algorithm | Justification |
|---|---|
| **Logistic Regression** | Interpretable baseline. Provides clear feature coefficients that can be explained to clinical stakeholders. Establishes a lower-bound for performance. |
| **LightGBM** | Gradient boosting optimised for speed and efficiency. Handles categorical features natively. Expected to excel on the large BRFSS dataset due to histogram-based binning. |
| **XGBoost** | Industry-standard gradient boosting. Robust to outliers and handles missing values natively — critical for the Pima dataset's missing insulin and skin thickness values. |

---

## Repository Structure

```
Predicting-Type-2-Diabetes/
│
├── data/
│   ├── raw/                                          # Original, unmodified source datasets
│   │   ├── diabetes.csv                              # Pima Indians Diabetes dataset
│   │   ├── diabetes_prediction_dataset.csv           # Diabetes Prediction dataset (raw CSV)
│   │   └── diabetes_binary_health_indicators_BRFSS2015.csv  # CDC BRFSS 2015 dataset
│   │
│   └── preprocessed/                                 # Cleaned, split, and scaled data ready for modelling
│       ├── diabetes_prediction_dataset_preprocessed/
│       │   ├── X_train_scaled.parquet                # Training features (scaled)
│       │   ├── X_train_scaled_smote.parquet          # Training features (SMOTE-balanced)
│       │   ├── X_test_scaled.parquet                 # Test features (scaled)
│       │   ├── y_train.parquet                       # Training labels
│       │   ├── y_train_smote.parquet                 # Training labels (SMOTE-balanced)
│       │   └── y_test.parquet                        # Test labels
│       │
│       ├── pima_preprocessed/
│       │   ├── pima_X_train_scaled.csv               # Training features (scaled)
│       │   ├── pima_X_test_scaled.csv                # Test features (scaled)
│       │   ├── pima_y_train.csv                      # Training labels
│       │   └── pima_y_test.csv                       # Test labels
│       │
│       └── brfss_preprocessed/
│           ├── brfss_train_original.parquet          # Training features (original, imbalanced)
│           ├── brfss_train_smote.parquet             # Training features (SMOTE-balanced)
│           └── brfss_test.parquet                    # Test features and labels
│
├── notebooks/
│   ├── preprocessing/                                # EDA and preprocessing pipelines
│   │   ├── diabetes_prediction_eda_preprocessing.ipynb
│   │   ├── pima_eda_preprocessing.ipynb
│   │   └── brfss_eda_preprocessing.ipynb
│   │
│   └── experiments/                                  # Modelling and evaluation notebooks
│       ├── dataset_2_diabetes_prediction_experiments.ipynb
│       ├── pima_experiments.ipynb
│       └── brfss_eda_experiments.ipynb
│
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/COMP3608-Intelligent-Systems-Project/Predicting-Type-2-Diabetes.git
cd Predicting-Type-2-Diabetes
```

2. **Install dependencies:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost imbalanced-learn pyarrow
```

> **Note:** All notebooks load data using paths relative to the project root (`Predicting-Type-2-Diabetes/`). Ensure your working directory is set to the project root before running any notebook, or run the following at the top of any notebook cell:
> ```python
> import os
> os.chdir('/path/to/Predicting-Type-2-Diabetes')
> ```

### Running the Notebooks

It is recommended to run the notebooks in the following order:

**Step 1 - Preprocessing (run first to generate the preprocessed data files):**

| Order | Notebook | Dataset |
|---|---|---|
| 1 | `notebooks/preprocessing/pima_eda_preprocessing.ipynb` | Pima Indians |
| 2 | `notebooks/preprocessing/diabetes_prediction_eda_preprocessing.ipynb` | Diabetes Prediction |
| 3 | `notebooks/preprocessing/brfss_eda_preprocessing.ipynb` | CDC BRFSS |

**Step 2 - Experiments (run after preprocessing):**

| Order | Notebook | Dataset |
|---|---|---|
| 4 | `notebooks/experiments/pima_experiments.ipynb` | Pima Indians |
| 5 | `notebooks/experiments/dataset_2_diabetes_prediction_experiments.ipynb` | Diabetes Prediction |
| 6 | `notebooks/experiments/brfss_eda_experiments.ipynb` | CDC BRFSS |

> The preprocessed data files are already included in the repository under `data/preprocessed/`, so the experiment notebooks can be run directly without re-running the preprocessing step.

---

## Experimental Design

Each experiment notebook follows a consistent four-experiment pipeline:

| Experiment | Description |
|---|---|
| **Experiment 1 - Baseline Comparison** | Train all three algorithms on raw, unbalanced data. Establish baseline performance using Accuracy, Precision, Recall, F1-Score, and AUC-ROC. |
| **Experiment 2 - Class Imbalance Handling** | Compare three strategies — no resampling, SMOTE, and class weighting — to determine which yields the best F1-Score and Recall on each dataset. |
| **Experiment 3 - Feature Importance Analysis** | Extract feature importance from tuned LightGBM and XGBoost models. Determine whether clinical features (HbA1c, glucose) consistently outrank lifestyle features (physical activity, diet) across datasets. Includes univariate analysis and K-Fold stability testing. |
| **Experiment 4 - Cross-Dataset Generalisation** | Train on one dataset and test on the remaining two without retraining. Assess how well models transfer across different populations and feature spaces. |

**Performance Metrics:**

| Metric | Rationale |
|---|---|
| Accuracy | Useful baseline; can be misleading on imbalanced datasets |
| Precision | Minimises false alarms and unnecessary clinical follow-ups |
| Recall (Sensitivity) | **Primary healthcare concern** - missing a positive diagnosis can delay treatment |
| F1-Score | **Primary comparison metric** - balances Precision and Recall; important for imbalanced datasets |
| AUC-ROC | Threshold-independent evaluation; allows algorithm comparison without fixing a decision boundary |

---

## Results Summary

*(Results are populated after running the experiment notebooks. The tables below reflect the structure of the output produced by each experiment notebook. Additionally, visuals such as heatmaps, bar graphs, and confussion matrices are included)*

**Experiment 1 - Baseline F1-Score by Dataset:**

| Algorithm | Pima Indians | Diabetes Prediction | CDC BRFSS |
|---|---|---|---|
| Logistic Regression | — | — | — |
| LightGBM | — | — | — |
| XGBoost | — | — | — |

**Experiment 2 - Best Imbalance Strategy per Dataset:**

| Dataset | Best Strategy | Best Algorithm | F1-Score |
|---|---|---|---|
| Pima Indians | — | — | — |
| Diabetes Prediction | — | — | — |
| CDC BRFSS | — | — | — |

---

## Project Management

- **GitHub Repository:** [https://github.com/COMP3608-Intelligent-Systems-Project/Predicting-Type-2-Diabetes](https://github.com/COMP3608-Intelligent-Systems-Project/Predicting-Type-2-Diabetes)
- **Trello Board:** [https://trello.com/invite/b/69bbabe642db587a4b2f596c/ATTI2667b282c5821e3b505d5e2b42aba23d1D218289/comp3608project](https://trello.com/invite/b/69bbabe642db587a4b2f596c/ATTI2667b282c5821e3b505d5e2b42aba23d1D218289/comp3608project)

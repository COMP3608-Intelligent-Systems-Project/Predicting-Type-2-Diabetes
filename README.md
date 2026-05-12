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
```

> **File-format note:** Some preprocessed outputs are stored as Parquet to reduce file size and improve loading speed. The notebooks use `pyarrow` where Parquet support is required.

---

## Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost imbalanced-learn pyarrow
```

### Clone the Repository

```bash
git clone https://github.com/COMP3608-Intelligent-Systems-Project/Predicting-Type-2-Diabetes.git
cd Predicting-Type-2-Diabetes
```

### Google Colab Setup

Most notebooks assume the working directory is the repository root. In Colab, run:

```python
import os

if not os.path.exists("/content/Predicting-Type-2-Diabetes"):
    !git clone https://github.com/COMP3608-Intelligent-Systems-Project/Predicting-Type-2-Diabetes.git

%cd /content/Predicting-Type-2-Diabetes
```

---

## Notebook Run Order

The preprocessed files are already included, so experiment notebooks can be run directly. To reproduce the full workflow from scratch, run preprocessing first.

### 1. Preprocessing Notebooks

| Order | Notebook | Dataset | Purpose |
|---:|---|---|---|
| 1 | `notebooks/preprocessing/pima_eda_preprocessing.ipynb` | Pima | EDA, missing-value handling, scaling |
| 2 | `notebooks/preprocessing/diabetes_prediction_eda_preprocessing.ipynb` | Diabetes Prediction | EDA, encoding, scaling, SMOTE |
| 3 | `notebooks/preprocessing/brfss_eda_preprocessing.ipynb` | CDC BRFSS | EDA, stratified split, SMOTE, Parquet outputs |

### 2. Experiment Notebooks

| Order | Notebook | Dataset | Purpose |
|---:|---|---|---|
| 4 | `notebooks/experiments/pima_experiments.ipynb` | Pima | Experiments 1–4, including top-k feature analysis |
| 5 | `notebooks/experiments/dataset_2_diabetes_prediction_experiments.ipynb` | Diabetes Prediction | Experiments 1–4, including top-3 vs all-feature testing |
| 6 | `notebooks/experiments/brfss_eda_experiments.ipynb` | CDC BRFSS | Experiments 1–4 plus Experiment 3B top-k comparison |

---

## Experimental Design

### Experiment 1 — Baseline Algorithm Comparison

Train the three algorithms on the raw imbalanced dataset:

- Logistic Regression
- LightGBM
- XGBoost

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

### Experiment 2 — Class Imbalance Handling

Compare three imbalance strategies:

- No resampling
- SMOTE
- Class weighting

The main goal is to improve **Recall** while monitoring the effect on **F1-score** and **Precision**.

### Experiment 3 — Feature Importance Analysis

Analyze which predictors the models rely on most.

Dataset-specific methods include:

- Tree-based feature importance from LightGBM and XGBoost
- Logistic Regression coefficient analysis where relevant
- Univariate feature power analysis for Dataset 2
- K-Fold stability testing for Dataset 2
- Top k-feature comparison analysis 

### Experiment 4 — Cross-Dataset Generalisation

Test whether models trained on one dataset can transfer to another dataset without retraining.

Because the three datasets do not share the same full feature set, cross-dataset testing is necessarily limited. The notebooks use dataset-specific transfer strategies, including:

- shared common features such as age and BMI
- feature alignment/mapping where appropriate
- comparison of internal performance against external transfer performance

---

## Results Summary

### Experiment 1 — Baseline Performance

| Dataset | Best Baseline Model | Best Baseline F1 | Main Observation |
|---|---|---:|---|
| Pima Indians | XGBoost | 0.6346 | Boosted trees performed best on the small clinical benchmark |
| Diabetes Prediction | LightGBM | 0.8085 | Strong clinical features produced high internal performance |
| CDC BRFSS | LightGBM | 0.2536 | Baseline F1 was low due to imbalance and lack of direct glucose/HbA1c features |

### Experiment 2 — Imbalance Handling

| Dataset | Best F1 Result | F1 | Recall | Main Observation |
|---|---|---:|---:|---|
| Pima Indians | XGBoost after SMOTE | 0.6429 | 0.6667 | SMOTE improved the balance between precision and recall |
| Diabetes Prediction | XGBoost with class weighting | 0.6417 | 0.8827 | Class weighting greatly increased recall but lowered precision |
| CDC BRFSS | SMOTE + Logistic Regression | 0.4429 | 0.7614 | Imbalance handling strongly improved diabetic-case detection |

> For CDC BRFSS, **Class Weighting + LightGBM** achieved the highest recall of approximately **0.7915**, but SMOTE + Logistic Regression produced the highest F1-score.

### Experiment 3 / 3B — Feature Importance and Top-K Features

| Dataset | Important Features | Top-K Observation |
|---|---|---|
| Pima Indians | Glucose, BMI, Age, Diabetes Pedigree Function | Top 3 features retained most performance, but all features gave the best XGBoost F1 |
| Diabetes Prediction | Blood glucose, HbA1c, age | Top 3 clinical features preserved high recall and AUC, showing strong clinical signal |
| CDC BRFSS | General health, high blood pressure, age, BMI, high cholesterol, smoking/activity indicators | Top 5 features improved recall/F1 for tuned tree models, though all features kept slightly higher AUC |

### Top-K vs All-Features Examples

| Dataset | Model | All-Feature F1 | Top-K F1 | Interpretation |
|---|---|---:|---:|---|
| Pima | XGBoost | 0.6429 | 0.6182 | Top 3 features kept most performance |
| Diabetes Prediction | LightGBM | 0.6437 | 0.5559 | Top 3 features raised recall but reduced precision/F1 |
| CDC BRFSS | XGBoost | 0.3195 | 0.4051 | Top 5 features improved F1 by focusing on stronger survey indicators |

### Experiment 4 — Cross-Dataset Generalisation

Cross-dataset performance was generally weaker than internal testing.

Key observations:

- Models trained and tested on the same dataset performed better than models transferred to another dataset.
- Transfer performance was limited because the datasets do not share the same full feature set.
- Shared features such as **age** and **BMI** were not enough to fully capture diabetes risk across populations.
- The CDC notebook's combined transfer test found the strongest transfer direction as **Pima → CDC BRFSS using Logistic Regression**, with F1 ≈ **0.3542** and Recall ≈ **0.7829**.

---

## Key Findings

1. **Dataset 2 performed best overall.**  
   The Diabetes Prediction dataset achieved the strongest internal results because it includes direct clinical predictors such as blood glucose and HbA1c.

2. **CDC BRFSS was the hardest dataset.**  
   It was large and realistic, but strongly imbalanced and based on survey indicators rather than lab measurements.

3. **Accuracy alone was misleading.**  
   Imbalanced datasets could produce high accuracy while still missing many positive diabetic cases.

4. **Recall and F1-score were more appropriate metrics.**  
   In screening contexts, false negatives are especially concerning because missed diabetic cases may delay treatment.

5. **Top predictors matched real-world diabetes risk factors.**  
   Blood glucose and HbA1c dominated in Dataset 2. Glucose and BMI were important in Pima. General health, blood pressure, age, BMI, and cholesterol were important in CDC BRFSS.

6. **Top-k feature subsets were useful but not always superior.**  
   Smaller feature sets improved interpretability and sometimes recall, but full feature sets usually provided more complete information.

7. **Cross-dataset generalisation was weak.**  
   Models did not transfer strongly across populations without retraining or recalibration.

---

## Limitations

- The three datasets use different feature spaces.
- CDC BRFSS does not include direct clinical lab measurements such as glucose or HbA1c.
- Pima is small and population-specific.
- Dataset 2 contains strong clinical predictors that are not available in the other two datasets.
- Cross-dataset testing required simplified feature alignment, which limits direct comparability.
- SMOTE improves minority-class learning but can increase false positives.
- These models should support screening and analysis, not replace clinical diagnosis.

---

## References and Context

- American Diabetes Association — Standards of Care in Diabetes and diagnostic guidance for A1C/glucose-based diagnosis.
- CDC — Type 2 diabetes risk factors including age, overweight/obesity, physical inactivity, and related health conditions.
- Kaggle datasets:
  - Pima Indians Diabetes Dataset
  - Diabetes Prediction Dataset
  - CDC BRFSS 2015 Diabetes Health Indicators

---

## Project Management

- **GitHub Repository:** <https://github.com/COMP3608-Intelligent-Systems-Project/Predicting-Type-2-Diabetes>
- **Trello Board:** <https://trello.com/invite/b/69bbabe642db587a4b2f596c/ATTI2667b282c5821e3b505d5e2b42aba23d1D218289/comp3608project>

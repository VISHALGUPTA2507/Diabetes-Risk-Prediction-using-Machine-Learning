# 🩺 Diabetes Risk Prediction

An end-to-end machine learning project that predicts a patient's **diabetes risk level** as **Low, Moderate, or High** using clinical, lifestyle, and demographic features.

The project covers exploratory data analysis, preprocessing, clinical feature engineering, multi-model comparison, cross-validation, hyperparameter tuning, ensemble experiments, model persistence, and a live **Streamlit** web application.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-189A3E)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project builds a complete multiclass machine learning pipeline using the **Diabetes Risk Prediction dataset** and deploys the final trained model through an interactive Streamlit application.

### Pipeline

- Exploratory Data Analysis (EDA)
- Missing-value handling
- Duplicate checking
- Clinical feature engineering
- Categorical encoding
- Train/test splitting
- Feature scaling
- Multi-model comparison
- 5-fold stratified cross-validation
- Hyperparameter optimization with `RandomizedSearchCV`
- Stacking ensemble experiments
- Final model selection using Macro F1
- Model persistence with `joblib`
- Streamlit frontend for real-time predictions

---

## 🗂️ Dataset

**Dataset:** `diabetes_risk.csv`  
**Source:** Kaggle  
**Records:** 15,000 patients  
**Original features:** 18  
**Duplicate rows:** None

### Target Variable

The target column is `diabetes_risk`.

| Risk Level | Samples | Percentage |
|---|---:|---:|
| Low | 9,000 | 60% |
| Moderate | 3,750 | 25% |
| High | 2,250 | 15% |
| **Total** | **15,000** | **100%** |

The target is imbalanced, so **Macro F1-score** was used as the primary model-selection metric.

### Features

- `age`
- `bmi`
- `hours_sleep_per_night`
- `stress_level`
- `fasting_blood_sugar`
- `hba1c_level`
- `blood_pressure_systolic`
- `blood_pressure_diastolic`
- `waist_circumference_cm`
- `gender`
- `city`
- `family_history_diabetes`
- `physical_activity_level`
- `diet_type`
- `smoking_status`
- `alcohol_consumption`
- `income_bracket`
- `diabetes_risk`

---

## 🧹 Data Cleaning

| Feature | Missing Values |
|---|---:|
| `smoking_status` | 472 |
| `alcohol_consumption` | 3,788 |
| `income_bracket` | 463 |

Missing categorical values were replaced with `unknown`.

No duplicate rows were found.

The `patient_id` column was removed before model training because it is an identifier rather than a meaningful predictive feature.

---

# 📊 Exploratory Data Analysis

## Target Class Distribution

![Target class distribution](images/01_target_class_distribution.png)

## Numeric Feature Distributions

![Numeric histograms](images/02_numeric_histograms.png)

## Categorical Feature Distributions

### Gender
![Gender](images/03_cat_gender.png)

### City
![City](images/04_cat_city.png)

### Family History of Diabetes
![Family history](images/05_cat_family_history_diabetes.png)

### Physical Activity Level
![Physical activity](images/06_cat_physical_activity_level.png)

### Diet Type
![Diet type](images/07_cat_diet_type.png)

### Smoking Status
![Smoking status](images/08_cat_smoking_status.png)

### Alcohol Consumption
![Alcohol consumption](images/09_cat_alcohol_consumption.png)

### Income Bracket
![Income bracket](images/10_cat_income_bracket.png)

### Diabetes Risk
![Diabetes risk](images/11_cat_diabetes_risk.png)

## Correlation Heatmap

Important relationships included:

- `fasting_blood_sugar` ↔ `hba1c_level`: **0.91**
- `bmi` ↔ `waist_circumference_cm`: **0.94**
- `age` ↔ `blood_pressure_systolic`: **0.48**

![Correlation heatmap](images/12_correlation_heatmap.png)

---

# ⚙️ Feature Engineering & Preprocessing

### BMI Category

Created using clinical BMI ranges:

```text
Underweight
Healthy
Overweight
Obesity Class 1
Obesity Class 2
Obesity Class 3
```

### HbA1c Category

```text
Normal
Prediabetes
Diabetes
```

### Blood Pressure Category

Systolic and diastolic blood pressure were combined into a clinical category ranging from low blood pressure to hypertensive crisis.

### Train/Test Split

```text
Training: 80%
Testing: 20%
random_state: 42
```

Stratification was used to preserve class proportions.

`StandardScaler` was used for scale-sensitive models such as Logistic Regression, SVM, and KNN. Tree-based models were trained without scaling.

---

# 🤖 Model Training & Comparison

The following models were evaluated:

1. Logistic Regression
2. Naive Bayes
3. Decision Tree
4. SVM
5. KNN
6. Random Forest
7. Extra Trees
8. Bagging
9. AdaBoost
10. Gradient Boosting
11. Hist Gradient Boosting
12. XGBoost

## 5-Fold Stratified Cross-Validation

| Model | CV Mean F1 | CV Std |
|---|---:|---:|
| Logistic Regression | 0.7394 | 0.0074 |
| Gradient Boosting | 0.7367 | 0.0065 |
| Random Forest | 0.7354 | 0.0051 |
| XGBoost | 0.7322 | 0.0070 |
| SVM | 0.7304 | 0.0043 |
| Bagging | 0.7303 | 0.0082 |
| Hist Gradient Boosting | 0.7275 | 0.0083 |
| AdaBoost | 0.7229 | 0.0086 |
| Extra Trees | 0.7024 | 0.0120 |
| Naive Bayes | 0.6847 | 0.0116 |
| Decision Tree | 0.6826 | 0.0117 |
| KNN | 0.6346 | 0.0087 |

Logistic Regression achieved the highest initial cross-validation Macro F1 of **0.7394**.

---

# 🧪 Test-Set Evaluation

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.8037 | 0.7753 | 0.7457 | 0.7591 |
| Logistic Regression | 0.8023 | 0.7666 | 0.7466 | 0.7559 |
| Gradient Boosting | 0.8017 | 0.7678 | 0.7419 | 0.7538 |

Random Forest produced the highest initial test-set Macro F1.

## Random Forest Confusion Matrix

![Random Forest confusion matrix](images/13_rf_confusion_matrix.png)

---

# 🎯 Hyperparameter Tuning

Random Forest was optimized using `RandomizedSearchCV` with Macro F1 as the scoring metric.

### Best Parameters

```text
n_estimators = 150
max_depth = 20
min_samples_split = 10
min_samples_leaf = 1
max_features = log2
class_weight = balanced_subsample
```

### Best Cross-Validation Score

```text
Macro F1 = 0.7402
```

## Tuned Random Forest Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.7993 |
| Precision (Macro) | 0.7649 |
| Recall (Macro) | 0.7621 |
| **F1 (Macro)** | **0.7627** |

![Tuned Random Forest confusion matrix](images/14_tuned_rf_confusion_matrix.png)

---

# 🧩 Stacking Ensemble Experiments

Stacking was tested using Logistic Regression, Random Forest, Gradient Boosting, and XGBoost as base learners.

| Ensemble | Accuracy | Macro F1 |
|---|---:|---:|
| Stacking + Logistic Regression | **0.8060** | 0.7593 |
| Stacking + Random Forest | 0.8010 | 0.7543 |

Although stacking with Logistic Regression achieved the highest accuracy (**80.60%**), the tuned Random Forest achieved the higher Macro F1 (**76.27%**). Since the dataset is imbalanced, Macro F1 was prioritized for final model selection.

---

# 🏆 Final Model

## Tuned Random Forest Classifier

### Final Performance

```text
Accuracy        : 79.93%
Macro Precision : 76.49%
Macro Recall    : 76.21%
Macro F1        : 76.27%
```

The tuned Random Forest was selected because it provided the strongest balance across the three risk classes according to the project's primary metric, Macro F1.

---

# 💾 Saved Artifacts

| File | Purpose |
|---|---|
| `diabetes_risk_model.pkl` | Final tuned Random Forest model |
| `diabetes_risk_features.pkl` | Feature names and expected feature order |

---

# 🖥️ Streamlit Web App

The Streamlit frontend accepts:

- Age
- Gender
- BMI
- Hours of sleep
- Stress level
- Fasting blood sugar
- HbA1c level
- Systolic blood pressure
- Diastolic blood pressure
- Waist circumference
- Family history of diabetes
- Physical activity
- Diet type
- Smoking status
- Alcohol consumption
- Income bracket
- City

### Prediction Pipeline

```text
User Input
    ↓
Feature Engineering
    ↓
Categorical Encoding
    ↓
Feature Alignment
    ↓
Tuned Random Forest
    ↓
Risk Prediction
    ↓
Class Probabilities
```

The application displays Low, Moderate, or High risk along with class probabilities.

---

# 🚀 Getting Started

## Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost streamlit joblib
```

## Run the Streamlit App

```bash
streamlit run app.py
```

## Explore the Notebook

Open:

```text
diabeties-prediction.ipynb
```

to view the complete EDA → preprocessing → training → evaluation → tuning → model export workflow.

---

# 📁 Repository Structure

```text
Diabetes-Risk-Prediction/
│
├── diabeties-prediction.ipynb
├── app.py
├── diabetes_risk.csv
├── diabetes_risk_model.pkl
├── diabetes_risk_features.pkl
├── README.md
│
└── images/
    ├── 01_target_class_distribution.png
    ├── 02_numeric_histograms.png
    ├── 03_cat_gender.png
    ├── 04_cat_city.png
    ├── 05_cat_family_history_diabetes.png
    ├── 06_cat_physical_activity_level.png
    ├── 07_cat_diet_type.png
    ├── 08_cat_smoking_status.png
    ├── 09_cat_alcohol_consumption.png
    ├── 10_cat_income_bracket.png
    ├── 11_cat_diabetes_risk.png
    ├── 12_correlation_heatmap.png
    ├── 13_rf_confusion_matrix.png
    └── 14_tuned_rf_confusion_matrix.png
```

---

# 🧠 Machine Learning Concepts Demonstrated

- Exploratory Data Analysis
- Data Cleaning
- Missing Value Handling
- Categorical Encoding
- Feature Engineering
- Feature Scaling
- Multiclass Classification
- Class Imbalance
- Stratified Cross-Validation
- Macro F1 Evaluation
- Confusion Matrix Analysis
- Random Forest
- Gradient Boosting
- XGBoost
- Ensemble Learning
- Hyperparameter Optimization
- `RandomizedSearchCV`
- Model Serialization
- Streamlit Application Development

---

# 🔍 Implementation Notes

For scale-sensitive models, the scaler should be fitted only on training data:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The test set should not be independently fitted with `fit_transform()` because that introduces data leakage.

For a production-ready version, preprocessing and the model should ideally be stored together in a `Pipeline` or `ColumnTransformer` so training and inference always use the same transformations.

---

# 🔮 Future Improvements

- Complete preprocessing `Pipeline`
- `ColumnTransformer`
- SHAP model explainability
- Feature importance visualization
- Probability calibration
- Advanced class-imbalance handling
- Prediction history
- Improved Streamlit UI/UX
- FastAPI REST API
- Cloud deployment
- Automated testing
- Model monitoring and versioning

---

# ⚠️ Disclaimer

This project is intended for **educational and machine learning demonstration purposes only**.

It is **not a certified medical diagnostic system** and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

# 👤 Author

**Potta Vishalgupta**

[GitHub](https://github.com/VISHALGUPTA2507) · [LinkedIn](https://linkedin.com/in/vishalgupta-potta-414723329)

---
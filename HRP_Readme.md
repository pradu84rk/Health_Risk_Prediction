# Predictive Health Outcome Analytics

## Impact of Junk Food vs Healthy Food Consumption on 10-Year Health Risk

A machine learning classification project that analyzes dietary, lifestyle, demographic, and health-related factors in teenagers and predicts their **Health Risk Category** over a 10-year period.

The project follows a complete data science workflow:

**EDA → Data Cleaning → Feature Engineering → Data Encoding → Class Imbalance Handling → Model Training → Evaluation → Model Comparison**

---

## 1. Project Objective

The main objective is to use machine learning to classify teenagers into three health-risk categories:

- **Low**
- **Moderate**
- **High**

The analysis focuses on factors such as:

- Junk food and fast-food consumption
- Sugary drink intake
- Healthy food consumption
- BMI and body composition
- Physical activity
- Sleep
- Stress
- Smoking and alcohol consumption
- Family medical history
- Blood pressure
- Glucose and HbA1c
- Cholesterol and triglycerides
- Food accessibility
- Screen time

The project is intended as a predictive analytics and educational data science project based on the provided dataset.

---

## 2. Dataset

The dataset contains **5,200 teenager records** and initially contains **53 columns**.

### Target Variable

`Health_Risk_Category`

| Category | Encoded Value |
|---|---:|
| Low | 0 |
| Moderate | 1 |
| High | 2 |

### Target Distribution

| Health Risk | Records |
|---|---:|
| Low | 1,409 |
| Moderate | 3,428 |
| High | 363 |

The target is therefore imbalanced, with the **High-risk class being the smallest class**.

---

## 3. Project Structure

The project has been separated into three notebooks so that each stage of the workflow is easier to understand and maintain.

```text
Capstone-1/
│
├── Capstone_1_EDA.ipynb
├── Capstone_1_Feature_Engineering.ipynb
├── Capstone_1_Model.ipynb
│
├── cleaned_health_dataset1.csv
├── Model_comparison.csv
├── confusion_matrix.csv
├── predictions.csv
└── feature_importance.csv
```

### Notebook 1 — EDA

`Capstone_1_EDA.ipynb`

Contains:

- Dataset loading
- Dataset shape and structure
- Data types
- Summary statistics
- Categorical variable analysis
- Missing-value analysis
- Duplicate-value check
- Unique-value checks
- Text cleaning
- Outlier inspection
- Domain-based value checks
- Target distribution
- Numerical feature analysis
- Correlation analysis
- Feature vs target analysis

Output:

```text
cleaned_health_dataset1.csv
```

### Notebook 2 — Feature Engineering

`Capstone_1_Feature_Engineering.ipynb`

Contains:

- Blood pressure decomposition
- Blood pressure category creation
- BMI category creation
- Distribution and skewness checks
- Target encoding
- Ordinal encoding
- Binary encoding
- One-hot encoding

Output:

```text
feature_engineered_health_dataset.csv
```

### Notebook 3 — Model Building

`Capstone_1_Model.ipynb`

Contains:

- Feature/target separation
- Correlation check
- Stratified train-test split
- Class imbalance handling using SMOTE
- Feature scaling for Logistic Regression
- Logistic Regression
- Random Forest
- Random Forest hyperparameter tuning
- XGBoost
- Cross-validation
- Confusion matrices
- Classification reports
- Model comparison
- Result export

---

## 4. Data Cleaning

### Missing Values

Missing categorical values were handled using meaningful categories rather than removing rows.

Examples:

- `Medication_Use` → `No Medication`
- `Existing_Conditions` → `No Known Condition`
- `Alcohol_Consumption` → `Non-Drinker`
- `Whole_Grain_Intake` → `Unknown`

### Duplicate Records

Duplicate records were checked and removed.

### Text Cleaning

Extra spaces in categorical variables were removed before further processing.

### Outlier Treatment

Outliers were first inspected using boxplots and the IQR method.

Only values considered potentially unrealistic or impossible were constrained. Examples include:

- Height: 130–220 cm
- Weight: 25–180 kg
- Water intake: 0.5–12 L
- Sleep: 3–15 hours
- Exercise frequency: 0–7 days/week
- Screen time: 0–24 hours/day
- Fruit servings: 0–10/day
- Vegetable servings: 0–10/day
- Saturated fat: 0–90 g
- Protein: 0–200 g

Potentially meaningful extreme values were not automatically removed.

---

## 5. Feature Engineering

### Blood Pressure

The original `Blood_Pressure` column contains values such as:

```text
120/80
```

It was separated into:

```text
Systolic_BP
Diastolic_BP
```

A new categorical feature was then created:

```text
BP_Category
```

with:

- Normal
- Elevated
- High

### BMI Category

BMI was converted into categories:

| BMI Range | Category |
|---|---|
| < 18.5 | Underweight |
| 18.5–25 | Normal |
| 25–30 | Overweight |
| > 30 | Obese |

### Encoding

Different encoding techniques were used depending on the type of variable.

**Ordinal encoding** was used for variables with a natural order, including:

- Education Level
- Income Level
- Processed Food Intake
- Whole Grain Intake
- Stress Level
- Alcohol Consumption
- BP Category
- BMI Category

**Binary encoding** was used for:

- Gender
- Family History of Diabetes
- Family History of Heart Disease

**One-hot encoding** was used for nominal variables such as:

- Ethnicity
- Occupation
- Protein Source
- Smoking Status
- Existing Conditions
- Medication Use
- Urban/Rural

---

## 6. Feature Selection

The following columns were excluded from the model input:

```text
Person_ID
Health_Risk_Category
10_Year_Obesity_Risk_Percent
10_Year_Diabetes_Risk_Percent
10_Year_CVD_Risk_Percent
```

The three 10-year risk percentage columns were excluded from the model features because they are directly related to the health-risk prediction target and could introduce target leakage.

---

## 7. Train-Test Split

A stratified train-test split was used:

```text
Training set: 80%
Testing set: 20%
random_state: 42
```

Stratification was used to preserve the class distribution across training and testing data.

---

## 8. Handling Class Imbalance

The dataset contains substantially fewer High-risk records than Moderate-risk records.

**SMOTE (Synthetic Minority Over-sampling Technique)** was therefore applied to the training data.

Important:

> SMOTE was applied only to the training set, not to the test set.

This prevents synthetic samples from influencing the final test evaluation.

---

## 9. Machine Learning Models

Three classification algorithms were evaluated.

### Logistic Regression

Used as an interpretable baseline model.

Feature scaling was applied using:

```text
StandardScaler
```

### Random Forest

A tree-based ensemble model was trained initially and then tuned using:

```text
GridSearchCV
```

The tuning process optimized for:

```text
F1 Macro
```

### XGBoost

An XGBoost multiclass classifier was also trained with controlled parameters including:

- Number of estimators
- Maximum depth
- Learning rate
- Subsampling
- Column sampling
- Regularization

---

## 10. Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-average metrics
- Confusion matrix
- Classification report

Special attention was given to **High-risk recall**, because failing to identify an actual High-risk case is important in this predictive health-risk use case.

---

## 11. Model Results

The test-set comparison from the project is:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8885 | 0.8401 | 0.8716 | 0.8545 |
| XGBoost | 0.8654 | 0.7966 | 0.8475 | 0.8189 |
| Random Forest | 0.8500 | 0.7461 | 0.8293 | 0.7955 |

The project selected **Logistic Regression as the final predictive model** based on the evaluation results documented in the original notebook.

### Logistic Regression — Class-wise Results

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Low | 0.85 | 0.84 | 0.85 |
| Moderate | 0.92 | 0.91 | 0.91 |
| High | 0.75 | 0.86 | 0.80 |

Overall:

```text
Accuracy  = 88.85%
Macro Recall = 87.16%
Macro F1 = 85.45%
```

---

## 12. Cross-Validation

5-fold cross-validation was performed.

### Logistic Regression

```text
Mean Macro F1 ≈ 0.7234
```

### XGBoost

```text
Mean Macro F1 ≈ 0.8230
```

Cross-validation was used as an additional check of model performance across multiple training splits.

---

## 13. Output Files

The model notebook exports:

### Model Comparison

```text
Model_comparison.csv
```

Contains accuracy, precision, recall and F1-score for the evaluated models.

### Confusion Matrix

```text
confusion_matrix.csv
```

Contains the confusion matrix for the Logistic Regression predictions.

### Predictions

```text
predictions.csv
```

Contains:

```text
Actual
Predicted
```

### Feature Importance / Coefficients

```text
feature_importance.csv
```

Contains the mean absolute Logistic Regression coefficients used to inspect feature influence.

---

## 14. Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- Imbalanced-learn / SMOTE

### Models

- Logistic Regression
- Random Forest
- XGBoost

---

## 15. Workflow

```text
Raw Dataset
     │
     ▼
EDA
     │
     ├── Missing Value Handling
     ├── Duplicate Check
     ├── Outlier Analysis
     ├── Data Validation
     └── Exploratory Analysis
     │
     ▼
Feature Engineering
     │
     ├── BP Features
     ├── BMI Category
     ├── Ordinal Encoding
     ├── Binary Encoding
     └── One-Hot Encoding
     │
     ▼
Feature Selection
     │
     ▼
Stratified Train-Test Split
     │
     ▼
SMOTE on Training Data
     │
     ├───────────────┬───────────────┐
     ▼               ▼               ▼
Logistic         Random Forest    XGBoost
Regression       + Tuning
     │               │               │
     └───────────────┴───────────────┘
                     │
                     ▼
              Model Evaluation
                     │
                     ▼
             Model Comparison
                     │
                     ▼
             Final Prediction
```

---

## 16. Key Project Takeaways

- The dataset represents a **multiclass health-risk classification problem**.
- The target classes are imbalanced, particularly the High-risk class.
- Data cleaning and domain-based validation were performed before modeling.
- Blood pressure and BMI were transformed into additional categorical features.
- Different encoding strategies were used based on feature type.
- SMOTE was applied to the training set to address class imbalance.
- Three machine learning models were compared.
- Logistic Regression achieved the highest test accuracy and macro F1 among the models reported in the final comparison.
- Confusion matrices and class-wise recall were used to understand model behavior beyond accuracy.
- The project demonstrates an end-to-end machine learning workflow from raw data to model evaluation.

---

## 17. Limitations

- The dataset is used for predictive analytics and should not be interpreted as a clinical diagnostic system.
- The data contains synthetic/structured health and lifestyle information as provided by the project dataset.
- Model performance depends on the quality and representativeness of the dataset.
- Predictions should not be treated as medical advice or individual clinical risk assessments.

---

## 18. Future Improvements

Possible extensions include:

- More extensive hyperparameter optimization
- Feature selection and dimensionality reduction
- Explainable AI using SHAP
- Calibration of predicted probabilities
- Testing additional ensemble models
- External validation using an independent dataset
- Development of an interactive prediction application
- Integration with the Power BI dashboard for business-style health-risk analytics

---

## Author

**Pradusha R**

**Background:** Food Technology | Data Science & Analytics

**Project:** Predictive Health Outcome Analytics

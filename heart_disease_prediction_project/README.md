# Heart Disease Prediction using Logistic Regression

## Task Objective
The goal of this project is to build a **Logistic Regression** model that predicts whether a patient is likely to have heart disease based on clinical and demographic features.  
We also evaluate the model using multiple performance metrics and identify the most important features influencing the prediction.


## Dataset Used
- **Source:** `heart_disease_uci_cleaned.csv`
- **Description:** A cleaned version of the UCI Heart Disease dataset containing patient information such as age, sex, chest pain type, blood pressure, cholesterol levels, and other diagnostic results.
- **Target Variable:** `num` (converted to binary `target`: 1 = Heart disease present, 0 = No heart disease)


##  Models Applied
- **Logistic Regression**
  - Binary classification (heart disease vs. no heart disease)
  - One-hot encoding applied to categorical variables
  - Train-test split (80% train, 20% test)
  - Model saved using `joblib` for future use


## Key Results and Findings
1. **Model Performance:**
   - **Accuracy:** `0.7989`  
   - **Confusion Matrix:**
     ```
     [[60 15]
      [22 87]]
     ```
   - **Classification Report:**
     ```
                   precision    recall  f1-score   support

               0       0.73      0.80      0.76        75
               1       0.85      0.80      0.82       109

         accuracy                           0.80       184
        macro avg       0.79      0.80      0.79       184
     weighted avg       0.80      0.80      0.80       184
     ```
   - **ROC-AUC Score:** `0.8872`
   - **ROC Curve:** Plot generated showing good separation between positive and negative classes.

2. **Top 10 Features Affecting Prediction (absolute importance):**
   | Feature                      | Coefficient | Absolute Importance |
   |------------------------------|-------------|---------------------|
   | cp_atypical angina           | -1.7474     | 1.7474              |
   | sex_Male                     |  1.3591     | 1.3591              |
   | cp_non-anginal               | -1.2792     | 1.2792              |
   | cp_typical angina            | -1.1935     | 1.1935              |
   | exang                        |  0.9125     | 0.9125              |
   | ca                           |  0.7837     | 0.7837              |
   | oldpeak                      |  0.5886     | 0.5886              |
   | thal_normal                  | -0.5074     | 0.5074              |
   | slope_flat                   |  0.5046     | 0.5046              |
   | restecg_st-t abnormality     |  0.4872     | 0.4872              |

   **Interpretation:**
   - Positive coefficients → increase likelihood of heart disease (e.g., being male, exercise-induced angina, certain slope values).
   - Negative coefficients → decrease likelihood (e.g., certain chest pain types, normal thalassemia results).


## Files in this Repository
- `Trained_model.py` – Python script containing data loading, preprocessing, training, evaluation, and feature importance extraction.
- `heart_disease_uci_cleaned.csv` – Dataset used in the project after cleaning.
- `logistic_regression_model.pkl` – Saved trained Logistic Regression model.
- `README.md` – This documentation.

---

# Credit Risk Modeling Project

A machine learning project to predict loan default probability and calculate credit scores for loan applicants.

## Overview

This project builds a credit risk model using customer, loan, and bureau data. The model predicts whether a customer will default on their loan and assigns them a credit score (300-900) with a rating (Poor/Average/Good/Excellent).

## Dataset

The project uses three CSV files:
- **customers.csv** - Customer demographic and financial information
- **loans.csv** - Loan details and disbursement info
- **bureau_data.csv** - Credit bureau data (accounts, delinquencies, etc.)

Total records: 50,000 customer loans

## Project Structure

```
├── app/
│   ├── main.py                    # Streamlit web app
│   ├── prediction_helper.py       # Prediction logic
│   └── artifacts/
│       └── model_data.joblib      # Trained model + preprocessing objects
├── dataset/                       # Raw CSV files
├── credit_risk_modelling_project.ipynb  # Main analysis notebook
└── WOE_IV.xlsx                   # Weight of Evidence analysis
```

## Key Features

### Data Processing
- Merged three datasets on customer ID
- Handled missing values and duplicates
- Removed outliers using business rules
- Fixed data quality issues (typos in categorical variables)

### Feature Engineering
- **Loan-to-Income Ratio** - measures loan affordability
- **Delinquency Ratio** - percentage of months with late payments
- **Avg DPD per Delinquency** - average days past due when delinquent

### Modeling Approach
Tried multiple approaches:
1. Logistic Regression, Random Forest, XGBoost (baseline)
2. With undersampling for class imbalance
3. With SMOTE-Tomek for class imbalance
4. Hyperparameter tuning using Optuna

**Final Model:** Logistic Regression with SMOTE-Tomek
- Used Weight of Evidence (WOE) and Information Value (IV) for feature selection
- Handled class imbalance (10:1 ratio of non-default to default)
- Tuned hyperparameters for best F1 score

### Model Performance
- **Accuracy:** 93%
- **Precision (Default):** 57%
- **Recall (Default):** 94%
- **F1-Score (Default):** 71%
- **ROC-AUC:** Good separation between classes

## How to Run

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit joblib imbalanced-learn optuna
```

### Run the Notebook
```bash
jupyter notebook credit_risk_modelling_project.ipynb
```

### Run the Web App
```bash
cd app
streamlit run main.py
```

The app will open in your browser where you can input customer details and get instant credit risk predictions.

## Techniques Used

- **EDA:** Box plots, histograms, KDE plots, correlation heatmaps
- **Feature Selection:** VIF for multicollinearity, Information Value (IV) analysis
- **Sampling:** SMOTE-Tomek for handling imbalanced data
- **Model Tuning:** RandomizedSearchCV, Optuna
- **Evaluation:** Classification report, ROC-AUC, decile analysis

## Web App Features

Input customer information:
- Age, income, loan amount
- Loan tenure and purpose
- Credit history (DPD, delinquency ratio, utilization)
- Residence type

Get instant results:
- Default probability
- Credit score (300-900)
- Rating category

## Notes

- The model prioritizes recall for defaults (catching risky borrowers) over precision
- Credit scores are scaled from probability: higher scores = lower default risk
- Feature engineering was crucial - derived features showed stronger predictive power than raw ones

## Future Improvements

- Add more sophisticated feature interactions
- Try ensemble methods (stacking, blending)
- Implement model monitoring and drift detection
- Add explainability (SHAP values) to understand predictions
- Improve UI with better visualizations

---

Built as a machine learning project


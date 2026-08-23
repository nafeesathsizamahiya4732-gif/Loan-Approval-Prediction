# Loan Approval Prediction

## Project Overview

This project predicts whether a loan application will be approved or rejected using machine learning classification models.

The project includes data preprocessing, exploratory data analysis, feature selection, model training, model evaluation, and loan prediction.

## Dataset

The dataset contains information about loan applicants, including:

- Gender
- Marital Status
- Number of Dependents
- Education
- Self Employment
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status

## Data Preprocessing

The following preprocessing steps were performed:

1. Checked the dataset structure.
2. Checked for missing values and duplicate rows.
3. Filled missing categorical values using the mode.
4. Filled missing numerical values using the median.
5. Converted categorical values into numerical values.
6. Converted the property area using one-hot encoding.
7. Converted the loan status into numerical values.

## Exploratory Data Analysis

The project includes visualizations for:

- Loan Approval Status Distribution
- Education vs Loan Status
- Credit History vs Loan Status

The graphs are stored in the `images` folder.

## Machine Learning Models

The following classification models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- Support Vector Machine (SVM)

The models were compared using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The best model was selected based on the F1-Score.

## Model

The selected best model is saved as:

`model/best_loan_model.pkl`

The name of the selected model is stored in:

`model/best_model.txt`

## Loan Prediction

The project also tests the selected model using an example loan applicant and predicts whether the loan is:

- APPROVED
- REJECTED

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

## How to Run

Install the required packages using:

`pip install -r requirements.txt`

Then run:

`main.py`


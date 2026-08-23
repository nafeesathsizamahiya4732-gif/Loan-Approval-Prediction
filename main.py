import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Create folders for project files
os.makedirs("images", exist_ok=True)
os.makedirs("model", exist_ok=True)


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/loan_data.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print("Number of rows and columns:", df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", df.duplicated().sum())


# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================

# Fill missing categorical values with mode
categorical_columns = [
    "gender",
    "married",
    "dependents",
    "self_employed"
]

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Fill missing numerical values with median
numerical_columns = [
    "loanamount",
    "loan_amount_term",
    "credit_history"
]

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

# Convert categorical values into numerical values
df["gender"] = df["gender"].map({"male": 1, "female": 0})
df["married"] = df["married"].map({"yes": 1, "no": 0})
df["education"] = df["education"].map({
    "graduate": 1,
    "not graduate": 0
})
df["self_employed"] = df["self_employed"].map({
    "yes": 1,
    "no": 0
})

# Convert dependents
df["dependents"] = df["dependents"].replace({"3+": 3})
df["dependents"] = pd.to_numeric(df["dependents"])

# Convert property area using one-hot encoding
df = pd.get_dummies(
    df,
    columns=["property_area"],
    drop_first=True,
    dtype=int
)

# Convert target variable
df["loan_status"] = df["loan_status"].map({
    "y": 1,
    "n": 0
})

print("\n========== PREPROCESSING COMPLETE ==========")
print(df.head())

print("\n========== MISSING VALUES AFTER PREPROCESSING ==========")
print(df.isnull().sum())


# ==========================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# Loan Status Distribution
plt.figure(figsize=(6, 4))
df["loan_status"].value_counts().plot(kind="bar")
plt.title("Loan Approval Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig("images/loan_status_distribution.png")
plt.show()
plt.close()

# Education vs Loan Status
plt.figure(figsize=(6, 4))
pd.crosstab(
    df["education"],
    df["loan_status"]
).plot(
    kind="bar",
    figsize=(6, 4)
)
plt.title("Education vs Loan Status")
plt.xlabel("Education")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig("images/education_vs_loan_status.png")
plt.show()
plt.close()

# Credit History vs Loan Status
plt.figure(figsize=(6, 4))
pd.crosstab(
    df["credit_history"],
    df["loan_status"]
).plot(
    kind="bar",
    figsize=(6, 4)
)
plt.title("Credit History vs Loan Status")
plt.xlabel("Credit History")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig("images/credit_history_vs_loan_status.png")
plt.show()
plt.close()

print("\n========== EDA COMPLETE ==========")


# ==========================================
# STEP 4: FEATURE SELECTION
# ==========================================

features = [
    "gender",
    "married",
    "dependents",
    "education",
    "self_employed",
    "applicantincome",
    "coapplicantincome",
    "loanamount",
    "loan_amount_term",
    "credit_history",
    "property_area_semiurban",
    "property_area_urban"
]

X = df[features]
y = df["loan_status"]

print("\n========== FEATURE SELECTION COMPLETE ==========")
print("Features used:")
print(X.columns)

print("\n========== FEATURE SHAPE ==========")
print("X shape:", X.shape)

print("\n========== TARGET SHAPE ==========")
print("y shape:", y.shape)


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# STEP 5: MODEL SELECTION AND TRAINING
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC()
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name}: Trained")

print("\n========== MODEL TRAINING COMPLETE ==========")


# ==========================================
# STEP 6: MODEL EVALUATION
# ==========================================

results = {}

for name, model in trained_models.items():

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

    print(f"\n========== {name.upper()} ==========")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    print("Confusion Matrix:")
    print(cm)

print("\n========== MODEL EVALUATION COMPLETE ==========")


# ==========================================
# STEP 7: LOAN PREDICTION
# ==========================================

# Select the best model based on F1-Score
best_model_name = max(
    results,
    key=lambda model: results[model]["F1-Score"]
)

best_model = trained_models[best_model_name]

print("\n========== BEST MODEL ==========")
print("Best Model:", best_model_name)

# Example new applicant
new_applicant = pd.DataFrame({
    "gender": [1],
    "married": [1],
    "dependents": [0],
    "education": [1],
    "self_employed": [0],
    "applicantincome": [5000],
    "coapplicantincome": [2000],
    "loanamount": [150],
    "loan_amount_term": [360],
    "credit_history": [1.0],
    "property_area_semiurban": [0],
    "property_area_urban": [1]
})

prediction = best_model.predict(new_applicant)[0]

print("\n========== LOAN PREDICTION ==========")

if prediction == 1:
    print("Loan Status: APPROVED")
else:
    print("Loan Status: REJECTED")


# ==========================================
# STEP 8: SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "model/best_loan_model.pkl"
)

with open("model/best_model.txt", "w") as file:
    file.write(best_model_name)

print("\n========== BEST MODEL SAVED ==========")
print("Best model:", best_model_name)
print("File: model/best_loan_model.pkl")
print("Model name saved in: model/best_model.txt")


# ==========================================
# PROJECT COMPLETE
# ==========================================

print("\n========== PROJECT COMPLETE ==========")
print("Loan Approval Prediction System completed successfully.")
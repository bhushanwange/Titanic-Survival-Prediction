import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==========================================
# Step 1: Load Titanic data & clean
# ==========================================
file_name = "titanic.csv"

if not os.path.exists(file_name):
    raise FileNotFoundError(f"'{file_name}' was not found in {os.getcwd()}.")

df = pd.read_csv(file_name)
df.columns = df.columns.str.strip()

print("Columns found and mapped successfully!")

# Map your specific CSV column names to standard working names
df = df.rename(
    columns={
        "Siblings/Spouses Aboard": "SibSp",
        "Parents/Children Aboard": "Parch",
    }
)

# Select relevant features and target variable
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
target = "Survived"

df = df[features + [target]].copy()

# Handle missing values: Fill Age and Fare with median
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# ==========================================
# Step 2: Encode categorical variables (Sex)
# ==========================================
df = pd.get_dummies(df, columns=["Sex"], drop_first=True)

X = df.drop(columns=[target])
y = df[target]

# ==========================================
# Step 3: Single-Split Baseline Evaluation
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Using a Pipeline ensures scaling is correctly applied per fold without data leakage
base_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])

base_pipeline.fit(X_train, y_train)
y_pred = base_pipeline.predict(X_test)
single_split_f1 = f1_score(y_test, y_pred)

print("\n--- Single-Split Model Performance ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {single_split_f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# Step 4: 5-Fold Cross-Validation Enhancement
# ==========================================
cv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])

# Calculate F1 score across 5 folds
cv_scores = cross_val_score(cv_pipeline, X, y, cv=5, scoring='f1')
mean_cv_f1 = cv_scores.mean()

print("--- 5-Fold Cross-Validation Performance ---")
print(f"F1 Scores across folds: {[round(score, 4) for score in cv_scores]}")
print(f"Mean 5-Fold CV F1 Score: {mean_cv_f1:.4f}\n")

# ==========================================
# Step 5: Performance Comparison & Documentation
# ==========================================
print("--- Comparison & Summary ---")
print(f"Single-Split F1 Score : {single_split_f1:.4f}")
print(f"Mean 5-Fold CV F1 Score: {mean_cv_f1:.4f}")
print(
    "Documentation Insight: Cross-validation gives a more reliable, generalized "
    "performance estimate across multiple data partitions, reducing the risk of "
    "overfitting to a single random train-test split."
)

# ==========================================
# Step 6: Plot ROC curve (using test split)
# ==========================================
y_prob = base_pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr, tpr, color="blue", label=f"Logistic Regression (AUC = {roc_auc:.2f})"
)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title(
    "Receiver Operating Characteristic (ROC) Curve - Titanic (CV Enhanced)", fontsize=14
)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.show()
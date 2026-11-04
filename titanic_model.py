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
from sklearn.model_selection import train_test_split
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
# Notice we use 'Siblings/Spouses Aboard' and 'Parents/Children Aboard'
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
# Step 3: Split data and fit Logistic Regression
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# ==========================================
# Step 4: Assess performance metrics
# ==========================================
y_pred = model.predict(X_test_scaled)

print("\n--- Model Performance Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# Step 5: Plot ROC curve
# ==========================================
y_prob = model.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

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
    "Receiver Operating Characteristic (ROC) Curve - Titanic", fontsize=14
)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.show()
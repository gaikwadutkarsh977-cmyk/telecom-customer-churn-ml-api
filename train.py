import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# 1. Load Dataset
# ==========================
df = pd.read_csv("data.csv")

print("Columns in dataset:")
print(df.columns)

# ==========================
# 2. Target Variable
# ==========================
# We will use Churn Value (already 0/1)
y = df["Churn Value"]

# ==========================
# 3. Drop Unnecessary Columns
# ==========================
drop_columns = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Reason"
]

df = df.drop(columns=drop_columns)

# ==========================
# 4. Features
# ==========================
X = df.drop("Churn Value", axis=1).copy()

# Convert Total Charges to numeric
X.loc[:, "Total Charges"] = pd.to_numeric(X["Total Charges"], errors="coerce")

# Fill missing values
X.fillna(0, inplace=True)

# ==========================
# 5. Encode Categorical Data
# ==========================
X = pd.get_dummies(X)

# ==========================
# 6. Train Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# 7. Scaling
# ==========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# 8. Model Training
# ==========================
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# ==========================
# 9. Evaluation
# ==========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================
# 10. Save Model + Scaler + Feature Columns
# ==========================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
print("\nModel, Scaler and Features saved successfully!")
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load saved objects
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")

@app.get("/")
def home():
    return {"message": "Churn API Running 🚀"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input JSON to DataFrame
        df = pd.DataFrame([data])

        # Convert Total Charges to numeric
        if "Total Charges" in df.columns:
            df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

        # One-hot encode
        df = pd.get_dummies(df)

        # Add missing columns
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Ensure correct column order
        df = df[feature_columns]

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)

        return {
            "prediction": int(prediction[0]),
            "meaning": "Customer Will Churn" if prediction[0] == 1 else "Customer Will Stay"
        }

    except Exception as e:
        return {"error": str(e)}
# 🚀 Telecom Customer Churn Prediction API

## 📌 Project Overview
This project predicts whether a telecom customer will churn (leave the company) using Machine Learning.
The trained model is deployed using FastAPI.

---

## 🎯 Business Problem
Customer churn directly impacts revenue.
The goal is to predict which customers are likely to leave so the company can take preventive action.

---

## 🧠 Machine Learning Workflow
1. Data Cleaning
2. Feature Engineering
3. Encoding Categorical Variables
4. Feature Scaling
5. Logistic Regression Model Training
6. Model Evaluation
7. API Deployment using FastAPI

---

## ⚙️ Tech Stack
- Python
- Pandas
- Scikit-Learn
- FastAPI
- Uvicorn
- Git & GitHub

---

## 📊 Model Performance
Accuracy: ~80%

The model performs well in identifying churn customers with balanced precision and recall.

---

## 🚀 API Endpoint

### POST /predict

### Sample Input JSON

```json
{
  "Gender": "Male",
  "Senior Citizen": 0,
  "Tenure Months": 5,
  "Monthly Charges": 75.5
}

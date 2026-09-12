# model/predict.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import shap
from schema import FEATURES

with open("model/credit_model.pkl", "rb") as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer(model)

def score(features: dict):
    df = pd.DataFrame([features])[FEATURES]
    pred_score = int(model.predict(df)[0])

    if pred_score >= 670:
        risk = "Low"
        max_loan = 500000
    elif pred_score >= 500:
        risk = "Medium"
        max_loan = 200000
    else:
        risk = "High"
        max_loan = 50000

    shap_values = explainer.shap_values(df)[0]
    explanation = dict(zip(FEATURES, shap_values.tolist()))

    return {
        "score": pred_score,
        "risk": risk,
        "max_loan": max_loan,
        "explanation": explanation,
    }

if __name__ == "__main__":
    test_input = {
        "daily_revenue": 45000,
        "restock_frequency": 4,
        "restock_amount": 60000,
        "pos_sales_consistency": 0.8,
        "supplier_payment_delay": 3,
        "revenue_volatility": 0.3,
        "months_operating": 24,
    }
    print(score(test_input))
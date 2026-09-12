import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from schema import FEATURES, FEATURE_RANGES, TARGET

def normalize(x, lo, hi, invert=False):
    v = (x - lo) / (hi - lo)
    return 1 - v if invert else v

def generate_dataset(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    data = {}

    for feat, (lo, hi) in FEATURE_RANGES.items():
        data[feat] = rng.uniform(lo, hi, n)

    df = pd.DataFrame(data)

    # Reliability index: weighted sum of normalized features (0-1)
    # Direction: higher consistency/revenue/months = better, higher volatility/delay/restock_frequency = worse
    reliability = (
        0.30 * normalize(df["pos_sales_consistency"], *FEATURE_RANGES["pos_sales_consistency"]) +
        0.20 * normalize(df["revenue_volatility"], *FEATURE_RANGES["revenue_volatility"], invert=True) +
        0.15 * normalize(df["supplier_payment_delay"], *FEATURE_RANGES["supplier_payment_delay"], invert=True) +
        0.15 * normalize(df["restock_frequency"], *FEATURE_RANGES["restock_frequency"], invert=True) +
        0.10 * normalize(df["daily_revenue"], *FEATURE_RANGES["daily_revenue"]) +
        0.10 * normalize(df["months_operating"], *FEATURE_RANGES["months_operating"])
    )

    noise = rng.normal(0, 0.04, n)  # keeps model from fitting a perfect linear formula
    score = 300 + (reliability + noise) * 550
    df[TARGET] = np.clip(score, 300, 850).round().astype(int)

    return df[FEATURES + [TARGET]]

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/synthetic_dataset.csv", index=False)
    print(df.describe())
    df = pd.read_csv("data/synthetic_dataset.csv")
    print(df.corr()["credit_score"])
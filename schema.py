FEATURES = [
    "daily_revenue",
    "restock_frequency",
    "restock_amount",
    "pos_sales_consistency",
    "supplier_payment_delay",
    "revenue_volatility",
    "months_operating",
] 

TARGET = "credit_score"  # 300-850

FEATURE_RANGES = {
    "daily_revenue": (2000, 150000),       # naira/day
    "restock_frequency": (1, 21),          # days between restocks
    "restock_amount": (5000, 300000),      # naira
    "pos_sales_consistency": (0.0, 1.0),   # fraction of days with a recorded sale
    "supplier_payment_delay": (0, 30),     # avg days late
    "revenue_volatility": (0.05, 1.5),     # coefficient of variation
    "months_operating": (1, 240),          # months
}
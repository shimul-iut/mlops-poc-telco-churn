import pandas as pd
import os

RAW_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUTPUT_PATH = "data/telco_churn.csv"

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

def prepare():
    # Make sure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    print(f"Downloading data from {RAW_URL}...")
    df = pd.read_csv(RAW_URL)
    
    # Minimal cleaning — keep it simple, this isn't the point of the POC
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    # Simple label-encode the categoricals so scikit-learn can consume them directly
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category").cat.codes
        
    keep_cols = CATEGORICAL_COLS + [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn",
    ]
    df = df[keep_cols]
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare()

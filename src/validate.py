import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.metrics import accuracy_score

DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/model.joblib"

EXPECTED_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "SeniorCitizen", "tenure",
    "MonthlyCharges", "TotalCharges", "Churn"
]

def validate():
    print("Starting validation checks...")
    
    # 1. Check data file existence
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(DATA_PATH)
    
    # 2. Schema validation
    actual_cols = list(df.columns)
    if len(actual_cols) != len(EXPECTED_COLS) or not all(col in actual_cols for col in EXPECTED_COLS):
        print(f"ERROR: Column schema mismatch. Expected: {EXPECTED_COLS}, Got: {actual_cols}")
        sys.exit(1)
        
    # 3. Data quality check (no missing values)
    if df.isnull().values.any():
        print("ERROR: Processed data contains null/missing values.")
        print(df.isnull().sum())
        sys.exit(1)
        
    # 4. Target values check
    unique_churn = df["Churn"].unique()
    if not all(val in [0, 1] for val in unique_churn):
        print(f"ERROR: Target Churn contains invalid values: {unique_churn}")
        sys.exit(1)
        
    # 5. Feature ranges/types check
    # tenure must be non-negative
    if (df["tenure"] < 0).any():
        print("ERROR: Feature 'tenure' contains negative values.")
        sys.exit(1)
    # MonthlyCharges must be positive
    if (df["MonthlyCharges"] < 0).any():
        print("ERROR: Feature 'MonthlyCharges' contains negative values.")
        sys.exit(1)
        
    print("Data quality checks passed successfully!")
    
    # 6. Model performance and existence check
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1)
        
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Failed to load model from {MODEL_PATH}: {e}")
        sys.exit(1)
        
    # Evaluate model accuracy on the full data as a quick sanity check
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"Validation accuracy on full dataset: {accuracy:.4f}")
    if accuracy < 0.75:
        print(f"ERROR: Model accuracy {accuracy:.4f} is below the threshold of 0.75")
        sys.exit(1)
        
    print("Model validation checks passed successfully!")
    print("All validation checks passed!")
    sys.exit(0)

if __name__ == "__main__":
    validate()

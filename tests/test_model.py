import os
import pandas as pd
import numpy as np
import joblib
import pytest
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

@pytest.fixture
def dataset():
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Data file {DATA_PATH} not found. Run prepare_data.py first.")
    return pd.read_csv(DATA_PATH)

@pytest.fixture
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file {MODEL_PATH} not found. Run train.py first.")
    return joblib.load(MODEL_PATH)

def test_data_schema(dataset):
    """Verify columns match schema, have data, and row count is reasonable"""
    assert len(dataset) > 0, "Dataset is empty"
    actual_cols = list(dataset.columns)
    assert len(actual_cols) == len(EXPECTED_COLS), "Column count mismatch"
    assert all(col in actual_cols for col in EXPECTED_COLS), "Missing columns"
    # Ensure targets are correct
    assert set(dataset["Churn"].unique()).issubset({0, 1}), "Churn must only be 0 or 1"

def test_no_missing_values(dataset):
    """Verify there are no missing values in the processed data"""
    assert not dataset.isnull().values.any(), "Missing values found in the dataset"

def test_model_loads(model):
    """Verify model can be loaded and has the required sklearn api methods"""
    assert hasattr(model, "predict"), "Model does not have 'predict' method"
    assert hasattr(model, "predict_proba"), "Model does not have 'predict_proba' method"

def test_model_accuracy_threshold(dataset, model):
    """Verify model accuracy on the full dataset exceeds 0.75"""
    X = dataset.drop(columns=["Churn"])
    y = dataset["Churn"]
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    assert accuracy >= 0.75, f"Model accuracy {accuracy:.4f} is less than threshold 0.75"

def test_prediction_shape(dataset, model):
    """Verify prediction with a single input row succeeds and returns correct shape"""
    X = dataset.drop(columns=["Churn"]).iloc[[0]]
    prediction = model.predict(X)
    proba = model.predict_proba(X)
    
    assert len(prediction) == 1, "Prediction count is not 1"
    assert prediction[0] in [0, 1], "Prediction is not binary"
    assert proba.shape == (1, 2), "Probability shape is not (1, 2)"
    assert np.isclose(np.sum(proba[0]), 1.0), "Probabilities do not sum to 1.0"

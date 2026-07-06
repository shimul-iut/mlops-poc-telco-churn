import os
import json
import sys
import joblib
import pandas as pd

MODEL_PATH = "models/model.joblib"

# Load model on cold start
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully on cold start.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print(f"Warning: Model not found at {MODEL_PATH}")
    model = None

FEATURE_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "SeniorCitizen", "tenure",
    "MonthlyCharges", "TotalCharges"
]

# Sensible default values for any missing features in the input payload
DEFAULT_INPUTS = {
    "gender": 0,
    "Partner": 0,
    "Dependents": 0,
    "PhoneService": 0,
    "MultipleLines": 0,
    "InternetService": 0,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "Contract": 0,
    "PaperlessBilling": 0,
    "PaymentMethod": 0,
    "SeniorCitizen": 0,
    "tenure": 0.0,
    "MonthlyCharges": 0.0,
    "TotalCharges": 0.0
}

def handler(event, context):
    global model
    
    # Lazy load model if it wasn't loaded during cold start
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
            except Exception as e:
                return {
                    "statusCode": 500,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"error": f"Model failed to load: {str(e)}"})
                }
        else:
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Model file not found"})
            }

    print(f"Received event: {json.dumps(event)}")
    
    # Determine the payload body
    body = None
    if isinstance(event, dict):
        if "body" in event:
            # Lambda proxy integration via API Gateway / Function URL
            try:
                body = json.loads(event["body"])
            except Exception as e:
                return {
                    "statusCode": 400,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"error": f"Invalid JSON in body: {str(e)}"})
                }
        else:
            # Direct invocation where event is already the dictionary payload
            body = event
    else:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Invalid event structure, must be a JSON object"})
        }

    # Extract features, filling in defaults for any missing features
    input_data = {}
    for col in FEATURE_COLS:
        val = body.get(col, DEFAULT_INPUTS[col])
        # Convert numeric fields to correct types
        if col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
            try:
                input_data[col] = float(val)
            except (ValueError, TypeError):
                return {
                    "statusCode": 400,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"error": f"Invalid numeric value for field '{col}': {val}"})
                }
        else:
            # Categorical are encoded as integers
            try:
                input_data[col] = int(val)
            except (ValueError, TypeError):
                return {
                    "statusCode": 400,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"error": f"Invalid integer code for categorical field '{col}': {val}"})
                }
                
    # Create DataFrame with exact column order
    df = pd.DataFrame([input_data])[FEATURE_COLS]
    
    # Run prediction
    try:
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])
        
        response = {
            "prediction": prediction,
            "probability": round(probability, 4)
        }
        
        print(f"Prediction response: {json.dumps(response)}")
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response)
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Prediction failed: {str(e)}"})
        }

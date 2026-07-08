import boto3
import requests
import json
import sys
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

# Deployed AWS Lambda Function URL
URL = "https://wzeani3emrzvvz54tbk3jcdgsa0fqdyc.lambda-url.us-east-1.on.aws/"
REGION = "us-east-1"
PROFILE = "mlops-poc"

# 10 diverse test payloads (0 to 9) representing various customer profiles with descriptions
PAYLOADS = [
    {
        "description": "High-risk, month-to-month, high charges, senior citizen customer",
        "data": {
            "Contract": 0, "tenure": 1, "MonthlyCharges": 89.9, "TotalCharges": 89.9,
            "InternetService": 1, "PaymentMethod": 2, "SeniorCitizen": 1
        }
    },
    {
        "description": "Low-risk, 2-year contract, long tenure, low charges",
        "data": {
            "Contract": 2, "tenure": 72, "MonthlyCharges": 20.15, "TotalCharges": 1450.8,
            "InternetService": 0, "PaymentMethod": 0, "SeniorCitizen": 0
        }
    },
    {
        "description": "Medium-risk, 1-year contract, mid tenure, mid charges",
        "data": {
            "Contract": 1, "tenure": 36, "MonthlyCharges": 55.5, "TotalCharges": 1998.0,
            "InternetService": 1, "PaymentMethod": 1, "SeniorCitizen": 0
        }
    },
    {
        "description": "High-risk, Month-to-month, new customer, mid charges",
        "data": {
            "Contract": 0, "tenure": 3, "MonthlyCharges": 75.3, "TotalCharges": 225.9,
            "InternetService": 1, "PaymentMethod": 2, "SeniorCitizen": 0
        }
    },
    {
        "description": "Low-risk, 2-year contract, high tenure, very high charges",
        "data": {
            "Contract": 2, "tenure": 68, "MonthlyCharges": 115.0, "TotalCharges": 7820.0,
            "InternetService": 2, "PaymentMethod": 3, "SeniorCitizen": 1
        }
    },
    {
        "description": "High-risk, Month-to-month, long tenure, extremely high charges",
        "data": {
            "Contract": 0, "tenure": 48, "MonthlyCharges": 110.5, "TotalCharges": 5304.0,
            "InternetService": 2, "PaymentMethod": 2, "SeniorCitizen": 0
        }
    },
    {
        "description": "Low-risk, 1-year contract, long tenure, low charges",
        "data": {
            "Contract": 1, "tenure": 50, "MonthlyCharges": 25.0, "TotalCharges": 1250.0,
            "InternetService": 0, "PaymentMethod": 0, "SeniorCitizen": 0
        }
    },
    {
        "description": "High-risk, Month-to-month, brand new customer, low charges",
        "data": {
            "Contract": 0, "tenure": 0, "MonthlyCharges": 45.0, "TotalCharges": 45.0,
            "InternetService": 1, "PaymentMethod": 2, "SeniorCitizen": 0
        }
    },
    {
        "description": "Medium-risk, 2-year contract, short tenure, high charges",
        "data": {
            "Contract": 2, "tenure": 12, "MonthlyCharges": 95.0, "TotalCharges": 1140.0,
            "InternetService": 2, "PaymentMethod": 3, "SeniorCitizen": 0
        }
    },
    {
        "description": "Medium-risk, Month-to-month, long tenure, low charges",
        "data": {
            "Contract": 0, "tenure": 60, "MonthlyCharges": 19.5, "TotalCharges": 1170.0,
            "InternetService": 0, "PaymentMethod": 1, "SeniorCitizen": 0
        }
    }
]

# Read target payload ID from command line arguments
payload_id = 0
if len(sys.argv) > 1:
    try:
        val = int(sys.argv[1])
        if 0 <= val <= 9:
            payload_id = val
        else:
            print(f"Warning: Index {val} out of range (0-9). Defaulting to 0.")
    except ValueError:
        print("Warning: Invalid index argument. Defaulting to 0.")

# Extract corresponding payload and description
payload_info = PAYLOADS[payload_id]
payload_data = payload_info["data"]
payload_desc = payload_info["description"]
payload_str = json.dumps(payload_data)

# Print CLI invocation info
print("=" * 60)
print(f"RUNNING TEST INVOCATION (Payload ID: {payload_id})")
print(f"Description: {payload_desc}")
print("=" * 60)
print(f"Request Endpoint: {URL}")
print(f"Request Body:\n{json.dumps(payload_data, indent=4)}")
print("-" * 60)

try:
    # Set up session with the mlops-poc profile credentials
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    credentials = session.get_credentials()
    
    if not credentials:
        print(f"Error: Could not retrieve credentials for profile '{PROFILE}'. Run aws configure --profile {PROFILE} first.")
        sys.exit(1)

    # Build and sign the AWS SigV4 request
    request = AWSRequest(
        method="POST",
        url=URL,
        data=payload_str,
        headers={"Content-Type": "application/json"}
    )
    SigV4Auth(credentials, "lambda", REGION).add_auth(request)
    
    # Send the request
    response = requests.post(URL, data=payload_str, headers=dict(request.headers))
    
    print(f"Response Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response Body:\n{json.dumps(response.json(), indent=4)}")
    else:
        print(f"Response Raw Text:\n{response.text}")
    print("=" * 60)

except Exception as e:
    print(f"Failed to query endpoint: {e}")
    sys.exit(1)

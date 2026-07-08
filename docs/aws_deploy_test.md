# AWS Container Deployment & API Testing

This guide outlines how to build a production-ready container, push it to Amazon Elastic Container Registry (ECR), deploy it to AWS Lambda as a serving endpoint, and test the cloud-hosted prediction API.

---

## 1. Production Docker Build
AWS Lambda requires a clean, standard single-architecture image without build attestations or manifest lists.

### Step 1.1: Configure Inference-Only Requirements
Create a production requirements file named `requirements-predict.txt` containing only the packages required for inference to reduce container size and cold-start latency:
```text
numpy==1.26.4
pandas==2.2.2
scipy==1.11.4
scikit-learn==1.5.2
joblib==1.4.2
```
*Note: Pinned versions are configured with `manylinux_2_17` wheels so they install in AWS Lambda's minimal environment without compilation.*

### Step 1.2: Build the Container Image
Build the container with attestations disabled (`--provenance=false --sbom=false`):
```powershell
docker build --provenance=false --sbom=false -t telco-churn-predictor .
```

---

## 2. Amazon ECR Repository Setup & Push

### Step 2.1: Authenticate Docker with your ECR Registry
```powershell
aws ecr get-login-password --region us-east-1 --profile mlops-poc | docker login --username AWS --password-stdin 520687296884.dkr.ecr.us-east-1.amazonaws.com
```

### Step 2.2: Create ECR Repository & Apply Lifecycle Policy
Create the repository:
```powershell
aws ecr create-repository --repository-name telco-churn-predictor --region us-east-1 --profile mlops-poc
```
Set up a lifecycle policy using `ecr-lifecycle-policy.json` to expire older untagged images:
```powershell
aws ecr put-lifecycle-policy --repository-name telco-churn-predictor --lifecycle-policy-text file://ecr-lifecycle-policy.json --region us-east-1 --profile mlops-poc
```

### Step 2.3: Push Image to ECR
Tag and push the container:
```powershell
docker tag telco-churn-predictor:latest 520687296884.dkr.ecr.us-east-1.amazonaws.com/telco-churn-predictor:latest
docker push 520687296884.dkr.ecr.us-east-1.amazonaws.com/telco-churn-predictor:latest
```

---

## 3. IAM Role Setup for Lambda
Create an execution role allowing Lambda to run and publish logs to CloudWatch.

### Step 3.1: Create Trust Policy File (`trust-policy.json`)
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### Step 3.2: Create and Attach Policies
```powershell
# Create the IAM Role
aws iam create-role --role-name telco-churn-lambda-role --assume-role-policy-document file://trust-policy.json --profile mlops-poc

# Attach AWSLambdaBasicExecutionRole policy
aws iam attach-role-policy --role-name telco-churn-lambda-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole --profile mlops-poc
```

---

## 4. Deploying AWS Lambda Function

### Step 4.1: Create Lambda Function
Deploy the containerized Lambda function. Note that we allocate **`1024 MB`** of memory. Upgrading from 512 MB to 1024 MB allocates a full vCPU, speeding up cold-start package imports (`pandas`, `scikit-learn`) to prevent cold-start initialization timeouts:
```powershell
aws lambda create-function --function-name telco-churn-predictor --package-type Image --code ImageUri=520687296884.dkr.ecr.us-east-1.amazonaws.com/telco-churn-predictor:latest --role arn:aws:iam::520687296884:role/telco-churn-lambda-role --timeout 30 --memory-size 1024 --profile mlops-poc
```

### Step 4.2: Expose Lambda via Secure Function URL
Configure the Function URL with `AWS_IAM` authentication (SigV4 signing required) to secure the endpoint:
```powershell
# Create the configuration
aws lambda create-function-url-config --function-name telco-churn-predictor --auth-type AWS_IAM --profile mlops-poc

# Grant invoke permission
aws lambda add-permission --function-name telco-churn-predictor --action lambda:InvokeFunctionUrl --statement-id FunctionURLAllowPublicAccess --principal "*" --function-url-auth-type AWS_IAM --profile mlops-poc
```

---

## 5. Testing the Cloud-Hosted API

### Step 5.1: Direct Invocation via AWS CLI
Invoke the function directly with an escaped JSON string:
```powershell
aws lambda invoke --function-name telco-churn-predictor --payload '{\"MonthlyCharges\": 75.0, \"tenure\": 12, \"Contract\": 0}' --cli-binary-format raw-in-base64-out response.json --profile mlops-poc
```
Check the parsed output:
```powershell
Get-Content response.json
```

### Step 5.2: Updating Code / Model
To push changes (e.g. retraining updates) to Lambda:
```powershell
# Build and Push
docker build --provenance=false --sbom=false -t telco-churn-predictor .
docker tag telco-churn-predictor:latest 520687296884.dkr.ecr.us-east-1.amazonaws.com/telco-churn-predictor:latest
docker push 520687296884.dkr.ecr.us-east-1.amazonaws.com/telco-churn-predictor:latest

# Trigger Lambda update
aws lambda update-function-code --function-name telco-churn-predictor --image-uri 520687296884.dkr.ecr.us-east-1.amazonaws.com/telco-churn-predictor:latest --profile mlops-poc
```

### Step 5.3: Querying Endpoint with Pre-configured Payloads
You can use the helper script `src/test_endpoint.py` to send signed SigV4 requests. The script contains 10 distinct, pre-configured customer profiles (index 0 to 9) that can be selected using a command-line argument:

```powershell
python src/test_endpoint.py <payload_id>
```

#### Key Payload Profiles to Test:
* **Payload 0 (High Risk)**: Month-to-month contract, short tenure (1 month), high charges.
  ```powershell
  python src/test_endpoint.py 0
  ```
* **Payload 1 (Low Risk)**: Two-year contract, long tenure (72 months), low charges.
  ```powershell
  python src/test_endpoint.py 1
  ```
* **Payload 7 (Brand New Customer)**: Month-to-month contract, zero tenure.
  ```powershell
  python src/test_endpoint.py 7
  ```

The script will output the full request body sent to AWS, the response status code, and the returned prediction and probability output.


---

## 6. CloudWatch Alerting & Monitoring Setup (One-time)

To actively monitor the deployed serving endpoint, we configure metric filters and real-time email alerts when the system experiences failures.

### Step 6.1: Create Log Metric Filter
Create a CloudWatch Metric Filter to scan the Lambda function's log group (`/aws/lambda/telco-churn-predictor`) for the keyword `ERROR` (representing unhandled exceptions or severe runtime failures) and increment a metric:
```powershell
aws logs put-metric-filter --log-group-name /aws/lambda/telco-churn-predictor --filter-name ErrorCountFilter --filter-pattern "ERROR" --metric-transformations metricName=ErrorCount,metricNamespace=churn-predictor,metricValue=1 --profile mlops-poc
```

### Step 6.2: Create SNS Alert Topic
Create an Amazon Simple Notification Service (SNS) topic to process alert distributions:
```powershell
aws sns create-topic --name telco-churn-alerts --profile mlops-poc
```
*Note the returned topic ARN (e.g. `arn:aws:sns:us-east-1:520687296884:telco-churn-alerts`).*

### Step 6.3: Subscribe Email Address
Subscribe your email address to the SNS topic (replace `YOUR_EMAIL@example.com` with your target email address):
```powershell
aws sns subscribe --topic-arn arn:aws:sns:us-east-1:520687296884:telco-churn-alerts --protocol email --notification-endpoint YOUR_EMAIL@example.com --profile mlops-poc
```
*Note: AWS will send a confirmation email. Click the **Confirm Subscription** link in the message to activate the notification channel.*

### Step 6.4: Create Metric Alarm
Create a CloudWatch Alarm that triggers when `ErrorCount` is greater than or equal to `1` in any 5-minute evaluation period, routing the notification to the SNS topic:
```powershell
aws cloudwatch put-metric-alarm --alarm-name ChurnPredictorErrorAlarm --alarm-description "Alarm when Lambda predictor logs errors" --metric-name ErrorCount --namespace churn-predictor --statistic Sum --period 300 --evaluation-periods 1 --threshold 1 --comparison-operator GreaterThanOrEqualToThreshold --alarm-actions arn:aws:sns:us-east-1:520687296884:telco-churn-alerts --profile mlops-poc
```

### Step 6.5: Verify Alarm Execution
To test the alerting infrastructure, trigger a real unhandled exception inside the Lambda runtime by sending a malformed payload (e.g. a JSON object with `"body": "null"` which evaluates body to Python `None` and throws an unhandled `AttributeError` when accessed):
```powershell
aws lambda invoke --function-name telco-churn-predictor --payload '{\"body\": \"null\"}' --cli-binary-format raw-in-base64-out response.json --profile mlops-poc
```
Check the output to verify the failure:
```powershell
Get-Content response.json
```
*Expected CLI output includes `"FunctionError": "Unhandled"`. Within 1–3 minutes, the alarm will transition to the **ALARM** state in the AWS console and send an alert notification email.*


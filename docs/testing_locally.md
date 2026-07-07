# Local MLOps Loop Simulation & Verification

This guide outlines how to simulate the complete MLOps cycle locally: generating drifted data, detecting the drift, retraining the model, verifying metrics in MLflow, and deploying the updated model container to test the prediction API.

---

## 1. Simulating Data Drift
We simulate market or customer behavior changes using `src/generate_batch.py`. This script applies predefined drift profiles (e.g. shifts in charges or tenure) and saves the batch output to `data/current_batch.csv`.

To generate a batch with severe drift:
```powershell
python src/generate_batch.py --drift severe
```

---

## 2. Detecting Data Drift (Continuous Monitoring)
Run the drift detection script using **Evidently AI** to compare the baseline dataset against our new drifted batch:
```powershell
python src/monitor.py data/current_batch.csv
```

### Expected Output:
* The CLI output will print a feature-by-feature pass/fail log (features like `MonthlyCharges` and `tenure` will be flagged as `[DRIFTED]`).
* A detailed HTML report is generated at `reports/drift_report.html`. You can open this in your browser to inspect distributions visually.

---

## 3. Ingestion & Retraining (Continuous Training)
When drift is detected, we ingest the new data by appending it to our training dataset and retraining the model to adapt to the new pattern.

### Step 3.1: Append new data to training baseline
Run this Python one-liner in your terminal to merge the new batch into your baseline training file:
```powershell
python -c "import pandas as pd; df1=pd.read_csv('data/telco_churn.csv'); df2=pd.read_csv('data/current_batch.csv'); pd.concat([df1, df2]).to_csv('data/telco_churn.csv', index=False)"
```

### Step 3.2: Retrain the model
Run the training script to train a new `RandomForestClassifier` on the updated dataset:
```powershell
python src/train.py
```
*This overwrites `models/model.joblib` with the newly trained model and logs metrics to MLflow.*

---

## 4. Tracking and Comparing in MLflow
Every time `src/train.py` runs, it logs parameters, metrics, and model artifacts to your local MLflow instance.

### Step 4.1: Start the local MLflow server
Run the following command to spin up the UI:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Step 4.2: Inspect and compare runs
1. Open **[http://localhost:5000](http://localhost:5000)** in your browser.
2. Select the **`Telco-Churn-Experiment`** from the left panel.
3. Every retraining iteration generates a new run (e.g., `dashing-llama-412`).
4. Select multiple runs and click **Compare** to observe how metric scores (such as `accuracy` and `f1_score`) shift as you retrain on drifted data.

---

## 5. Local Container Redeployment & Testing (CD)
Once the model is retrained, we rebuild the container image to package the new model artifact and test the API.

### Step 5.1: Stop the current running container
If the Docker container is currently running in a terminal, press **`Ctrl + C`** to stop it.

### Step 5.2: Rebuild the Docker image
Run the build command to package the updated model (`models/model.joblib`):
```powershell
docker build -t telco-churn-predictor .
```

### Step 5.3: Run the updated container
```powershell
docker run -p 9000:8080 telco-churn-predictor
```

### Step 5.4: Verify the API response
Open a separate terminal window and invoke the local Lambda emulator endpoint:
```powershell
Invoke-RestMethod -Uri "http://localhost:9000/2015-03-31/functions/function/invocations" -Method Post -Body '{"MonthlyCharges": 175.0, "tenure": 4, "Contract": 0}' -ContentType "application/json"
```

Verify that you receive a `200 OK` status and a valid JSON response containing `prediction` and `probability` values corresponding to the new retrained model:
```powershell
statusCode headers                          body
---------- -------                          ----
       200 @{Content-Type=application/json} {"prediction": 0, "probability": 0.46}
```

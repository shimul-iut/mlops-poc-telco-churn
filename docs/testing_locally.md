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

### Step 4.3: Understanding Model Performance Metrics
During training, the pipeline outputs a `metrics.json` file. Here is what each performance metric means in the context of our Telco Churn model:

* **Accuracy (77.30%)**: 
  * *What it means*: The percentage of correct classifications (both churners and loyal customers) out of all tested customers.
  * *Context*: While 77.3% seems good, churn datasets are heavily imbalanced (most customers don't churn). High accuracy can be misleading if the model is simply predicting "No Churn" for everyone.
* **Precision (60.64%)**: 
  * *What it means*: Out of all customers the model *predicted* would churn, what percentage actually did?
  * *Context*: If the model flags 100 customers as "high-risk", 60.6% of them will actually churn, while 39.4% are "false alarms" (healthy customers targeted by retention campaigns unnecessarily).
* **Recall / Sensitivity (38.26%)**: 
  * *What it means*: Out of all customers who *actually* churned, what percentage did the model successfully identify?
  * *Context*: The model successfully caught 38.3% of the real churners but missed 61.7% of them (False Negatives). This indicates the model is conservative and needs tuning to catch more churners.
* **F1-Score (46.91%)**: 
  * *What it means*: The harmonic mean of Precision and Recall. It balances both metrics.
  * *Context*: Since our recall is quite low (38.3%) compared to precision (60.6%), our F1-score sits at 46.9%, showing that the model's overall prediction balance has room for improvement.
* **ROC AUC (79.48%)**: 
  * *What it means*: Area Under the Receiver Operating Characteristic curve. Measures the model's ability to separate churners from non-churners across all possible probability thresholds.
  * *Context*: A score of 79.5% is considered good (50% is random guessing, 100% is perfect). It means if you randomly select one customer who churned and one who stayed, there is a 79.5% chance the model will rank the churner as higher risk.

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

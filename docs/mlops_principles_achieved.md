# Summary of MLOps Principles Achieved

This document summarizes the core MLOps (Machine Learning Operations) principles established and verified during this Proof of Concept (POC) pipeline.

---

## 1. Reproducibility & Version Control (Git + DVC)
* **The Principle**: Code, configuration, datasets, and model artifacts must be version-controlled in tandem.
* **Our Implementation**: 
  * Git tracks the source code, CI/CD configuration, and DVC pointer files (`.dvc`, `dvc.lock`).
  * **DVC (Data Version Control)** tracks the large datasets (`data/telco_churn.csv`) and model binaries (`models/model.joblib`) in an Amazon S3 remote store.
  * **The Result**: Any team member checking out a specific Git commit can run `dvc pull` and immediately retrieve the *exact* dataset and model binary associated with that code commit, guaranteeing 100% reproducibility.

---

## 2. Experiment Tracking & Lineage (MLflow)
* **The Principle**: Every training execution must be audited, logging hyper-parameters, metrics, and artifact references to track model evolution.
* **Our Implementation**:
  * Integrated **MLflow** into `src/train.py`.
  * Every training execution automatically logs hyper-parameters (`n_estimators`, `max_depth`), metrics (`accuracy`, `precision`, `recall`, `f1_score`), and the serialized model artifact.
  * **The Result**: Provides complete lineage auditability. You can easily compare historical runs (e.g. baseline vs. retrained models) to justify deployments.

---

## 3. Continuous Monitoring (CM) & Drift Detection
* **The Principle**: Models degrade in production due to real-world shifts. Inputs must be continuously audited for drift.
* **Our Implementation**:
  * Set up automated drift analysis using **Evidently AI** comparing incoming production batches to the training baseline.
  * Configured a daily cron job (`monitor.yml`) to automatically output visual HTML drift reports and return non-zero exit codes if critical feature distributions (like `MonthlyCharges` or `tenure`) skew.
  * **The Result**: Automated early warning sensor that flags model degradation *before* it leads to bad business decisions.

---

## 4. Continuous Training (CT)
* **The Principle**: When data drift or performance degradation is flagged, retraining the model on the updated distribution must be seamless.
* **Our Implementation**:
  * Appending new production logs to the training dataset updates the file hash.
  * **DVC pipelines** (`dvc repro`) detect the file modification and re-run the preparation and training steps automatically, producing an updated `model.joblib` and `metrics.json`.
  * **The Result**: Re-establishes accuracy and resets the monitoring baseline without manual code rewrites.

---

## 5. Continuous Deployment (CD) & Production Engineering
* **The Principle**: Deployment of a verified model must be automated, secure, and optimized for production workloads.
* **Our Implementation**:
  * **Automation**: Pushing code updates to GitHub triggers a workflow (`train-deploy.yml`) that builds an optimized Docker image, pushes it to ECR, and updates the Lambda serving code.
  * **Optimization**: Solved serverless cold-start bottlenecks by:
    1. Lazy-loading the model binary inside the Lambda handler.
    2. Pinning lightweight production dependencies (`requirements-predict.txt`) to reduce file size.
    3. Allocating **1024 MB** of memory to assign a full vCPU to the MicroVM, speeding up imports and block loading.
  * **Security**: Secured the endpoint using a Lambda Function URL protected by **`AWS_IAM` authentication** (requiring SigV4-signed requests).
  * **The Result**: A secure, scalable, serverless API that starts up fast and responds in under **35 milliseconds** for warm starts.

# Phase 1 — Task Checklist

Tracking implementation progress for the MLOps POC.
Reference: [MLOps_POC.md](file:///c:/Work/MLOps/telco-churn-pipeline/docs/MLOps_POC.md)

**Resolved decisions:**
- S3 bucket: create new (`mlops-poc-dvc-telco-churn`)
- Lambda function name: `telco-churn-predictor`
- Python version: 3.11
- GitHub remote: `https://github.com/shimul-iut/mlops-poc-telco-churn.git`

---

## Day 1 — Foundations (Data + Model + Experiment Tracking)

### 1.1 Project Scaffolding
- [x] Create `requirements.txt` with pinned dependencies
- [x] Create `.gitignore` (data/, mlruns/, models/, __pycache__, .dvc/cache/)
- [x] Create `dvc.yaml` with `prepare` and `train` stages
- [ ] Initialize DVC (`dvc init`)
- [ ] Configure DVC S3 remote (`dvc remote add -d s3remote s3://mlops-poc-dvc-telco-churn/dvc-store`)
- [x] Set up Python virtual environment and install dependencies
- [ ] Connect Git remote (`git remote add origin https://github.com/shimul-iut/mlops-poc-telco-churn.git`)

### 1.2 Data Pipeline
- [x] Create `data/.gitkeep`
- [x] Create `src/prepare_data.py`
  - [x] Download Telco Churn CSV from IBM GitHub URL
  - [x] Clean `TotalCharges` (coerce + fill NaN with median)
  - [x] Map `Churn` → binary (1/0)
  - [x] Label-encode 15 categorical columns
  - [x] Write processed data to `data/telco_churn.csv`
- [x] Run `prepare_data.py` and verify output (7,043 rows × 19 columns + target)

### 1.3 Model Training & Experiment Tracking
- [x] Create `src/train.py`
  - [x] Load processed data, 80/20 stratified split
  - [x] Train `RandomForestClassifier` (n_estimators=100, random_state=42)
  - [x] Log params, metrics, model to MLflow (local `mlruns/`)
  - [x] Save model to `models/model.joblib`
  - [x] Write metrics to `metrics.json`
- [x] Run `train.py` and verify MLflow logs (`mlflow ui`)
- [x] Create `models/.gitkeep`

### 1.4 Data & Model Validation
- [x] Create `src/validate.py`
  - [x] Schema validation (19 expected columns, correct dtypes)
  - [x] Data quality checks (no NaNs, Churn ∈ {0,1}, sane numeric ranges)
  - [x] Model quality check (accuracy ≥ 0.75)
- [x] Create `tests/__init__.py`
- [x] Create `tests/test_model.py`
  - [x] `test_data_schema`
  - [x] `test_no_missing_values`
  - [x] `test_model_loads`
  - [x] `test_model_accuracy_threshold`
  - [x] `test_prediction_shape`
- [x] Run `pytest tests/ -v` — all green

### 1.5 Version & Push
- [ ] `dvc add data/telco_churn.csv models/model.joblib`
- [ ] Create S3 bucket (`aws s3 mb s3://mlops-poc-dvc-telco-churn`)
- [ ] `dvc push` — data + model to S3
- [ ] Initial Git commit + push to `origin/main`

---

## Day 2 — CI/CD (GitHub Actions → Lambda)

### 2.1 Inference Handler
- [x] Create `src/predict.py`
  - [x] Cold-start model loading from container filesystem
  - [x] `handler(event, context)` — parse JSON, build DataFrame, predict, return result
  - [x] Error handling (missing fields → 400)
  - [x] Stdout logging for CloudWatch

### 2.2 Docker Image
- [x] Create `Dockerfile` (base: `public.ecr.aws/lambda/python:3.11`)
- [ ] Build locally: `docker build -t telco-churn-predictor .`
- [ ] Test locally: `docker run -p 9000:8080 telco-churn-predictor` + curl invocation

### 2.3 AWS Infrastructure (One-time Setup)
- [ ] Create ECR repository (`aws ecr create-repository --repository-name telco-churn-predictor`)
- [ ] Set ECR lifecycle policy (expire untagged images after 7 days)
- [ ] Create IAM role `telco-churn-lambda-role` with `AWSLambdaBasicExecutionRole`
- [ ] Push initial Docker image to ECR
- [ ] Create Lambda function (container image, 512MB memory, 30s timeout)
- [ ] Create Function URL (`--auth-type AWS_IAM`)
- [ ] Test with a signed request (`awscurl` or Python SigV4)

### 2.4 CI Workflow
- [x] Create `.github/workflows/ci.yml`
  - [x] Trigger: `pull_request` to `main`
  - [x] Steps: checkout → setup Python 3.11 → install deps → lint (`ruff`) → pytest → validate
- [ ] Configure GitHub secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ACCOUNT_ID`, `ECR_REPOSITORY`

### 2.5 CT + CD Workflow
- [x] Create `.github/workflows/train-deploy.yml`
  - [x] Trigger: `push` to `main`, `workflow_dispatch`
  - [x] Steps: checkout → setup Python → install deps → retrain → validate → docker build → ECR login → push → update Lambda → wait → smoke test
- [ ] Verify: merge a PR → pipeline runs end-to-end → Lambda updated

---

## Day 3 — CT Loop + Continuous Monitoring (CM)

### 3.1 Drift Monitoring
- [x] Create `src/monitor.py`
  - [x] Load training data as Evidently reference dataset
  - [x] Load "current" batch (simulated or from S3)
  - [x] Generate `DataDriftPreset` report → `reports/drift_report.html`
  - [x] Print pass/fail summary, exit non-zero on drift
- [x] Create `reports/.gitkeep`
- [x] Test locally with synthetic drifted data

### 3.2 Monitoring Workflow
- [x] Create `.github/workflows/monitor.yml`
  - [x] Trigger: `schedule` (daily cron), `workflow_dispatch`
  - [x] Steps: checkout → setup Python → install deps → run monitor.py → upload drift report artifact
- [ ] Manually trigger and verify report artifact is downloadable

### 3.3 CloudWatch Setup (One-time)
- [ ] Create metric filter on Lambda log group for `ERROR` → `churn-predictor/ErrorCount`
- [ ] Create CloudWatch alarm: `ErrorCount > 5` in 5 min → SNS notification
- [ ] Verify alarm fires with a test error

### 3.4 End-to-End Loop Simulation
- [ ] Inject drifted data → run `monitor.py` → drift flagged
- [ ] Manually trigger `train-deploy.yml` → new model deployed
- [ ] Verify new model is live via Function URL smoke test

---

## Final

- [x] Create `README.md` (quick-start, architecture, usage)
- [ ] Review all success criteria from Section 8:
  - [ ] Model is trained, versioned, and tracked (Git + DVC + MLflow)
  - [ ] CI blocks a bad model/data change automatically
  - [ ] Merge to `main` deploys a new model to Lambda with no manual steps
  - [ ] Scheduled job detects data drift and produces a report
  - [ ] Total AWS spend stays under $10
- [ ] Clean up: verify no unexpected AWS costs in billing dashboard

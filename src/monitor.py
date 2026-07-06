import os
import sys
import pandas as pd
import numpy as np
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

REFERENCE_DATA_PATH = "data/telco_churn.csv"
REPORT_DIR = "reports"
REPORT_PATH = os.path.join(REPORT_DIR, "drift_report.html")

def run_monitoring():
    print("Starting drift monitoring...")
    
    # Ensure reports directory exists
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    if not os.path.exists(REFERENCE_DATA_PATH):
        print(f"ERROR: Reference dataset not found at {REFERENCE_DATA_PATH}. Run prepare_data.py first.")
        sys.exit(1)
        
    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    
    # Determine the "current" dataset
    # If a path is provided as command-line argument, use it; otherwise generate synthetic drifted data
    if len(sys.argv) > 1:
        current_data_path = sys.argv[1]
        if not os.path.exists(current_data_path):
            print(f"ERROR: Current dataset not found at {current_data_path}")
            sys.exit(1)
        print(f"Loading current dataset from: {current_data_path}")
        current_df = pd.read_csv(current_data_path)
    else:
        print("No current dataset provided. Generating synthetic drifted dataset...")
        current_df = reference_df.copy()
        
        # Inject synthetic drift by skewing MonthlyCharges and tenure (Day 3 exercise)
        # Shift MonthlyCharges up by $35 and decrease tenure by 12 months
        current_df["MonthlyCharges"] = current_df["MonthlyCharges"] + 35.0
        current_df["tenure"] = (current_df["tenure"] - 12).clip(lower=0)
        
        # Randomly modify a few categorical variables to simulate behavior shifts
        current_df["Contract"] = np.random.choice([0, 1, 2], size=len(current_df), p=[0.7, 0.2, 0.1])
        print("Synthetic drift injected successfully into 'MonthlyCharges', 'tenure', and 'Contract'.")

    # Drop target column 'Churn' for drift analysis on input features
    features_ref = reference_df.drop(columns=["Churn"], errors="ignore")
    features_cur = current_df.drop(columns=["Churn"], errors="ignore")
    
    # Set up Evidently report with DataDriftPreset
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    
    print("Running Evidently drift analysis...")
    drift_report.run(reference_data=features_ref, current_data=features_cur)
    
    # Save HTML report
    drift_report.save_html(REPORT_PATH)
    print(f"Drift report HTML saved successfully to {REPORT_PATH}")
    
    # Extract report metrics to print summary and determine pass/fail
    report_dict = drift_report.as_dict()
    
    # Convert list of metrics to dictionary for easy access
    metrics_by_id = {m["metric"]: m["result"] for m in report_dict["metrics"]}
    dataset_drift_result = metrics_by_id.get("DatasetDriftMetric", {})
    drift_table_result = metrics_by_id.get("DataDriftTable", {})
    
    number_of_features = dataset_drift_result.get("number_of_columns", 0)
    drifted_features = dataset_drift_result.get("number_of_drifted_columns", 0)
    share_of_drifted_features = dataset_drift_result.get("share_of_drifted_columns", 0.0)
    dataset_drift = dataset_drift_result.get("dataset_drift", False)
    
    print(f"\n--- Monitoring Summary ---")
    print(f"Total features checked: {number_of_features}")
    print(f"Drifted features: {drifted_features} ({share_of_drifted_features * 100:.2f}%)")
    print(f"Dataset-level drift detected: {dataset_drift}")
    
    # Log individual feature drift details
    print("\nFeature drift details:")
    for col, detail in drift_table_result.get("drift_by_columns", {}).items():
        is_drifted = detail["drift_detected"]
        p_value = detail.get("drift_score", "N/A")
        print(f"  - {col}: {'[DRIFTED]' if is_drifted else '[STABLE]'} (p-value: {p_value:.4f})" if isinstance(p_value, (int, float)) else f"  - {col}: {'[DRIFTED]' if is_drifted else '[STABLE]'} (p-value: {p_value})")
        
    print("--------------------------\n")
    
    # Exit with non-zero code if dataset-level drift is detected
    if dataset_drift:
        print("WARNING: Dataset-level drift detected above acceptable threshold!")
        sys.exit(2) # Exit code 2 indicates drift detected
    else:
        print("Success: Dataset is stable. No significant drift detected.")
        sys.exit(0)

if __name__ == "__main__":
    try:
        run_monitoring()
    except Exception as e:
        print(f"Error executing monitoring: {e}")
        sys.exit(1)

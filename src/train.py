"""Training script. Works locally AND as a SageMaker training job entry point."""
import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "monthly_cost_usd"
CATEGORICAL = ["region", "pricing_model", "service_type"]
NUMERIC = ["vcpus", "ram_gb", "gpu_count", "storage_gb", "hours_per_month", "data_transfer_gb"]


def build_pipeline(n_estimators: int, max_depth: int, learning_rate: float) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=42,
    )
    return Pipeline([("pre", pre), ("model", model)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.1)
    # SageMaker injects these environment variables; defaults make local runs work too
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "models"))
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    p.add_argument("--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "models"))
    a = p.parse_args()

    df = pd.read_csv(os.path.join(a.train, "cloud_workloads.csv"))
    X, y = df[CATEGORICAL + NUMERIC], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline(a.n_estimators, a.max_depth, a.learning_rate).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {"mae_usd": float(mean_absolute_error(y_te, pred)), "r2": float(r2_score(y_te, pred))}
    print(f"[metrics] mae_usd={metrics['mae_usd']:.2f} r2={metrics['r2']:.4f}")  # SageMaker can regex these

    os.makedirs(a.model_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(a.model_dir, "model.joblib"))
    os.makedirs(a.output_data_dir, exist_ok=True)
    with open(os.path.join(a.output_data_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

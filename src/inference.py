"""SageMaker inference handlers (model_fn / input_fn / predict_fn / output_fn)."""
import json
import os

import joblib
import pandas as pd

COLUMNS = ["region", "pricing_model", "service_type", "vcpus", "ram_gb",
           "gpu_count", "storage_gb", "hours_per_month", "data_transfer_gb"]


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    data = json.loads(request_body)
    if isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)[COLUMNS]


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, accept="application/json"):
    return json.dumps({"monthly_cost_usd": [round(float(p), 2) for p in prediction]})

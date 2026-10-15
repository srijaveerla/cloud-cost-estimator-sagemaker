"""CLI: predict monthly cost for a workload using the local model or a SageMaker endpoint.

Examples:
  python src/estimate.py --vcpus 4 --ram-gb 16 --storage-gb 200 --hours 730
  python src/estimate.py --endpoint <endpoint-name> --vcpus 4 --ram-gb 16
"""
import argparse
import json

import pandas as pd

COLUMNS = ["region", "pricing_model", "service_type", "vcpus", "ram_gb",
           "gpu_count", "storage_gb", "hours_per_month", "data_transfer_gb"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vcpus", type=int, default=2)
    p.add_argument("--ram-gb", type=int, default=8)
    p.add_argument("--gpu-count", type=int, default=0)
    p.add_argument("--storage-gb", type=int, default=100)
    p.add_argument("--hours", type=int, default=730)
    p.add_argument("--transfer-gb", type=int, default=50)
    p.add_argument("--region", default="ap-south-1")
    p.add_argument("--pricing-model", default="on_demand",
                   choices=["on_demand", "reserved_1yr", "spot"])
    p.add_argument("--service-type", default="web_app")
    p.add_argument("--endpoint", help="SageMaker endpoint name (omit to use local model)")
    p.add_argument("--model-path", default="models/model.joblib")
    a = p.parse_args()

    row = {"region": a.region, "pricing_model": a.pricing_model, "service_type": a.service_type,
           "vcpus": a.vcpus, "ram_gb": a.ram_gb, "gpu_count": a.gpu_count,
           "storage_gb": a.storage_gb, "hours_per_month": a.hours,
           "data_transfer_gb": a.transfer_gb}

    if a.endpoint:
        import boto3
        rt = boto3.client("sagemaker-runtime")
        resp = rt.invoke_endpoint(EndpointName=a.endpoint, ContentType="application/json",
                                  Body=json.dumps(row))
        cost = json.loads(resp["Body"].read())["monthly_cost_usd"][0]
    else:
        import joblib
        cost = float(joblib.load(a.model_path).predict(pd.DataFrame([row])[COLUMNS])[0])

    print(f"Estimated monthly cost: ${cost:,.2f}")


if __name__ == "__main__":
    main()

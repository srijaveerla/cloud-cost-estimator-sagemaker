"""Run training on Amazon SageMaker and (optionally) deploy a real-time endpoint.

Prereqs: AWS CLI configured (`aws configure`) and a SageMaker execution role ARN.
Usage:
  python sagemaker/launch_training.py --role arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerRole> --deploy
"""
import argparse

import sagemaker
from sagemaker.sklearn.estimator import SKLearn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--deploy", action="store_true")
    a = ap.parse_args()

    session = sagemaker.Session()
    bucket = session.default_bucket()
    train_uri = session.upload_data("data/cloud_workloads.csv", bucket=bucket,
                                    key_prefix="cloud-cost-estimator/train")

    est = SKLearn(
        entry_point="train.py",
        source_dir="src",
        role=a.role,
        instance_type=a.instance_type,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        hyperparameters={"n-estimators": 300, "max-depth": 4, "learning-rate": 0.1},
        metric_definitions=[{"Name": "mae_usd", "Regex": r"mae_usd=([0-9\.]+)"},
                            {"Name": "r2", "Regex": r"r2=([0-9\.]+)"}],
        sagemaker_session=session,
    )
    est.fit({"train": train_uri})

    if a.deploy:
        predictor = est.deploy(initial_instance_count=1, instance_type="ml.t2.medium",
                               entry_point="inference.py", source_dir="src")
        print("Endpoint name:", predictor.endpoint_name)
        print("IMPORTANT: delete it when done to stop charges ->",
              "python sagemaker/cleanup.py --endpoint", predictor.endpoint_name)


if __name__ == "__main__":
    main()

"""Delete a SageMaker endpoint so you are not billed while idle."""
import argparse
import boto3

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", required=True)
a = ap.parse_args()
sm = boto3.client("sagemaker")
sm.delete_endpoint(EndpointName=a.endpoint)
print(f"Deleted endpoint {a.endpoint}")

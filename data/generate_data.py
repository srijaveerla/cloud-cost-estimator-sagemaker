"""Generate a synthetic cloud-workload dataset (replace with real AWS billing/pricing data later)."""
import argparse
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# Illustrative (NOT official) unit prices in USD
PRICE_VCPU_HR = 0.032
PRICE_RAM_GB_HR = 0.0045
PRICE_STORAGE_GB_MONTH = 0.08
PRICE_TRANSFER_GB = 0.09
PRICE_GPU_HR = 0.90

REGION_MULT = {"ap-south-1": 1.00, "us-east-1": 0.95, "eu-west-1": 1.10, "ap-southeast-1": 1.08}
PRICING_MULT = {"on_demand": 1.00, "reserved_1yr": 0.62, "spot": 0.35}
SERVICES = ["web_app", "database", "ml_training", "batch_job", "analytics"]


def make_dataset(n: int = 5000) -> pd.DataFrame:
    df = pd.DataFrame({
        "vcpus": RNG.choice([1, 2, 4, 8, 16, 32], n),
        "ram_gb": RNG.choice([2, 4, 8, 16, 32, 64, 128], n),
        "gpu_count": RNG.choice([0, 0, 0, 0, 1, 2, 4], n),
        "storage_gb": RNG.integers(20, 2000, n),
        "hours_per_month": RNG.integers(20, 730, n),
        "data_transfer_gb": RNG.integers(0, 1500, n),
        "region": RNG.choice(list(REGION_MULT), n),
        "pricing_model": RNG.choice(list(PRICING_MULT), n, p=[0.55, 0.3, 0.15]),
        "service_type": RNG.choice(SERVICES, n),
    })
    compute = df.hours_per_month * (
        df.vcpus * PRICE_VCPU_HR + df.ram_gb * PRICE_RAM_GB_HR + df.gpu_count * PRICE_GPU_HR
    )
    base = compute * df.region.map(REGION_MULT) * df.pricing_model.map(PRICING_MULT)
    base += df.storage_gb * PRICE_STORAGE_GB_MONTH + df.data_transfer_gb * PRICE_TRANSFER_GB
    df["monthly_cost_usd"] = (base * RNG.normal(1.0, 0.05, n)).round(2)  # 5% noise
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--out", default="data/cloud_workloads.csv")
    a = ap.parse_args()
    make_dataset(a.rows).to_csv(a.out, index=False)
    print(f"Saved {a.rows} rows to {a.out}")

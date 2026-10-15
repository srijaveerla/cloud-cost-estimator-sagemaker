# ☁️ AI Cloud Cost Estimator (Amazon SageMaker)

A machine-learning project that **predicts the monthly cloud cost of a workload** (vCPUs, RAM, GPU, storage,
hours, data transfer, region, pricing model) and runs **on AWS SageMaker** for training and real-time inference.

> Built as an internship project. Data is synthetic and prices are illustrative — swap in real AWS billing/pricing data for production use.

## Architecture
```
generate_data.py ─► cloud_workloads.csv ─► S3 ─► SageMaker Training Job (src/train.py)
                                                       │
                                                       ▼
                                           model.tar.gz in S3 ─► SageMaker Endpoint (src/inference.py)
                                                                          │
                                                       src/estimate.py ◄──┘ (CLI client)
```

## Project structure
| Path | Purpose |
|---|---|
| `data/generate_data.py` | Builds the synthetic workload/cost dataset |
| `src/train.py` | Trains a Gradient Boosting pipeline (local or SageMaker) |
| `src/inference.py` | SageMaker inference handlers |
| `src/estimate.py` | CLI to get a cost estimate (local model or endpoint) |
| `sagemaker/launch_training.py` | Starts the training job and deploys an endpoint |
| `sagemaker/cleanup.py` | Deletes the endpoint (avoid charges) |
| `tests/` | Pytest tests |

## Quick start (local, no AWS needed)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/generate_data.py
python src/train.py
python src/estimate.py --vcpus 4 --ram-gb 16 --storage-gb 200 --hours 730
pytest -q
```

## Run on AWS SageMaker
1. `aws configure` (IAM user with SageMaker + S3 access)
2. Create a SageMaker execution role (IAM → Roles → SageMaker) and copy its ARN
3. Train and deploy:
```bash
python sagemaker/launch_training.py --role arn:aws:iam::<ACCOUNT_ID>:role/<ROLE_NAME> --deploy
python src/estimate.py --endpoint <ENDPOINT_NAME> --vcpus 8 --ram-gb 32 --pricing-model spot
```
4. **Delete the endpoint when finished:** `python sagemaker/cleanup.py --endpoint <ENDPOINT_NAME>`

## Results
Add your metrics here after training (MAE in USD, R²) — see `models/metrics.json`.

## Future work
Real AWS Pricing API data · Streamlit dashboard · Cost forecasting over time · Multi-cloud (Azure/GCP)

## License
MIT

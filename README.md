
# ☁️ AI Cloud Cost Estimator (Amazon SageMaker)

An AI and Machine Learning project that estimates monthly cloud costs based on workload resources such as vCPUs, RAM, GPU, storage, usage hours, data transfer, region, and pricing model.

The project uses a machine learning model to predict cloud expenses and demonstrates how training and real-time inference can run on AWS SageMaker.

> Built as an AI/ML student project. The dataset is synthetic and prices are illustrative, so predictions are for demonstration purposes only.

## Features

- Estimate monthly cloud costs using workload specifications.
- Generate a synthetic cloud workload dataset.
- Train a Gradient Boosting machine learning model.
- Run predictions locally without AWS.
- Support AWS SageMaker training and real-time inference.
- Evaluate model performance using MAE and R² metrics.
- Clean up deployed SageMaker endpoints to help avoid unnecessary charges.

## Architecture

```text
Generate Dataset
       |
       v
cloud_workloads.csv
       |
       v
Train ML Model
       |
       v
Local Model or AWS SageMaker
       |
       v
Cloud Cost Prediction
```

## Project Structure

| Path | Purpose |
|---|---|
| `data/generate_data.py` | Generates the synthetic workload dataset |
| `data/cloud_workloads.csv` | Workload and cost data |
| `src/train.py` | Trains the machine learning model |
| `src/inference.py` | SageMaker inference handlers |
| `src/estimate.py` | Estimates costs using a local model or endpoint |
| `sagemaker/launch_training.py` | Launches training and optionally deploys an endpoint |
| `sagemaker/cleanup.py` | Deletes a deployed SageMaker endpoint |
| `models/metrics.json` | Stores model evaluation metrics |
| `tests/test_pipeline.py` | Tests the project pipeline |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.10 or later
- pip
- AWS account and AWS CLI only if using SageMaker

## Quick Start (Local)

No AWS account is required for local model training and prediction.

### 1. Clone the repository

```bash
git clone https://github.com/srijaveerla/cloud-cost-estimator-sagemaker.git
cd cloud-cost-estimator-sagemaker
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate data and train the model

```bash
python data/generate_data.py
python src/train.py
```

### 5. Estimate cloud costs

```bash
python src/estimate.py --vcpus 4 --ram-gb 16 --storage-gb 200 --hours 730
```

### 6. Run tests

```bash
pytest -q
```

## AWS SageMaker Deployment

AWS deployment may incur charges.

1. Install and configure the AWS CLI.
2. Configure AWS credentials with the necessary permissions.
3. Create or identify an appropriate SageMaker execution role.
4. Launch training and optionally deploy an endpoint:

```bash
python sagemaker/launch_training.py --role YOUR_SAGEMAKER_ROLE_ARN --deploy
```

5. Use the endpoint for prediction:

```bash
python src/estimate.py --endpoint YOUR_ENDPOINT_NAME --vcpus 8 --ram-gb 32 --pricing-model spot
```

6. Delete the endpoint when finished:

```bash
python sagemaker/cleanup.py --endpoint YOUR_ENDPOINT_NAME
```

Replace the example role ARN and endpoint name with your actual AWS values. Do not commit AWS credentials or secrets to GitHub.

## Model Evaluation

Model evaluation metrics such as Mean Absolute Error (MAE) and R² are stored in `models/metrics.json` after training, if supported by the training pipeline.

Because the dataset is synthetic, results do not establish accuracy against actual AWS bills.

## Future Improvements

- Integrate real cloud pricing and billing data.
- Add an interactive Streamlit dashboard.
- Forecast cloud spending over time.
- Recommend cost-saving configurations.
- Explore multi-cloud support for AWS, Azure, and Google Cloud.

## Technologies

Python · Pandas · Scikit-learn · Machine Learning · AWS SageMaker · Pytest

## Author

AI & Machine Learning Student

## License

MIT

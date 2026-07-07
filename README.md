# 🏦 End-to-End Production Loan Approval Prediction System

An asynchronous, containerized Machine Learning microservice built around the **Ethicalstar Loan Prediction Dataset**. It evaluates applicant default parameters and leverages **SHAP TreeExplainers** to deliver localized feature attributions for banking compliance.

---

## Overview
This repository contains:
- **FastAPI inference API** (async request handling)
- **Streamlit dashboard** for interactive exploration
- **Training + evaluation pipeline** (RandomizedSearchCV + MLflow tracking)
- **Model explainability** using SHAP

---

## Dataset
- Dataset: **Ethicalstar Loan Prediction Dataset**
- Source file: `data/raw/Loan Prediction.csv`
- Target (label): loan default / approval (as defined in the dataset schema)
- Notes: preprocessing is implemented to reduce leakage (missing values/outliers handling).

(See `notebooks/01_exploratory_data_analysis.ipynb` for deeper dataset exploration.)

---

## Installation
### Local setup
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## Training
Run the training pipeline (hyperparameter search + export pipelines):
```bash
python -m src.train
```

Generate evaluation artifacts (metrics/plots + SHAP summaries):
```bash
python -m src.evaluate
```

---

## API Usage
### Start the FastAPI service
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Predict endpoint
- Route: `POST /api/v1/predict`
- Auto docs:
  - Swagger: `http://localhost:8000/docs`
  - ReDoc: `http://localhost:8000/redoc`

### Run API tests
```bash
pytest tests/test_api.py
```

---

## Results
- Selected champion model: **`XGBClassifier`** optimized via `RandomizedSearchCV`
- Reported F1 (holdout): **0.8940**
- Compared against Logistic Regression (reported F1: **0.7120**)
- Explainability:
  - SHAP global summary plots and per-request attributions are generated for compliance-friendly interpretability.

---

## Future improvements
- Add drift monitoring for categorical features (e.g., `CITY`, `Profession`).
- Expand tests to cover additional edge cases (invalid schema inputs, extreme values).
- Improve observability: structured logging for inference latency + prediction distribution.
- Add model version registry & automatic promotion thresholds.

---

## 📊 Workspace Directory Architecture
```text
loan_approval_system/
├── api/                     # FastAPI App (Asynchronous inference framework)
│   ├── dependencies.py      # Dependency Injection decoupling ML models
│   └── main.py              # Schema contracts and API gateways
├── dashboard/               # Streamlit application UI engine
│   └── app.py
├── data/
│   ├── processed/           # Filtered matrices
│   └── raw/                 # Base dataset snapshot (Loan Prediction.csv)
├── models/                  # Serialized Joblib pipelines & training results
├── src/                     # Core engineering modules
│   ├── features.py          # Custom domain underwriting logic
│   ├── preprocessing.py     # Leakage-proof missing value and outlier management
│   ├── train.py             # RandomizedSearchCV and MLflow tracking engine
│   └── evaluate.py          # Metric validation plots and SHAP evaluations
├── tests/                   # PyTest integration suites
├── utils/                   # Loguru unified tracing modules
├── Dockerfile               # Secure multi-stage deployment blueprint
└── requirements.txt         # Strict dependency mapping list
```

---

## 📦 Containerization Platform Commands
Build the production image and run the API + dashboard:

```bash
docker build -t loan-underwriting-system:latest .

docker run -d -p 8000:8000 --name loan-api loan-underwriting-system:latest

docker run -d -p 8501:8501 --net=host --name loan-ui loan-underwriting-system:latest streamlit run dashboard/app.py --server.port=8501
```


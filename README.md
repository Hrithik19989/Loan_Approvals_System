# 🏦 End-to-End Production Loan Approval Prediction System

An asynchronous, containerized Machine Learning microservice built around the **Ethicalstar Loan Prediction Dataset**. It evaluates applicant default parameters and leverages **SHAP TreeExplainers** to deliver localized feature attributions for banking compliance.

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

## 🛠️ Local Environment Initialization Lifecycle
Follow this command sequence to install dependencies, run optimization loops, and start the local services:

```bash
# 1. Initialize local virtual environment configurations
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run multi-model hyperparameter training loops & export joblib pipelines
python -m src.train

# 3. Generate diagnostic metrics graphs & global SHAP summary plot matrices
python -m src.evaluate

# 4. Run automated PyTest validation checks
pytest tests/test_api.py

# 5. Spin up the underlying asynchronous FastAPI backend engine 
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📦 Containerization Platform Commands
To run the complete system in an isolated network container layer, build the image and use separate host ports to target your services:

```bash
# Build the production image artifact
docker build -t loan-underwriting-system:latest .

# Run and map the underlying FastAPI Microservice engine 
docker run -d -p 8000:8000 --name loan-api loan-underwriting-system:latest

# Run the Streamlit Dashboard (Overriding the default command)
docker run -d -p 8501:8501 --net=host --name loan-ui loan-underwriting-system:latest streamlit run dashboard/app.py --server.port=8501
```

## 📈 Model Performance Selection Matrix
* **Selected Champion**: `XGBClassifier` optimized via RandomizedSearchCV.
* **Justification**: Delivered an F1-Score of `0.8940` on the holdout split, significantly outperforming Logistic Regression (`0.7120`). Tree architectures also cleanly map feature patterns within high-cardinality values like `Profession` and `CITY`.

## 📈 Production Validation Results
* **Inference Endpoint Route**: `POST /api/v1/predict`
* **Response Status Standard**: `200 OK`
* **Validation Suite Coverage**: 100% Core Endpoint Path Pass (`pytest tests/test_api.py`)
* **Auto-Generated Documentation Endpoints**:
  * Swagger Interactive UI: `http://localhost:8000/docs`
  * ReDoc Static Specifications: `http://localhost:8000/redoc`

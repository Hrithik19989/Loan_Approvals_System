# api/main.py
"""
Asynchronous credit risk microservice exposing predictive inference models 
and localized SHAP feature explanations using modern lifespan events.
"""

import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List

from api.dependencies import model_provider
from utils.logger import get_production_logger

app_logger = get_production_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan event handler that guarantees heavy model components 
    finish caching before the gateway opens for network traffic.
    """
    try:
        app_logger.info("Triggering app initialization lifecycle...")
        model_provider.bootstrap_artifacts()
        app_logger.info("API resources successfully mapped. Serving active paths.")
        yield
    except Exception as initialization_error:
        app_logger.critical(f"App initialization failed to launch safely: {str(initialization_error)}")
        raise initialization_error
    finally:
        app_logger.info("Shutting down microservice context pipelines...")


app = FastAPI(
    title="Institutional Risk Assessment API Gateway",
    description="Asynchronous credit underwriting classification layer optimized around the Ethicalstar schema.",
    version="1.0.0",
    lifespan=lifespan
)

class ApplicationSchema(BaseModel):
    Id: int = Field(..., description="Unique record index mapping identifier.")
    Income: float = Field(..., ge=0.0, description="Annual continuous gross income.")
    Age: int = Field(..., ge=18, le=100, description="Applicant age in years.")
    Experience: int = Field(..., ge=0, description="Total professional career experience.")
    Married_Single: Literal["married", "single"] = Field(..., alias="Married/Single")
    House_Ownership: Literal["rented", "owned", "norent_noown"] = Field(...)
    Car_Ownership: Literal["yes", "no"] = Field(...)
    Profession: str = Field(..., min_length=1)
    CITY: str = Field(..., min_length=1)
    STATE: str = Field(..., min_length=1)
    CURRENT_JOB_YRS: int = Field(..., ge=0)
    CURRENT_HOUSE_YRS: int = Field(..., ge=0)

    # FIXED: Replaced deprecated class-based Config with Pydantic v2 ConfigDict
    model_config = ConfigDict(populate_by_name=True)

class EvaluationResponse(BaseModel):
    Id: int
    underwriting_decision: Literal["Approved (Low Risk)", "Rejected (High Risk)"]
    approval_probability: float
    primary_risk_drivers: List[str]


@app.post("/api/v1/predict", response_model=EvaluationResponse)
async def execute_underwriting_prediction(payload: ApplicationSchema):
    """
    Transforms runtime JSON strings, scales features via historical parameters,
    and returns a decision alongside local SHAP driver features.
    """
    pipeline = model_provider.get_pipeline()
    explainer = model_provider.get_explainer()

    if pipeline is None or explainer is None:
        app_logger.error("Inbound request blocked: Memory artifacts are uninitialized.")
        raise HTTPException(status_code=503, detail="Prediction models are offline or booting.")

    app_logger.info(f"Received scoring sequence payload for Application ID: {payload.Id}")

    try:
        # 1. Structured DataFrame conversion preserving alias fields safely
        raw_payload_df = pd.DataFrame([payload.model_dump(by_alias=True)])
        
        # 2. Transform raw structural vectors up to the estimator threshold
        processor_step = pipeline.named_steps['processor']
        preprocessed_feature_matrix = processor_step.transform(raw_payload_df)
        
        # OPTIMIZATION: Ensure sparse matrix output from OneHotEncoder is converted to dense format for SHAP compatibility
        if hasattr(preprocessed_feature_matrix, "toarray"):
            dense_feature_matrix = preprocessed_feature_matrix.toarray()
        else:
            dense_feature_matrix = preprocessed_feature_matrix
        
        # 3. Calculate target probabilities safely out of index matrices
        risk_probability = float(pipeline.named_steps['estimator'].predict_proba(dense_feature_matrix)[0][1])
        approval_probability = 1.0 - risk_probability
        
        # Mapped evaluation threshold (Standard 0.4 constraint)
        decision = "Approved (Low Risk)" if risk_probability < 0.4 else "Rejected (High Risk)"
        
        # 4. Generate local localized explanation profiles using SHAP over dense array matrix
        shap_values = explainer(dense_feature_matrix)
        
        # Fallback feature names extraction if custom transformers do not implement get_feature_names_out
        try:
            engineered_feature_labels = processor_step.get_feature_names_out()
        except AttributeError:
            engineered_feature_labels = [f"feature_{i}" for i in range(dense_feature_matrix.shape[1])]
        
        # Zip, match, and extract top absolute driver metrics
        # SHAP may return array-like values per feature (e.g., multi-class). Reduce to a scalar per feature
        # so ranking logic remains deterministic.
        def _to_scalar(v):
            import numpy as np
            arr = np.asarray(v)
            if arr.ndim == 0:
                return float(arr)
            # Rank by magnitude across the vector (mean absolute contribution)
            return float(np.mean(np.abs(arr)))

        shap_per_feature = shap_values.values[0]
        feature_importance_map = {
            label: _to_scalar(v)
            for label, v in zip(engineered_feature_labels, shap_per_feature)
        }

        sorted_risk_drivers = sorted(
            feature_importance_map,
            key=lambda key_flag: feature_importance_map[key_flag],
            reverse=True,
        )[:3]

        
        app_logger.info(f"Underwriting evaluation for ID {payload.Id} processed cleanly. Decision: {decision}")
        
        return EvaluationResponse(
            Id=payload.Id,
            underwriting_decision=decision,
            approval_probability=round(approval_probability, 4),
            primary_risk_drivers=sorted_risk_drivers
        )

    except Exception as execution_exception:
        app_logger.exception(f"Runtime failure tracking execution for application matrix ID {payload.Id}: {str(execution_exception)}")
        raise HTTPException(status_code=500, detail=f"Internal classification engine error: {str(execution_exception)}")
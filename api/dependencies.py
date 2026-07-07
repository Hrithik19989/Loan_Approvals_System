# api/dependencies.py
"""
Clean Dependency Injection provider decoupling asset loading sequences 
from active network transmission pipeline protocols.
"""

import os
import joblib
import shap
from utils.logger import get_production_logger

runtime_logger = get_production_logger()

class ModelAssetProvider:
    """Manages the in-memory lifetime and access states of core ML pipelines and explainers."""
    def __init__(self):
        self.pipeline = None
        self.explainer = None

    def bootstrap_artifacts(self, artifact_path: str = "models/loan_model_pipeline.joblib"):
        """Reads serialization files from disk and sets up the SHAP TreeExplainer wrapper."""
        if self.pipeline is None:
            try:
                runtime_logger.info(f"Validating physical path trace target: {artifact_path}")
                if not os.path.exists(artifact_path):
                    raise FileNotFoundError(f"Missing serialization target artifact at path location: {artifact_path}")

                runtime_logger.info("De-serializing packaged joblib training layout...")
                self.pipeline = joblib.load(artifact_path)
                
                runtime_logger.info("Isolating underlying structural estimator weights to build SHAP instance...")
                underlying_estimator = self.pipeline.named_steps['estimator']
                self.explainer = shap.TreeExplainer(underlying_estimator)
                
                runtime_logger.info("Production microservice memory artifacts successfully mapped and live.")
            except Exception as exc:
                runtime_logger.critical(f"FATAL SYSTEM ERROR: Memory registration loop halted. Trace: {str(exc)}")
                raise exc

    def get_pipeline(self):
        return self.pipeline

    def get_explainer(self):
        return self.explainer

# Instantiated single reference resource instance to keep runtime states isolated
model_provider = ModelAssetProvider()

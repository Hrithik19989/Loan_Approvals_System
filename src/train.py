# src/train.py
"""
Model training and hyperparameter optimization suite with MLflow experiment tracking.
Compares Logistic Regression, Random Forest, and XGBoost models.
"""

import os
import warnings
import multiprocessing
from sklearn.base import clone

# Suppress annoying convergence and deprecation warnings across dependencies
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

from src.preprocessing import ProductionDataPreprocessor
from src.features import RiskFeatureGenerator
from utils.logger import get_production_logger

logger = get_production_logger()

def train_and_optimize_system(data_path: str):
    logger.info("Initializing baseline data ingestion pipeline...")
    if not os.path.exists(data_path):
        logger.critical(f"Data target file missing: {data_path}")
        raise FileNotFoundError(f"File not found at target: {data_path}")

    # 1. Load Data
    df = pd.read_csv(data_path)
    target = 'Risk_Flag' if 'Risk_Flag' in df.columns else 'loan_status'
    
    X = df.drop(columns=[target])
    y = df[target]

    # Stratified Split to maintain risk profile balance
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Splits extracted successfully. Training records: {X_train.shape}")

    # Calculate class weights to solve the 0.0000 F1 score issue
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight_value = neg_count / max(1, pos_count)
    logger.info(f"Class imbalance detected. Negative: {neg_count}, Positive: {pos_count}. Scale weight: {scale_pos_weight_value:.2f}")

    # Set up MLflow Workspace
    mlflow.set_experiment("Loan_Underwriting_Risk_Optimization")

    # Define features for engineering transforms
    categorical_cols = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE']
    numeric_cols = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS',
                    'career_stability_ratio', 'income_per_year_experience', 'residence_volatility_index', 'leverage_asset_score']

    # Preprocessing Pipeline setup
    core_processing = Pipeline([
        ('clean_impute_clip', ProductionDataPreprocessor()),
        ('engineer_features', RiskFeatureGenerator()),
        ('encode_scale', ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.01, sparse_output=True), categorical_cols)
        ], remainder='passthrough'))
    ])

    # Fit and transform data once before tuning to prevent redundant preprocessing inside CV loops
    logger.info("Pre-transforming feature space to optimize grid search computation time...")
    X_train_transformed = core_processing.fit_transform(X_train)
    X_val_transformed = core_processing.transform(X_val)

    # Dictionary of algorithms to evaluate
    models_to_evaluate = {
        "Logistic_Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'), 
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0] 
            }
        },
        "Random_Forest": {
            "estimator": RandomForestClassifier(random_state=42, class_weight='balanced'), 
            "params": {
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 12, 20], 
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [2, 4]
            }
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                random_state=42, 
                eval_metric='logloss', 
                tree_method='hist',
                scale_pos_weight=scale_pos_weight_value,
                n_jobs=1  
            ), 
            "params": {
                "n_estimators": [50, 100, 150],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        }
    }

    best_overall_score = -1.0
    best_pipeline = None  
    os.makedirs("models", exist_ok=True)

    # Define the custom types that MLflow/skops must trust
    trusted_types = [
        "src.features.RiskFeatureGenerator",
        "src.preprocessing.MissingValueImputer",
        "src.preprocessing.OutlierClipper",
        "src.preprocessing.ProductionDataPreprocessor",
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier"
    ]

    # Evaluate each model configuration
    for name, config in models_to_evaluate.items():
        logger.info(f"Starting lifecycle execution loop for algorithm: {name}")
        
        with mlflow.start_run(run_name=name, nested=True):
            searcher = RandomizedSearchCV(
                config["estimator"], 
                param_distributions=config["params"],
                n_iter=4, 
                cv=3, 
                scoring='f1', 
                random_state=42, 
                n_jobs=-1,
                pre_dispatch='2*n_jobs' 
            )
            searcher.fit(X_train_transformed, y_train)

            # Log model parameters and Cross-Validation scores
            mlflow.log_params(searcher.best_params_)
            mlflow.log_metric("best_cv_f1_score", searcher.best_score_)

            # FIXED: Create a fresh clone of the preprocessing steps to avoid tracking pre-fit states
            fresh_processor = clone(core_processing)
            fresh_processor.steps[0][1].__dict__.update(core_processing.steps[0][1].__dict__)
            fresh_processor.steps[1][1].__dict__.update(core_processing.steps[1][1].__dict__)
            fresh_processor.steps[2][1].__dict__.update(core_processing.steps[2][1].__dict__)

            # Build production deployment pipeline containing fully setup components
            deployment_pipeline = Pipeline([
                ('processor', fresh_processor),
                ('estimator', searcher.best_estimator_)
            ])
            
            # Pass trusted types list and parameter name
            mlflow.sklearn.log_model(
                deployment_pipeline, 
                name="model",
                skops_trusted_types=trusted_types
            )

            # Extract, save, and log Cross-Validation tabular statistics
            cv_results = pd.DataFrame(searcher.cv_results_)
            cv_csv_path = f"models/{name}_cv_results.csv"
            cv_results.to_csv(cv_csv_path, index=False)
            mlflow.log_artifact(cv_csv_path)

            # Compute holdout validation metrics using optimized data matrix
            y_val_pred = searcher.best_estimator_.predict(X_val_transformed)
            val_f1 = f1_score(y_val, y_val_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            mlflow.log_metric("validation_f1", val_f1)
            mlflow.log_metric("validation_accuracy", val_acc)
            
            logger.info(f"Algorithm {name} finished. CV F1: {searcher.best_score_:.4f} | Validation F1: {val_f1:.4f}")

            # Track the overall best performing champion model
            if val_f1 > best_overall_score:
                best_overall_score = val_f1
                best_pipeline = deployment_pipeline

    # Serialize and save the final best model artifact to disk via joblib
    model_output_path = "models/loan_model_pipeline.joblib"
    joblib.dump(best_pipeline, model_output_path)
    
    # Log the champion model deployment metadata
    with mlflow.start_run(run_name="Champion_Model_Export"):
        mlflow.sklearn.log_model(
            best_pipeline, 
            name="champion_loan_pipeline",
            skops_trusted_types=trusted_types
        )
        mlflow.log_metric("champion_validation_f1", best_overall_score)
        
    logger.info(f"Champion deployment model successfully saved to: {model_output_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train_and_optimize_system("data/raw/Loan Prediction.csv")
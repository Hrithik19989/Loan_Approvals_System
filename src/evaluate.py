# src/evaluate.py
"""
Model validation evaluation suite. Computes classification accuracy matrices,
ROC curves, and extracts global feature explanations using SHAP.
Handles sparse matrix structures cleanly to avoid memory footprint failures.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from utils.logger import get_production_logger

logger = get_production_logger()

def evaluate_and_explain_system(data_path: str, model_path: str, output_dir: str = "models/"):
    logger.info("Initializing diagnostic model evaluation validation run...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data and Model
    df = pd.read_csv(data_path)
    target = 'Risk_Flag' if 'Risk_Flag' in df.columns else 'loan_status'
    X = df.drop(columns=[target])
    y = df[target]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if not os.path.exists(model_path):
        logger.error(f"Evaluation failed: Missing pipeline file at {model_path}")
        return

    pipeline = joblib.load(model_path)
    
    # 2. Generate Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 3. Calculate Performance Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC Score": roc_auc_score(y_test, y_prob)
    }

    logger.info("--- Validation Split Performance Metrics ---")
    for metric, score in metrics.items():
        logger.info(f"{metric}: {score:.4f}")

    # 4. Generate Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Approved", "Rejected"])
    cm_display.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.title("Underwriting Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # 5. Generate ROC Curve Plot
    plt.figure(figsize=(6, 5))
    roc_display = RocCurveDisplay.from_predictions(y_test, y_prob, ax=plt.gca())
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline Chance')
    plt.title("Underwriting Risk ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=200)
    plt.close()

    # 6. Global Model Explainability via SHAP TreeExplainer
    logger.info("Computing global feature attributions via SHAP TreeExplainer...")
    processor = pipeline.named_steps['processor']
    estimator = pipeline.named_steps['estimator']

    # CRITICAL SECURITY FIX: Subsample the evaluation array to a manageable size
    # Generating SHAP calculations over all 50,000+ test records on high-cardinality arrays causes out-of-memory errors
    X_test_sample = X_test.head(100)

    # Transform test features to match the model input shape
    X_test_transformed = processor.transform(X_test_sample)
    
    # Safely convert compressed sparse column format back to a dense layout for text mapping
    if hasattr(X_test_transformed, 'toarray'):
        X_test_transformed = X_test_transformed.toarray()
    
    # Dynamically extract labels from the internal transformation pipelines
    transformer_layer = processor.named_steps['encode_scale']
    if hasattr(transformer_layer, 'get_feature_names_out'):
        feature_names = transformer_layer.get_feature_names_out()
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)

    # Compute SHAP Values
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer(X_test_transformed)

    # Generate SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_transformed, show=False)
    plt.title("SHAP Global Risk Feature Attribution Analysis", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"), dpi=250)
    plt.close()
    
    logger.info("Evaluation complete. Performance plots and SHAP charts saved to the models/ directory.")

if __name__ == "__main__":
    evaluate_and_explain_system("data/raw/Loan Prediction.csv", "models/loan_model_pipeline.joblib")

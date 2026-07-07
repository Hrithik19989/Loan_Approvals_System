# src/preprocessing.py
"""
Data preprocessing utilities for the Ethicalstar Loan Prediction dataset.

This module provides modular, custom Scikit-Learn transformers to:
1. Handle dynamic numerical and categorical missing values.
2. Clip extreme outliers using Interquartile Range (IQR) bounds.
3. Keep pipeline structures clear to prevent downstream data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values securely across data splits.
    
    Numerical features are filled via the median.
    Categorical features are filled via the training split mode.
    """
    def __init__(self):
        self.numeric_fill_values = {}
        self.categorical_fill_values = {}

    def fit(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()
        
        # Isolate datatypes to dynamically build fallback registries
        numeric_cols = X_copy.select_dtypes(include=np.number).columns
        categorical_cols = X_copy.select_dtypes(exclude=np.number).columns

        for col in numeric_cols:
            self.numeric_fill_values[col] = float(X_copy[col].median())

        for col in categorical_cols:
            mode_series = X_copy[col].mode()
            if not mode_series.empty:
                self.categorical_fill_values[col] = str(mode_series[0])
            else:
                self.categorical_fill_values[col] = "unknown"
                
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        # Clean structural record identifiers if present during training runs
        if 'Id' in X_copy.columns:
            X_copy = X_copy.drop(columns=['Id'])

        for col, value in self.numeric_fill_values.items():
            if col in X_copy.columns:
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(value)

        for col, value in self.categorical_fill_values.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].astype(str).str.strip().str.lower().fillna(value)
                
        return X_copy


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips extreme outliers based on the statistical IQR rule.
    
    Upper and lower mathematical bounds are derived solely from training data.
    """
    def __init__(self, factor: float = 3.0):
        self.factor = factor
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()
        numeric_cols = X_copy.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            q1 = X_copy[col].quantile(0.25)
            q3 = X_copy[col].quantile(0.75)
            iqr = q3 - q1

            # Store bounds to handle incoming test/production payloads
            self.lower_bounds[col] = q1 - (self.factor * iqr)
            self.upper_bounds[col] = q3 + (self.factor * iqr)
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        for col in self.lower_bounds:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].clip(
                    lower=max(0.0, self.lower_bounds[col]),  # Financial bounds cannot fall beneath 0
                    upper=self.upper_bounds[col]
                )
                
        return X_copy


class ProductionDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Compound preprocessor that unifies imputation and outlier clipping rules 
    into a single step for inference engines.
    """
    def __init__(self, iqr_multiplier: float = 3.0):
        self.iqr_multiplier = iqr_multiplier
        self.imputer = MissingValueImputer()
        self.clipper = OutlierClipper(factor=self.iqr_multiplier)

    def fit(self, X: pd.DataFrame, y=None):
        X_transformed = self.imputer.fit_transform(X, y)
        self.clipper.fit(X_transformed, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imputed = self.imputer.transform(X)
        X_clipped = self.clipper.transform(X_imputed)
        return X_clipped

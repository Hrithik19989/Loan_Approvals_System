# src/features.py
"""
Feature engineering utilities for the Ethicalstar Loan Prediction dataset.

Generates structured credit underwriting risk signals and stability profiles.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RiskFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Derives professional asset profiles and demographic stability indices 
    safely mapped to the Ethicalstar schema configurations.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # 1. Professional Career Stability Ratio (Tenure vs Overall Experience)
        if {"CURRENT_JOB_YRS", "Experience"}.issubset(df.columns):
            exp_safe = df['Experience'].replace(0, 1.0)
            df['career_stability_ratio'] = df['CURRENT_JOB_YRS'] / exp_safe
        else:
            df['career_stability_ratio'] = 0.0

        # 2. Economic Velocity Index (Income Generation per Year of Experience)
        if {"Income", "Experience"}.issubset(df.columns):
            exp_safe = df['Experience'].replace(0, 1.0)
            df['income_per_year_experience'] = df['Income'] / exp_safe
        else:
            df['income_per_year_experience'] = 0.0
        
        # 3. Residential Mobility/Volatility Flag
        if {"CURRENT_HOUSE_YRS", "CURRENT_JOB_YRS"}.issubset(df.columns):
            job_safe = df['CURRENT_JOB_YRS'].replace(0, 1.0)
            df['residence_volatility_index'] = df['CURRENT_HOUSE_YRS'] / job_safe
        else:
            df['residence_volatility_index'] = 0.0
        
        # 4. Total Leveraged Wealth Factor (Categorical to Numerical Indicator)
        if {"Car_Ownership", "House_Ownership"}.issubset(df.columns):
            # Safe string normalization mapping
            car_numeric = df['Car_Ownership'].astype(str).str.strip().str.lower().eq('yes').astype(int)
            home_numeric = df['House_Ownership'].astype(str).str.strip().str.lower().eq('owned').astype(int)
            df['leverage_asset_score'] = car_numeric + home_numeric
        else:
            df['leverage_asset_score'] = 0.0

        return df

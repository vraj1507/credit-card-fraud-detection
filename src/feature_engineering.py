"""
Feature engineering: log transforms, cyclical time, interaction features.
"""
import numpy as np
import pandas as pd
from src.config import AMOUNT_LOG_COL, TIME_SIN_COL, TIME_COS_COL, SECONDS_IN_DAY


def engineer_features(df):
    """Add engineered features to a DataFrame (works on train or test)."""
    df = df.copy()

    # 1. Log-transformed amount (handles skewness)
    df[AMOUNT_LOG_COL] = np.log1p(df["Amount"])

    # 2. Cyclical time-of-day encoding
    time_of_day = df["Time"] % SECONDS_IN_DAY
    df[TIME_SIN_COL] = np.sin(2 * np.pi * time_of_day / SECONDS_IN_DAY)
    df[TIME_COS_COL] = np.cos(2 * np.pi * time_of_day / SECONDS_IN_DAY)

    # 3. Interaction features — top PCA components × Amount
    for v in ["V1", "V2", "V3", "V4"]:
        df[f"{v}_x_Amount"] = df[v] * df["Amount"]

    # 4. Statistical aggregates of PCA features
    pca_cols = [f"V{i}" for i in range(1, 29)]
    df["V_mean"] = df[pca_cols].mean(axis=1)
    df["V_std"] = df[pca_cols].std(axis=1)
    df["V_skew"] = df[pca_cols].skew(axis=1)
    df["V_kurtosis"] = df[pca_cols].kurtosis(axis=1)

    # 5. High-magnitude flag (transactions with extreme PCA values)
    df["V_max_abs"] = df[pca_cols].abs().max(axis=1)

    return df

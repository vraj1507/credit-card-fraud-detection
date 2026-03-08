"""
Tests for data preparation module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_preparation import get_preprocessor
from src.feature_engineering import engineer_features


@pytest.fixture
def sample_dataframe():
    """Create a small synthetic DataFrame mimicking the credit card dataset."""
    np.random.seed(42)
    n = 100
    data = {"Time": np.random.uniform(0, 172800, n), "Amount": np.random.exponential(50, n)}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)
    data["Class"] = np.array([0] * 95 + [1] * 5)
    return pd.DataFrame(data)


def test_preprocessor_scales_correctly():
    """Scaler should zero-mean and unit-variance the data."""
    scaler = get_preprocessor()
    X = pd.DataFrame({"Amount": [100, 200, 300], "Time": [0, 1000, 2000]})
    X_scaled = scaler.fit_transform(X)
    assert abs(X_scaled[:, 0].mean()) < 1e-10
    assert abs(X_scaled[:, 0].std(ddof=0) - 1.0) < 1e-10


def test_feature_engineering_adds_columns(sample_dataframe):
    """Feature engineering should add new columns."""
    df = sample_dataframe.drop(columns=["Class"])
    df_eng = engineer_features(df)

    expected_new = ["Amount_log", "Time_sin", "Time_cos", "V1_x_Amount",
                    "V_mean", "V_std", "V_skew", "V_kurtosis", "V_max_abs"]
    for col in expected_new:
        assert col in df_eng.columns, f"Missing column: {col}"


def test_feature_engineering_no_nulls(sample_dataframe):
    """Engineered features should not contain NaN."""
    df = sample_dataframe.drop(columns=["Class"])
    df_eng = engineer_features(df)
    assert df_eng.isnull().sum().sum() == 0


def test_log_amount_positive(sample_dataframe):
    """Log-transformed amount should always be non-negative."""
    df = sample_dataframe.drop(columns=["Class"])
    df_eng = engineer_features(df)
    assert (df_eng["Amount_log"] >= 0).all()

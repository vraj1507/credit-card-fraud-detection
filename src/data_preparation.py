"""
Data loading, validation, and train/test splitting.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import RAW_DATA_PATH, TARGET_COL, TEST_SIZE, RANDOM_STATE


def load_data(path=None):
    """Load raw CSV and perform basic validation."""
    path = path or RAW_DATA_PATH
    df = pd.read_csv(path)

    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found"
    assert df[TARGET_COL].isin([0, 1]).all(), "Target must be binary (0/1)"
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"

    print(f"Loaded {len(df):,} transactions | Frauds: {df[TARGET_COL].sum():,} ({df[TARGET_COL].mean()*100:.3f}%)")
    return df


def split_data(df):
    """Stratified train/test split."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train fraud rate: {y_train.mean()*100:.3f}% | Test fraud rate: {y_test.mean()*100:.3f}%")
    return X_train, X_test, y_train, y_test


def get_preprocessor():
    """Return a StandardScaler for Amount and Time."""
    return StandardScaler()


def preprocess(X_train, X_test):
    """Scale Amount and Time features; return transformed DataFrames and fitted scaler."""
    scaler = get_preprocessor()
    cols_to_scale = ["Amount", "Time"]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train, X_test, scaler

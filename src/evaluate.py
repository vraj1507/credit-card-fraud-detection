"""
Standalone evaluation script — loads saved model and generates metrics report.
"""
import json
import joblib
import pandas as pd
from src.config import API_MODEL_PATH, API_PREPROCESSOR_PATH, API_THRESHOLD_PATH
from src.data_preparation import load_data, split_data
from src.feature_engineering import engineer_features
from src.model import evaluate_model


def main():
    """Load saved model and evaluate on test set."""
    model = joblib.load(API_MODEL_PATH)
    scaler = joblib.load(API_PREPROCESSOR_PATH)
    with open(API_THRESHOLD_PATH, "r") as f:
        config = json.load(f)
    threshold = config["threshold"]

    df = load_data()
    _, X_test, _, y_test = split_data(df)

    cols_to_scale = ["Amount", "Time"]
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    X_test = engineer_features(X_test)

    metrics, _, _, _ = evaluate_model(model, X_test, y_test, threshold)
    print(f"\nAll metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()

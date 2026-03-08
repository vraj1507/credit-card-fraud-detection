"""
Tests for model training and evaluation utilities.
"""
import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification
from src.model import find_optimal_threshold, evaluate_model, apply_resampling


@pytest.fixture
def imbalanced_dataset():
    """Create a synthetic imbalanced binary classification dataset."""
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, weights=[0.95, 0.05], random_state=42
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(20)]), pd.Series(y)


def test_resampling_increases_minority(imbalanced_dataset):
    """SMOTE+Tomek should increase the minority class count."""
    X, y = imbalanced_dataset
    minority_before = (y == 1).sum()
    X_res, y_res = apply_resampling(X, y)
    minority_after = (y_res == 1).sum()
    assert minority_after > minority_before


def test_optimal_threshold_in_range():
    """Optimal threshold should be between 0 and 1."""
    np.random.seed(42)
    y_true = np.array([0]*950 + [1]*50)
    y_prob = np.random.rand(1000)
    threshold = find_optimal_threshold(y_true, y_prob)
    assert 0.0 <= threshold <= 1.0


def test_evaluate_returns_all_metrics(imbalanced_dataset):
    """Evaluate should return all expected metric keys."""
    X, y = imbalanced_dataset
    model = xgb.XGBClassifier(n_estimators=10, random_state=42, verbosity=0)
    model.fit(X, y)
    metrics, _, _, cm = evaluate_model(model, X, y, threshold=0.5)

    expected_keys = ["pr_auc", "roc_auc", "precision", "recall", "f1_score", "fbeta_score", "threshold"]
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
    assert cm.shape == (2, 2)

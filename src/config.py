"""
Central configuration for the fraud detection pipeline.
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"

# Data
RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
TARGET_COL = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature Engineering
AMOUNT_LOG_COL = "Amount_log"
TIME_SIN_COL = "Time_sin"
TIME_COS_COL = "Time_cos"
SECONDS_IN_DAY = 86400

# Resampling
SMOTE_SAMPLING_STRATEGY = 0.5  # Ratio of minority to majority after SMOTE
SMOTE_K_NEIGHBORS = 5

# XGBoost defaults (Optuna will search around these)
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

# Optuna
OPTUNA_N_TRIALS = 50
OPTUNA_CV_FOLDS = 5

# Threshold tuning
FBETA_BETA = 2  # Emphasize recall

# MLflow
MLFLOW_EXPERIMENT_NAME = "credit-card-fraud-detection"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")

# API
API_MODEL_PATH = MODELS_DIR / "best_model.joblib"
API_PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
API_THRESHOLD_PATH = MODELS_DIR / "optimal_threshold.json"

"""
Model training, Optuna hyperparameter optimization, threshold tuning, and evaluation.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    fbeta_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from src.config import (
    XGBOOST_PARAMS, OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS,
    RANDOM_STATE, SMOTE_SAMPLING_STRATEGY, SMOTE_K_NEIGHBORS, FBETA_BETA
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def apply_resampling(X_train, y_train):
    """Apply SMOTE + Tomek links to balance the training set."""
    smote_tomek = SMOTETomek(
        smote=SMOTE(
            sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=RANDOM_STATE,
        ),
        random_state=RANDOM_STATE,
    )
    X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
    print(f"Resampled: {len(X_res):,} samples | Fraud ratio: {y_res.mean()*100:.2f}%")
    return X_res, y_res


def objective(trial, X, y, scale_pos_weight):
    """Optuna objective: cross-validated PR-AUC."""
    params = {
        **XGBOOST_PARAMS,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
    }

    cv = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pr_aucs = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        pr_aucs.append(average_precision_score(y_val, y_prob))

    return np.mean(pr_aucs)


def optimize_hyperparameters(X_train, y_train):
    """Run Optuna study and return best parameters."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, scale_pos_weight),
        n_trials=OPTUNA_N_TRIALS,
        show_progress_bar=True,
    )

    best_params = {**XGBOOST_PARAMS, **study.best_params, "scale_pos_weight": scale_pos_weight}
    print(f"Best PR-AUC (CV): {study.best_value:.4f}")
    return best_params, study


def train_model(X_train, y_train, params):
    """Train final XGBoost model with given parameters."""
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


def find_optimal_threshold(y_true, y_prob, beta=FBETA_BETA):
    """Find threshold that maximizes F-beta score on the precision-recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    f_betas = []
    for p, r in zip(precisions[:-1], recalls[:-1]):
        if p + r > 0:
            fb = (1 + beta**2) * (p * r) / ((beta**2 * p) + r)
        else:
            fb = 0.0
        f_betas.append(fb)

    best_idx = np.argmax(f_betas)
    best_threshold = float(thresholds[best_idx])
    print(f"Optimal threshold: {best_threshold:.4f} (F-beta={f_betas[best_idx]:.4f})")
    return best_threshold


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Compute all evaluation metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "pr_auc": average_precision_score(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "fbeta_score": fbeta_score(y_test, y_pred, beta=FBETA_BETA),
        "threshold": threshold,
    }

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:>15s}: {v:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    return metrics, y_prob, y_pred, cm

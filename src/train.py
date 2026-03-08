"""
Main training pipeline — orchestrates data loading, feature engineering,
resampling, model training, evaluation, and artifact saving.
"""
import json
import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from pathlib import Path

from src.config import (
    MODELS_DIR, REPORTS_DIR, MLFLOW_EXPERIMENT_NAME,
    API_MODEL_PATH, API_PREPROCESSOR_PATH, API_THRESHOLD_PATH,
)
from src.data_preparation import load_data, split_data, preprocess
from src.feature_engineering import engineer_features
from src.model import (
    apply_resampling, optimize_hyperparameters,
    train_model, find_optimal_threshold, evaluate_model,
)
from src.explainability import generate_shap_explanations


def plot_class_distribution(y, save_path):
    """Plot and save class distribution."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    counts = y.value_counts()
    bars = ax.bar(["Legitimate", "Fraud"], [counts[0], counts[1]], 
                  color=["#2ecc71", "#e74c3c"], edgecolor="black")
    for bar, count in zip(bars, [counts[0], counts[1]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{count:,}\n({count/len(y)*100:.3f}%)", ha="center", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Transaction Class Distribution")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_recall(y_test, y_prob, threshold, save_path):
    """Plot precision-recall curve with optimal threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, linewidth=2, color="#3498db")

    # Mark optimal threshold
    idx = np.argmin(np.abs(thresholds - threshold))
    ax.plot(recalls[idx], precisions[idx], "ro", markersize=12, 
            label=f"Optimal threshold={threshold:.3f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc(y_test, y_prob, roc_auc, save_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, color="#e74c3c", label=f"XGBoost (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"], ax=ax,
                annot_kws={"size": 14})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Execute the full training pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- MLflow setup ---
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost-optuna-pipeline"):

        # 1. Load data
        print("\n[1/8] Loading data...")
        df = load_data()

        # 2. Split
        print("\n[2/8] Splitting data...")
        X_train, X_test, y_train, y_test = split_data(df)

        # Save class distribution plot
        plot_class_distribution(y_train, REPORTS_DIR / "class_distribution.png")

        # 3. Preprocess
        print("\n[3/8] Preprocessing...")
        X_train, X_test, scaler = preprocess(X_train, X_test)

        # 4. Feature engineering
        print("\n[4/8] Engineering features...")
        X_train = engineer_features(X_train)
        X_test = engineer_features(X_test)

        # Log feature names
        feature_names = X_train.columns.tolist()
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("features", str(feature_names[:10]) + "...")

        # 5. Resampling
        print("\n[5/8] Applying SMOTE+Tomek resampling...")
        X_train_res, y_train_res = apply_resampling(X_train, y_train)
        mlflow.log_param("resampling", "SMOTETomek")
        mlflow.log_param("train_size_after_resampling", len(X_train_res))

        # 6. Hyperparameter optimization
        print("\n[6/8] Optimizing hyperparameters with Optuna...")
        best_params, study = optimize_hyperparameters(X_train_res, y_train_res)

        for k, v in best_params.items():
            mlflow.log_param(f"xgb_{k}", v)

        # 7. Train final model
        print("\n[7/8] Training final model...")
        model = train_model(X_train_res, y_train_res, best_params)

        # 8. Evaluate
        print("\n[8/8] Evaluating model...")
        y_prob = model.predict_proba(X_test)[:, 1]

        # Optimal threshold
        optimal_threshold = find_optimal_threshold(y_test, y_prob)

        # Full evaluation
        metrics, y_prob, y_pred, cm = evaluate_model(model, X_test, y_test, optimal_threshold)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Generate plots
        plot_precision_recall(y_test, y_prob, optimal_threshold, REPORTS_DIR / "precision_recall_curve.png")
        plot_roc(y_test, y_prob, metrics["roc_auc"], REPORTS_DIR / "roc_curve.png")
        plot_confusion_matrix(cm, REPORTS_DIR / "confusion_matrix.png")

        # SHAP explanations
        print("\nGenerating SHAP explanations...")
        shap_values = generate_shap_explanations(model, X_test.iloc[:1000])

        # Log artifacts
        mlflow.log_artifacts(str(REPORTS_DIR), artifact_path="figures")

        # Save model and preprocessor
        joblib.dump(model, API_MODEL_PATH)
        joblib.dump(scaler, API_PREPROCESSOR_PATH)
        with open(API_THRESHOLD_PATH, "w") as f:
            json.dump({"threshold": optimal_threshold, "features": feature_names}, f, indent=2)

        mlflow.xgboost.log_model(model, artifact_path="xgboost-model")

        print(f"\n{'='*50}")
        print("TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"Model saved to: {API_MODEL_PATH}")
        print(f"Preprocessor saved to: {API_PREPROCESSOR_PATH}")
        print(f"Threshold config saved to: {API_THRESHOLD_PATH}")
        print(f"Plots saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()

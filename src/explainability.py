"""
SHAP-based model explainability and visualization generation.
"""
import shap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import REPORTS_DIR


def generate_shap_explanations(model, X_test, save_dir=None):
    """Generate and save SHAP plots."""
    save_dir = Path(save_dir or REPORTS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 1. Summary plot (bar)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (Top 20)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Summary plot (beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.title("SHAP Feature Impact (Beeswarm)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Waterfall plot for a fraud sample
    fraud_indices = X_test.index[X_test.index.isin(
        X_test.index[:len(X_test)]  # Use first fraud found
    )]
    if len(fraud_indices) > 0:
        sample_idx = 0
        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[sample_idx].values,
            feature_names=X_test.columns.tolist(),
        )
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title("SHAP Waterfall — Sample Transaction", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"SHAP plots saved to {save_dir}")
    return shap_values

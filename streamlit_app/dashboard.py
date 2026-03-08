"""
Interactive Streamlit dashboard for fraud analysis.
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="💳", layout="wide")

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


@st.cache_resource
def load_artifacts():
    """Load model, scaler, and threshold."""
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    scaler = joblib.load(MODELS_DIR / "preprocessor.joblib")
    with open(MODELS_DIR / "optimal_threshold.json", "r") as f:
        config = json.load(f)
    return model, scaler, config


@st.cache_data
def load_sample_data():
    """Load sample of raw data for exploration."""
    path = DATA_DIR / "creditcard.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df
    return None


def main():
    st.title("💳 Credit Card Fraud Detection Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["📊 Data Explorer", "🔍 Single Prediction", "📈 Model Performance"])

    try:
        model, scaler, config = load_artifacts()
        model_loaded = True
    except Exception:
        model_loaded = False
        st.sidebar.warning("⚠️ Model not loaded. Run training first.")

    if page == "📊 Data Explorer":
        st.header("📊 Data Explorer")
        df = load_sample_data()
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", f"{len(df):,}")
            col2.metric("Fraudulent", f"{df['Class'].sum():,}")
            col3.metric("Legitimate", f"{(df['Class'] == 0).sum():,}")
            col4.metric("Fraud Rate", f"{df['Class'].mean()*100:.3f}%")

            st.subheader("Class Distribution")
            fig = px.histogram(df, x="Class", color="Class",
                             color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                             labels={"Class": "Transaction Class"},
                             title="Legitimate (0) vs Fraud (1)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Transaction Amount Distribution")
            fig2 = px.box(df, x="Class", y="Amount", color="Class",
                         color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                         title="Amount by Class")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Sample Data")
            st.dataframe(df.head(100))
        else:
            st.info("Place `creditcard.csv` in the `data/` directory to explore.")

    elif page == "🔍 Single Prediction":
        st.header("🔍 Real-Time Fraud Prediction")
        if not model_loaded:
            st.error("Model not loaded. Please run the training pipeline first.")
            return

        st.markdown("Enter transaction features below:")

        col1, col2 = st.columns(2)
        with col1:
            time_val = st.number_input("Time (seconds)", value=406.0)
            amount_val = st.number_input("Amount ($)", value=529.0, min_value=0.0)

        with col2:
            st.markdown("**PCA Features (V1-V28)**")
            v_values = {}
            for i in range(1, 29):
                v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v{i}")

        if st.button("🔍 Predict", type="primary"):
            from src.feature_engineering import engineer_features

            input_data = {"Time": time_val, "Amount": amount_val, **v_values}
            df_input = pd.DataFrame([input_data])
            df_input[["Amount", "Time"]] = scaler.transform(df_input[["Amount", "Time"]])
            df_input = engineer_features(df_input)
            df_input = df_input[config["features"]]

            prob = model.predict_proba(df_input)[:, 1][0]
            threshold = config["threshold"]
            is_fraud = prob >= threshold

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Fraud Probability", f"{prob:.4f}")
            col2.metric("Prediction", "🚨 FRAUD" if is_fraud else "✅ LEGITIMATE")
            col3.metric("Threshold", f"{threshold:.4f}")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Fraud Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c" if is_fraud else "#2ecc71"},
                    "steps": [
                        {"range": [0, 10], "color": "#d5f5e3"},
                        {"range": [10, 50], "color": "#fdebd0"},
                        {"range": [50, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": threshold * 100,
                    },
                },
            ))
            st.plotly_chart(fig, use_container_width=True)

    elif page == "📈 Model Performance":
        st.header("📈 Model Performance")
        reports_dir = PROJECT_ROOT / "reports" / "figures"

        plots = {
            "Confusion Matrix": "confusion_matrix.png",
            "ROC Curve": "roc_curve.png",
            "Precision-Recall Curve": "precision_recall_curve.png",
            "SHAP Summary": "shap_summary.png",
            "SHAP Feature Importance": "shap_summary_bar.png",
            "Class Distribution": "class_distribution.png",
        }

        cols = st.columns(2)
        for idx, (title, filename) in enumerate(plots.items()):
            path = reports_dir / filename
            if path.exists():
                with cols[idx % 2]:
                    st.subheader(title)
                    st.image(str(path))
            else:
                with cols[idx % 2]:
                    st.info(f"{title}: Run training to generate this plot.")


if __name__ == "__main__":
    main()

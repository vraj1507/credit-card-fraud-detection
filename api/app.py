"""
FastAPI prediction API for credit card fraud detection.
"""
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from api.schemas import TransactionRequest, PredictionResponse, HealthResponse
from src.config import API_MODEL_PATH, API_PREPROCESSOR_PATH, API_THRESHOLD_PATH
from src.feature_engineering import engineer_features

# Global model store
model_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup."""
    try:
        model_store["model"] = joblib.load(API_MODEL_PATH)
        model_store["scaler"] = joblib.load(API_PREPROCESSOR_PATH)
        with open(API_THRESHOLD_PATH, "r") as f:
            config = json.load(f)
        model_store["threshold"] = config["threshold"]
        model_store["features"] = config["features"]
        print("Model artifacts loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise
    yield
    model_store.clear()


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud scoring with XGBoost + SHAP explainability",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded="model" in model_store,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest):
    """Score a single transaction for fraud probability."""
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build input DataFrame
        data = transaction.model_dump()
        df = pd.DataFrame([data])

        # Scale
        cols_to_scale = ["Amount", "Time"]
        df[cols_to_scale] = model_store["scaler"].transform(df[cols_to_scale])

        # Feature engineering
        df = engineer_features(df)

        # Ensure correct feature order
        df = df[model_store["features"]]

        # Predict
        fraud_probability = float(model_store["model"].predict_proba(df)[:, 1][0])
        threshold = model_store["threshold"]
        is_fraud = fraud_probability >= threshold

        risk_level = (
            "CRITICAL" if fraud_probability >= 0.9 else
            "HIGH" if fraud_probability >= 0.7 else
            "MEDIUM" if fraud_probability >= threshold else
            "LOW" if fraud_probability >= 0.1 else
            "MINIMAL"
        )

        return PredictionResponse(
            fraud_probability=round(fraud_probability, 6),
            is_fraud=is_fraud,
            risk_level=risk_level,
            threshold_used=threshold,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

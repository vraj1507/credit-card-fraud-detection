"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    """Input schema for a single transaction."""
    Time: float = Field(..., description="Seconds elapsed from first transaction in dataset")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = Field(..., description="Transaction amount", ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Time": 406.0,
                "V1": -2.3122, "V2": 1.9519, "V3": -1.6085, "V4": 3.9979,
                "V5": -0.5222, "V6": -1.4265, "V7": -2.5372, "V8": 1.3916,
                "V9": -2.7701, "V10": -2.7723, "V11": 3.2020, "V12": -2.8990,
                "V13": -0.5952, "V14": -7.2128, "V15": 0.2680, "V16": -1.4778,
                "V17": -3.4499, "V18": -1.4630, "V19": 1.6259, "V20": 0.5654,
                "V21": 1.0389, "V22": 0.3853, "V23": -0.1199, "V24": -0.1151,
                "V25": 0.3700, "V26": -0.2038, "V27": 0.6679, "V28": 0.3373,
                "Amount": 529.0
            }]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for fraud prediction."""
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    risk_level: str
    threshold_used: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool

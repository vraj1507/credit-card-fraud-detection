"""
Tests for the FastAPI prediction API.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


def test_health_check():
    """Health endpoint should return 200."""
    from api.app import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_schema_validation():
    """Predict endpoint should reject invalid input."""
    from api.app import app
    client = TestClient(app)
    # Missing required fields
    response = client.post("/predict", json={"Time": 100})
    assert response.status_code == 422  # Validation error

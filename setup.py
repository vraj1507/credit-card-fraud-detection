from setuptools import setup, find_packages

setup(
    name="credit-card-fraud-detection",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    author="Your Name",
    description="Production-grade credit card fraud detection with XGBoost, SHAP, MLflow, and FastAPI",
)

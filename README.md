💳 Credit Card Fraud Detection — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![SHAP](https://img.shields.io-badge/SHAP-Explainability-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-teal)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-grade credit card fraud detection system built with XGBoost, SHAP explainability, MLflow experiment tracking, FastAPI model serving, and an interactive Streamlit dashboard. Designed to demonstrate end-to-end machine learning engineering skills — from EDA to deployment.

---

## 🔎 Highlights

- Handles extreme class imbalance (0.173% fraud) with SMOTE + Tomek links and threshold tuning.
- XGBoost model reaches 0.878 PR-AUC and 0.980 ROC-AUC on a held-out test set.
- Full MLOps-ready stack: MLflow tracking, FastAPI service, Streamlit dashboard, pytest tests, and Dockerized deployment.

---

## 🏗️ Project Architecture

```text
credit-card-fraud-detection/
├── data/                      # Raw data (not tracked — see instructions below)
│   └── README.md
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # Full EDA + modeling notebook
├── src/
│   ├── __init__.py
│   ├── config.py              # Central config & hyperparameters
│   ├── data_preparation.py    # Data loading, splitting, preprocessing
│   ├── feature_engineering.py # Feature creation & transformations
│   ├── model.py               # Model training, tuning, evaluation
│   ├── explainability.py      # SHAP explanations & plots
│   ├── train.py               # Main training pipeline (CLI entry point)
│   └── evaluate.py            # Load saved model & generate eval report
├── api/
│   ├── app.py                 # FastAPI prediction API
│   └── schemas.py             # Pydantic request/response models
├── streamlit_app/
│   └── dashboard.py           # Interactive Streamlit fraud dashboard
├── tests/
│   ├── __init__.py
│   ├── test_data_preparation.py
│   ├── test_model.py
│   └── test_api.py
├── models/                    # Saved model artifacts (not committed)
├── reports/
│   └── figures/               # Generated plots & evaluation reports
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
├── Makefile
├── .gitignore
├── LICENSE
└── README.md

📊 Dataset
This project uses the Kaggle Credit Card Fraud Detection dataset:

284,807 transactions made by European cardholders over 2 days (Sept 2013).

492 frauds (0.172% positive class) — extreme class imbalance.

Features V1–V28 are PCA-transformed (anonymized); Amount and Time are original.

Binary target: Class (0 = legitimate, 1 = fraud).

Data Setup
Option 1: Kaggle CLI

bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
Option 2: Manual download

Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place creditcard.csv in the data/ directory.

The CSV file is intentionally excluded from git (see data/README.md and .gitignore).

🔬 Methodology
Problem Framing
Credit card fraud is a binary classification problem with severe class imbalance. A naive “predict all legitimate” classifier already achieves 99.83% accuracy, so plain accuracy is misleading. The goal is to maximize recall on fraud (catch as many fraudulent transactions as possible) while maintaining acceptable precision to avoid overwhelming analysts with false alarms.

Pipeline Steps
Data Preparation: Stratified train/test split (80/20), scaling of Amount and Time.

Feature Engineering:

Log-transformed transaction amount.

Cyclical time-of-day features from Time.

Interaction terms and transaction-velocity proxies.

Class Imbalance Handling: SMOTE + Tomek links hybrid resampling on the training set only.

Model Training: XGBoost with scale_pos_weight and Bayesian hyperparameter optimization via Optuna.

Threshold Optimization: Precision–recall curve analysis to choose an operating point that balances fraud recall and false positives.

Explainability: SHAP summary and waterfall plots, feature importance visualizations for model transparency.

Experiment Tracking: MLflow logs all parameters, metrics, and artifacts for each run.

Serving: FastAPI REST endpoint with Pydantic validation and health checks.

Dashboard: Streamlit app for interactive fraud analysis and model inspection.

🎯 Evaluation Metrics
For highly imbalanced problems, ranking and minority-class performance matter more than raw accuracy. This project tracks:

Metric	Description
PR-AUC	Primary metric; robust under class imbalance.
ROC-AUC	Overall discrimination ability.
Recall	Fraud detection rate (sensitivity).
Precision	Fraction of flagged transactions that are fraud.
F1-Score	Harmonic mean of precision and recall.
F-beta (β=2)	Weighted F-score emphasizing recall.
🚀 Quick Start
1. Clone & Setup Environment
bash
git clone https://github.com/vraj1507/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Download Data
bash
# Using Kaggle CLI (recommended)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
# Or manually place creditcard.csv in data/ (see Dataset section)
3. Run Training Pipeline
bash
python -m src.train
This will:

Load and preprocess the data.

Engineer features and apply SMOTE+Tomek resampling.

Train XGBoost with Optuna hyperparameter optimization.

Log metrics, parameters, and artifacts to MLflow.

Save the best model to models/.

Generate evaluation plots in reports/figures/.

4. Evaluate Model
bash
python -m src.evaluate
This script loads the saved model and test set, computes metrics, prints a detailed classification report, and updates plots in reports/figures/.

5. Launch API
bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
# Interactive docs at:
# http://localhost:8000/docs
6. Launch Dashboard
From the project root:

bash
export PYTHONPATH=$PWD          # on Windows: set PYTHONPATH=%cd%
streamlit run streamlit_app/dashboard.py
The app will be available at http://localhost:8502 by default.

7. Run Tests
bash
pytest tests/ -v --tb=short
8. Docker
bash
docker-compose up --build
# API:      http://localhost:8000
# Dashboard: http://localhost:8501
📈 Results
On the held-out test set, the XGBoost model achieves strong performance on the rare fraud class while maintaining near-perfect performance on legitimate transactions.

text
Loaded 284,807 transactions | Frauds: 492 (0.173%)
Train: 227,845 | Test: 56,962
Train fraud rate: 0.173% | Test fraud rate: 0.172%
Metrics (Test Set)

Metric	Score
PR-AUC	0.878
ROC-AUC	0.980
Precision	0.840
Recall	0.857
F1-Score	0.848
F-beta (β=2)	0.854
Optimal Threshold	0.733
At the optimized threshold of 0.733, the model correctly identifies about 85–86% of fraudulent transactions while keeping precision around 84%, meaning most flagged transactions are truly fraud.

📷 Key Visualizations
Plot	Description
reports/figures/class_distribution.png	Class imbalance visualization.
reports/figures/roc_curve.png	ROC curve across thresholds.
reports/figures/precision_recall_curve.png	Precision–recall curve with chosen threshold.
reports/figures/confusion_matrix.png	Confusion matrix at optimal threshold.
reports/figures/shap_summary.png	SHAP feature importance summary plot.
reports/figures/shap_waterfall.png	SHAP waterfall plot for an example fraud case.
(Remove any rows if you do not generate that specific figure.)

🛠️ Technology Stack
Category	Tools / Libraries
Language	Python 3.10+
ML Framework	XGBoost, scikit-learn
Imbalanced Learning	imbalanced-learn (SMOTE, Tomek)
Hyperparameter Tuning	Optuna
Explainability	SHAP
Experiment Tracking	MLflow
API Serving	FastAPI, Uvicorn
Dashboard	Streamlit
Testing	pytest
Containerization	Docker, Docker Compose
🔮 Future Improvements
Add anomaly detection models (Isolation Forest, Autoencoder) as ensemble members.

Implement real-time streaming with Apache Kafka.

Add model monitoring with Evidently AI for data and concept drift.

Deploy to AWS (ECR + ECS or Lambda) with CI/CD via GitHub Actions.

Incorporate geospatial and richer temporal features for enhanced detection.

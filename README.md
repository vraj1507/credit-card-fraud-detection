# 💳 Credit Card Fraud Detection — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-teal)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

A **production-grade** credit card fraud detection system built with XGBoost, SHAP explainability, MLflow experiment tracking, FastAPI model serving, and an interactive Streamlit dashboard. Designed to demonstrate end-to-end machine learning engineering skills — from EDA to deployment.

---

## 🏗️ Project Architecture

```
credit-card-fraud-detection/
├── data/                    # Raw data (not tracked — see instructions below)
│   └── README.md
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # Full EDA + modeling notebook
├── src/
│   ├── __init__.py
│   ├── config.py            # Central config & hyperparameters
│   ├── data_preparation.py  # Data loading, splitting, preprocessing
│   ├── feature_engineering.py  # Feature creation & transformations
│   ├── model.py             # Model training, tuning, evaluation
│   ├── explainability.py    # SHAP explanations & plots
│   ├── train.py             # Main training pipeline (CLI entry point)
│   └── evaluate.py          # Load saved model & generate eval report
├── api/
│   ├── app.py               # FastAPI prediction API
│   └── schemas.py           # Pydantic request/response models
├── streamlit_app/
│   └── dashboard.py         # Interactive Streamlit fraud dashboard
├── tests/
│   ├── __init__.py
│   ├── test_data_preparation.py
│   ├── test_model.py
│   └── test_api.py
├── models/                  # Saved model artifacts
├── reports/figures/          # Generated plots & evaluation reports
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
├── Makefile
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Dataset

This project uses the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset:

- **284,807 transactions** made by European cardholders over 2 days (Sept 2013)
- **492 frauds** (0.172% positive class) — extreme class imbalance
- Features `V1`–`V28` are PCA-transformed (anonymized); `Amount` and `Time` are original
- Binary target: `Class` (0 = legitimate, 1 = fraud)

### Data Setup
```bash
# Option 1: Kaggle CLI
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip

# Option 2: Manual download
# Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in the data/ directory
```

---

## 🔬 Methodology

### Problem Framing
Credit card fraud is a **binary classification** problem with severe class imbalance. Standard accuracy is misleading (a naive "predict all legitimate" classifier achieves 99.83% accuracy). We optimize for **recall** (catching fraud) while maintaining acceptable **precision** (minimizing false alarms).

### Pipeline Steps

1. **Data Preparation**: Stratified train/test split (80/20), StandardScaler on `Amount` and `Time`
2. **Feature Engineering**: Log-transformed amount, time-of-day cyclical features, interaction terms, transaction velocity proxies
3. **Class Imbalance Handling**: SMOTE + Tomek links (hybrid resampling on training set only)
4. **Model Training**: XGBoost with `scale_pos_weight`, Bayesian hyperparameter optimization via Optuna
5. **Threshold Optimization**: Precision-recall curve analysis to select optimal decision threshold
6. **Explainability**: SHAP summary plots, force plots, and feature importance for model transparency
7. **Experiment Tracking**: MLflow logs all parameters, metrics, and artifacts
8. **Serving**: FastAPI REST endpoint with input validation
9. **Dashboard**: Streamlit app for interactive fraud analysis

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **PR-AUC** | Primary metric — robust under class imbalance |
| **ROC-AUC** | Discrimination ability |
| **Recall** | Fraud detection rate (sensitivity) |
| **Precision** | Fraction of flagged transactions that are truly fraud |
| **F1-Score** | Harmonic mean of precision and recall |
| **F-beta (β=2)** | Weighted F-score emphasizing recall |

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Place creditcard.csv in data/ directory (see Dataset section above)
```

### 3. Run Training Pipeline

```bash
python -m src.train
```

This will:
- Load and preprocess the data
- Engineer features
- Apply SMOTE+Tomek resampling
- Train XGBoost with Optuna hyperparameter optimization
- Log everything to MLflow
- Save the best model to `models/`
- Generate evaluation plots in `reports/figures/`

### 4. Evaluate Model

```bash
python -m src.evaluate
```

### 5. Launch API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
# Swagger docs at http://localhost:8000/docs
```

### 6. Launch Dashboard

```bash
streamlit run streamlit_app/dashboard.py
```

### 7. Run Tests

```bash
pytest tests/ -v --tb=short
```

### 8. Docker

```bash
docker-compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## 📈 Results

*(Update after training with your data)*

| Metric | Score |
|--------|-------|
| PR-AUC | — |
| ROC-AUC | — |
| Precision | — |
| Recall | — |
| F1-Score | — |
| F-beta (β=2) | — |
| Optimal Threshold | — |

### Key Visualizations

| Plot | Description |
|------|-------------|
| `reports/figures/class_distribution.png` | Class imbalance visualization |
| `reports/figures/correlation_heatmap.png` | Feature correlations with target |
| `reports/figures/roc_curve.png` | ROC curve |
| `reports/figures/precision_recall_curve.png` | PR curve with threshold selection |
| `reports/figures/confusion_matrix.png` | Confusion matrix at optimal threshold |
| `reports/figures/shap_summary.png` | SHAP feature importance |
| `reports/figures/shap_waterfall.png` | SHAP waterfall for sample fraud case |

---

## 🛠️ Technology Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| ML Framework | XGBoost, scikit-learn |
| Imbalanced Learning | imbalanced-learn (SMOTE, Tomek) |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| API Serving | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Testing | pytest |
| Containerization | Docker, Docker Compose |

---

## 🔮 Future Improvements

- [ ] Add anomaly detection models (Isolation Forest, Autoencoder) as ensemble members
- [ ] Implement real-time streaming with Apache Kafka
- [ ] Add model monitoring with Evidently AI for data/concept drift
- [ ] Deploy to AWS (ECR + ECS/Lambda) with CI/CD via GitHub Actions
- [ ] Add geospatial and temporal features for enhanced detection

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by ULB Machine Learning Group
- Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE Symposium Series on Computational Intelligence (2015)

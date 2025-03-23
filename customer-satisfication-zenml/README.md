# Customer Satisfaction Prediction with ZenML & MLflow

[![ZenML](https://img.shields.io/badge/Powered%20by-ZenML-blue)](https://zenml.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20&%20Models-orange)](https://mlflow.org)

End-to-end MLOps pipeline for predicting customer satisfaction using ZenML for orchestration and MLflow for experiment tracking.

## 🚀 Features
- Automated data preprocessing
- Model training (XGBoost/Random Forest)
- MLflow experiment tracking
- Model deployment-ready pipeline
- ZenML artifact/step caching

## ⚙️ Pipeline Steps
1. `load_data`: Fetch dataset from CSV/API
2. `preprocess`: Clean and feature-engineer data
3. `train`: Train model with hyperparameter tuning
4. `evaluate`: Track metrics with MLflow
5. `deploy`: Push best model to registry (optional)

## 🛠️ Getting Started

### Install
```bash
pip install zenml["server"] mlflow xgboost scikit-learn pandas
zenml integration install mlflow -y

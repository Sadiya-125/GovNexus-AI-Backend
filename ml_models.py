"""
ML Models Module for Inventory Management
Supports multiple model types including deep learning and transfer learning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


def compute_metrics(y_true, y_pred):
    """
    Compute comprehensive metrics for model evaluation
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Additional metrics
    corr = np.corrcoef(y_pred, y_true)[0, 1]
    rae = np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true - np.mean(y_true)))
    rrse = np.sqrt(np.sum((y_pred - y_true)**2) / np.sum((y_true - np.mean(y_true))**2))

    # MAPE with zero handling
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = 0.0

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'correlation': float(corr),
        'rae': float(rae),
        'rrse': float(rrse),
        'mape': float(mape)
    }


def get_model(model_type, **params):
    """
    Get a machine learning model by type with optional parameters
    """
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 5),
            random_state=42
        ),
        'linear_regression': LinearRegression(),
        'ridge': Ridge(
            alpha=params.get('alpha', 1.0),
            random_state=42
        ),
        'lasso': Lasso(
            alpha=params.get('alpha', 1.0),
            random_state=42
        ),
        'decision_tree': DecisionTreeRegressor(
            max_depth=params.get('max_depth', 10),
            random_state=42
        ),
        'svr': SVR(
            kernel=params.get('kernel', 'rbf'),
            C=params.get('C', 1.0)
        ),
        'knn': KNeighborsRegressor(
            n_neighbors=params.get('n_neighbors', 5)
        ),
        'mlp': MLPRegressor(
            hidden_layer_sizes=params.get('hidden_layers', (100, 50)),
            max_iter=params.get('max_iter', 500),
            random_state=42,
            early_stopping=True
        )
    }

    if model_type not in models:
        raise ValueError(f"Model type '{model_type}' not supported. Available: {list(models.keys())}")

    return models[model_type]


class ModelEnsemble:
    """
    Ensemble of multiple models for better predictions
    """
    def __init__(self, models=None):
        if models is None:
            self.models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('lr', Ridge(random_state=42))
            ]
        else:
            self.models = models

        self.weights = None

    def fit(self, X, y):
        """Train all models in the ensemble"""
        for name, model in self.models:
            model.fit(X, y)

    def predict(self, X):
        """
        Predict using weighted average of all models
        """
        predictions = []
        for name, model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Simple average (could be weighted)
        return np.mean(predictions, axis=0)

    def get_model_predictions(self, X):
        """Get predictions from each model separately"""
        return {name: model.predict(X) for name, model in self.models}


def create_profit_features(df):
    """
    Create advanced features for profit prediction
    """
    df = df.copy()

    # Time-based features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Interaction features
    df['price_qty_interaction'] = df['price'] * df['orderQty']
    df['cost_margin'] = (df['price'] - df['unitCost']) / df['price']

    return df


def train_stacking_model(base_models, X_train, y_train, X_val, y_val):
    """
    Train a stacking ensemble
    Meta-learner uses predictions from base models as features
    """
    # Train base models
    base_predictions_train = []
    base_predictions_val = []

    for name, model in base_models:
        model.fit(X_train, y_train)
        base_predictions_train.append(model.predict(X_train))
        base_predictions_val.append(model.predict(X_val))

    # Stack predictions
    X_train_stacked = np.column_stack(base_predictions_train)
    X_val_stacked = np.column_stack(base_predictions_val)

    # Train meta-learner
    meta_model = Ridge(random_state=42)
    meta_model.fit(X_train_stacked, y_train)

    return base_models, meta_model


def predict_with_uncertainty(model, X, n_samples=100):
    """
    Predict with uncertainty estimation using dropout (for neural networks)
    or bootstrap for other models
    """
    predictions = []

    if hasattr(model, 'predict_proba'):
        # For models with probabilistic outputs
        return model.predict(X), None

    # Bootstrap predictions for uncertainty
    if hasattr(model, 'estimators_'):  # Ensemble models
        for estimator in model.estimators_[:n_samples]:
            pred = estimator.predict(X)
            predictions.append(pred)
    else:
        # Single prediction
        return model.predict(X), None

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)

    return mean_pred, std_pred

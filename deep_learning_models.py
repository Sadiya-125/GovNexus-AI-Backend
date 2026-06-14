"""
Deep Learning Models for Inventory Management
Supports LSTM, GRU, CNN, and Transfer Learning
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, Dropout, LSTM, GRU, Conv1D, MaxPooling1D,
        Flatten, BatchNormalization, Input, Concatenate
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def create_sequences(data, n_steps):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape, units=[64, 32], dropout=0.2):
    """
    Build LSTM model for time series forecasting
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")

    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import MeanAbsoluteError

    model = Sequential([
        LSTM(units[0], return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        BatchNormalization(),
        LSTM(units[1], return_sequences=False),
        Dropout(dropout),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError(), MeanSquaredError()]
    )

    return model


def build_gru_model(input_shape, units=[64, 32], dropout=0.2):
    """
    Build GRU model for time series forecasting
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")

    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import MeanAbsoluteError

    model = Sequential([
        GRU(units[0], return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        BatchNormalization(),
        GRU(units[1], return_sequences=False),
        Dropout(dropout),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError(), MeanSquaredError()]
    )

    return model


def build_cnn_lstm_model(input_shape, conv_filters=[64, 32], lstm_units=50):
    """
    Build CNN-LSTM hybrid model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")

    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import MeanAbsoluteError

    model = Sequential([
        Conv1D(filters=conv_filters[0], kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=conv_filters[1], kernel_size=3, activation='relu'),
        BatchNormalization(),
        LSTM(lstm_units, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()]
    )

    return model


def build_mlp_model(input_shape, hidden_layers=[100, 50], dropout=0.2):
    """
    Build Multi-Layer Perceptron model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")

    model = Sequential()
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

    model.add(Dense(1))

    # Use MeanSquaredError() instead of 'mse' string to avoid deserialization issues
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import MeanAbsoluteError

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()]
    )

    return model


def build_attention_model(input_shape, lstm_units=64):
    """
    Build LSTM model with attention mechanism
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")

    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import MeanAbsoluteError

    inputs = Input(shape=input_shape)

    # LSTM layer
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    lstm_out = BatchNormalization()(lstm_out)

    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(lstm_out)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(lstm_units)(attention)
    attention = layers.Permute([2, 1])(attention)

    # Apply attention
    sent_representation = layers.multiply([lstm_out, attention])
    sent_representation = layers.Lambda(lambda xin: keras.backend.sum(xin, axis=1))(sent_representation)

    # Output layer
    output = Dense(32, activation='relu')(sent_representation)
    output = Dropout(0.2)(output)
    output = Dense(1)(output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()]
    )

    return model


def transfer_learning_model(base_model_path, input_shape, freeze_layers=2):
    """
    Create a transfer learning model from a pre-trained base model

    Args:
        base_model_path: Path to the base model weights
        input_shape: Input shape for new data
        freeze_layers: Number of layers to freeze from base model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")

    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import MeanAbsoluteError

    # Build new model architecture
    model = Sequential()

    # Add input layer with adaptation
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))

    # Try to load base model
    try:
        # Load model with custom objects to handle metric deserialization
        custom_objects = {
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
            'MeanSquaredError': MeanSquaredError,
            'MeanAbsoluteError': MeanAbsoluteError
        }
        base_model = keras.models.load_model(base_model_path, custom_objects=custom_objects, compile=False)

        # Freeze some layers
        for i, layer in enumerate(base_model.layers):
            if i < freeze_layers:
                layer.trainable = False
            model.add(layer)

    except Exception as e:
        print(f"Could not load base model: {e}")
        # Add default layers if base model fails
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for transfer learning
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()]
    )

    return model


def get_callbacks(checkpoint_path='models/dl_checkpoint.h5'):
    """
    Get training callbacks for deep learning models
    """
    if not TF_AVAILABLE:
        return []

    return [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
    ]


def prepare_dl_data(df, target_col='profit', n_steps=10):
    """
    Prepare data for deep learning models

    Returns sequences for time series models
    """
    # Sort by date
    df = df.sort_values('orderDate').copy()

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[target_col]])

    # Create sequences
    X, y = create_sequences(scaled_data, n_steps)

    # Reshape for LSTM/GRU (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler


def train_deep_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train deep learning model with callbacks
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(),
        verbose=0
    )

    return model, history


def predict_future(model, last_sequence, scaler, n_future=30):
    """
    Predict future values using trained model

    Args:
        model: Trained model
        last_sequence: Last known sequence
        scaler: Scaler used for data
        n_future: Number of future steps to predict
    """
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(n_future):
        # Predict next value
        pred = model.predict(current_seq.reshape(1, -1, 1), verbose=0)[0][0]
        predictions.append(pred)

        # Update sequence
        current_seq = np.append(current_seq[1:], pred)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predictions.flatten()


class EnsembleDeepLearning:
    """
    Ensemble of deep learning models
    """
    def __init__(self, models=None):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        self.models = models or []
        self.weights = None

    def fit(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train all models"""
        for model in self.models:
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                verbose=0
            )

    def predict(self, X):
        """Predict using weighted average"""
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)

        return np.mean(predictions, axis=0)

# Copyright 2026 Sina J
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model building, training utilities, and data splitting functions.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from src.config import RANDOM_STATE, TEST_SIZE, VAL_SPLIT_FROM_TEMP, EPOCHS, BATCH_SIZE, PATIENCE


def build_model(input_dim: int) -> Model:
    """
    Build a binary classification neural network.
    
    Args:
        input_dim: Dimension of input features
    
    Returns:
        Compiled Keras Model
    """
    inputs = Input(shape=(input_dim,), name="Input_Layer")
    x = Dense(64, activation="relu", name="Dense_1")(inputs)
    x = Dense(32, activation="relu", name="Dense_2")(x)
    outputs = Dense(1, activation="sigmoid", name="Output_Layer")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def split_data(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE):
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature matrix
        y: Labels
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=VAL_SPLIT_FROM_TEMP,
        stratify=y_temp,
        random_state=random_state,
    )
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data_with_indices(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE):
    """
    Split data into train, validation, and test sets preserving original indices.
    
    Args:
        X: Feature matrix
        y: Labels
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, 
                  idx_train, idx_val, idx_test)
    """
    indices = np.arange(len(y))
    
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, indices, test_size=TEST_SIZE, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp,
        y_temp,
        idx_temp,
        test_size=VAL_SPLIT_FROM_TEMP,
        stratify=y_temp,
        random_state=random_state,
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test


def oversample_1to1(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE):
    """
    Oversample minority class to match majority class count (1:1 ratio).
    
    Args:
        X: Feature matrix
        y: Labels
        random_state: Random seed
    
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_res, y_res = [], []
    for c in classes:
        X_c = X[y == c]
        y_c = y[y == c]
        X_up, y_up = resample(
            X_c, y_c, replace=True, n_samples=max_count, random_state=random_state
        )
        X_res.append(X_up)
        y_res.append(y_up)

    return np.vstack(X_res), np.hstack(y_res)


def scale_features(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
):
    """
    Standardize features using StandardScaler.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
    
    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    exp_name: str = "model",
    class_weight: dict = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = PATIENCE,
    verbose: int = 0,
):
    """
    Build and train a neural network model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        exp_name: Experiment name for the model
        class_weight: Class weights dictionary
        epochs: Maximum number of epochs
        batch_size: Batch size
        patience: Early stopping patience
        verbose: Verbosity level
    
    Returns:
        Trained model
    """
    model = build_model(X_train.shape[1])
    model.name = exp_name

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=verbose,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[es],
        verbose=verbose,
    )

    return model
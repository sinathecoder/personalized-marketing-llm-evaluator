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
Experiment runner for comparing embedding models and training strategies.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, precision_recall_curve,
    average_precision_score, roc_curve, auc
)
from tensorflow.keras.callbacks import EarlyStopping

from src.config import (
    EMBEDDING_COLS, STRATEGIES, WEIGHTED_CLASS_WEIGHT,
    EPOCHS, BATCH_SIZE, PATIENCE, RANDOM_STATE
)
from src.utils.embedding_utils import build_X
from src.models.model_utils import (
    split_data, scale_features, oversample_1to1, train_model
)


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """
    Compute classification metrics at a given threshold.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        threshold: Decision threshold
    
    Returns:
        Dictionary with Accuracy, Precision, Recall, F1
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_experiment(X: np.ndarray, y: np.ndarray, strategy: str, exp_name: str) -> dict:
    """
    Run a single experiment: train model and compute metrics.
    
    Args:
        X: Feature matrix
        y: Labels
        strategy: 'oversample_1to1' or 'weighted_99to1'
        exp_name: Display name for the experiment
    
    Returns:
        Dictionary with all metrics, curves, and experiment info
    """
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Scale
    X_train, X_val, X_test, _ = scale_features(X_train, X_val, X_test)

    print("shape:", X_train.shape)

    class_weight = None
    if strategy == "oversample_1to1":
        X_train, y_train = oversample_1to1(X_train, y_train)
    elif strategy == "weighted_99to1":
        class_weight = WEIGHTED_CLASS_WEIGHT

    model = train_model(
        X_train, y_train, X_val, y_val,
        exp_name=exp_name,
        class_weight=class_weight,
    )

    # Probabilities
    y_prob = model.predict(X_test).ravel()

    # Metrics at different thresholds
    m_05 = metrics_at_threshold(y_test, y_prob, threshold=0.5)
    m_09 = metrics_at_threshold(y_test, y_prob, threshold=0.9)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # PR
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    return {
        "Experiment": exp_name,
        "Strategy": strategy,
        "Acc@0.5": m_05["Accuracy"],
        "Prec@0.5": m_05["Precision"],
        "Rec@0.5": m_05["Recall"],
        "F1@0.5": m_05["F1"],
        "Acc@0.9": m_09["Accuracy"],
        "Prec@0.9": m_09["Precision"],
        "Rec@0.9": m_09["Recall"],
        "F1@0.9": m_09["F1"],
        "ROC_AUC": roc_auc,
        "AP": ap,
        "fpr": fpr,
        "tpr": tpr,
        "prec_curve": prec_curve,
        "rec_curve": rec_curve,
    }


def run_all_experiments(labled_data: pd.DataFrame) -> pd.DataFrame:
    """
    Run all experiments (4 embeddings x 2 strategies) and return results table.
    
    Args:
        labled_data: DataFrame with embedding columns and 'R' label
    
    Returns:
        DataFrame with experiment results (excluding curve data)
    """
    results = []
    y = labled_data["R"].astype(int).values

    for m_col, eid in EMBEDDING_COLS:
        X = build_X(labled_data, m_col)
        for s in STRATEGIES:
            results.append(run_experiment(X, y, s, eid))

    results_df = pd.DataFrame(results).drop(
        columns=["fpr", "tpr", "prec_curve", "rec_curve"], errors="ignore"
    )
    return results_df


def plot_roc_curves(results: list, save_prefix: str = "roc_curve"):
    """
    Plot ROC curves for all experiments grouped by embedding.
    
    Args:
        results: List of experiment result dictionaries
        save_prefix: Prefix for saved figure filenames
    """
    import matplotlib.pyplot as plt
    
    results_df = pd.DataFrame(results)
    for eid in results_df["Experiment"].unique():
        plt.figure(figsize=(6, 5))
        for r in results:
            if r["Experiment"] == eid:
                plt.plot(
                    r["fpr"], r["tpr"],
                    label=f'{r["Strategy"]} (AUC={r["ROC_AUC"]:.3f})'
                )
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_prefix}_{eid}.png", dpi=300)
        plt.show()


def plot_pr_curves(results: list, save_prefix: str = "precision_recall"):
    """
    Plot Precision-Recall curves for all experiments grouped by embedding.
    
    Args:
        results: List of experiment result dictionaries
        save_prefix: Prefix for saved figure filenames
    """
    import matplotlib.pyplot as plt
    
    results_df = pd.DataFrame(results)
    for eid in results_df["Experiment"].unique():
        plt.figure(figsize=(6, 5))
        for r in results:
            if r["Experiment"] == eid:
                plt.plot(
                    r["rec_curve"], r["prec_curve"],
                    label=f'{r["Strategy"]} (AP={r["AP"]:.3f})'
                )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_prefix}_{eid}.png", dpi=300)
        plt.show()
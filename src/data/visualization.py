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
Data visualization utilities for EDA (Exploratory Data Analysis).
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_numeric_distributions(
    customers: pd.DataFrame,
    exclude_cols: list = None,
    n_cols: int = 4,
    save_path: str = "cust_numeric_dist.png",
):
    """
    Plot histograms of numeric columns in the customers DataFrame.
    
    Args:
        customers: DataFrame with customer data
        exclude_cols: Columns to exclude (e.g. latitude, longitude)
        n_cols: Number of plots per row
        save_path: Path to save the figure
    """
    if exclude_cols is None:
        exclude_cols = ["latitude", "longitude"]

    numeric_cols = [
        col
        for col in customers.select_dtypes(include="number").columns
        if col not in exclude_cols
    ]

    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(
            customers[col].dropna(), bins=30, color="skyblue", edgecolor="black"
        )
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.show()
    print(f"✅ Saved numeric distributions to {save_path}")


def plot_categorical_distributions(
    customers: pd.DataFrame,
    exclude_cols: list = None,
    n_cols: int = 4,
    save_path: str = "cust_cat_dist.png",
):
    """
    Plot bar charts of categorical columns in the customers DataFrame.
    
    Args:
        customers: DataFrame with customer data
        exclude_cols: Columns to exclude
        n_cols: Number of plots per row
        save_path: Path to save the figure
    """
    if exclude_cols is None:
        exclude_cols = [
            "customer_id",
            "segment",
            "signup_date",
            "last_login_date",
        ]

    cat_cols = customers.select_dtypes(include="object").columns
    cat_cols = [col for col in cat_cols if col not in exclude_cols]

    n_rows = (len(cat_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        customers[col].value_counts(normalize=True).plot(
            kind="bar", ax=axes[i], color="skyblue", edgecolor="black"
        )
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_ylabel("Proportion")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(
        left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.9, wspace=0.4
    )
    plt.savefig(save_path, dpi=500, bbox_inches="tight", pad_inches=0.002)
    plt.show()
    print(f"✅ Saved categorical distributions to {save_path}")
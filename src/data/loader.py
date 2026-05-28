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
Data loading, preprocessing, and label generation utilities.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple

from src.config import (
    CUSTOMERS_FILE,
    PRODUCTS_FILE,
    GENERATED_COPY_CSV,
    GENERATED_COPY_EXCEL,
    LABELED_DATA_PARQUET,
    QUALITY_DIMS,
    HALF_LABELS_ZERO,
)


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean customers and products datasets."""
    customers = pd.read_csv(CUSTOMERS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)
    # Normalize product column names
    products.columns = products.columns.str.strip().str.lower().str.replace(" ", "_")
    return customers, products


def load_generated_copy() -> pd.DataFrame:
    """Load generated marketing copy from CSV."""
    return pd.read_csv(GENERATED_COPY_CSV)


def load_labeled_data(source: Optional[str] = None) -> pd.DataFrame:
    """
    Load labeled data from parquet or excel file.
    
    Args:
        source: 'parquet' or 'excel' or None (tries parquet first, then excel)
    
    Returns:
        DataFrame with labeled data
    """
    if source == "parquet" or source is None:
        try:
            return pd.read_parquet(LABELED_DATA_PARQUET)
        except (FileNotFoundError, ImportError):
            if source == "parquet":
                raise
    
    if source == "excel" or source is None:
        return pd.read_excel(GENERATED_COPY_EXCEL)


def generate_random_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate random quality labels (P, H, C, G) and compute R (Reject).
    
    Creates synthetic labels where the first half of the data gets zeros
    and the rest get random binary labels, then shuffles.
    """
    cols = ["P", "H", "C", "G"]
    n = len(df)
    half = n // 2

    zeros = np.zeros((half, len(cols)), dtype=int)
    randoms = np.random.randint(0, 2, size=(n - half, len(cols)))

    combined = np.vstack([zeros, randoms])
    np.random.shuffle(combined)

    df[cols] = combined
    df["R"] = (df[cols] == 1).any(axis=1).astype(int)
    return df


def break_text_by_words(text: str, words_per_line: int = 10) -> str:
    """
    Break text into lines with a specified number of words per line.
    
    Args:
        text: Input text to break
        words_per_line: Number of words per line
    
    Returns:
        Text with line breaks inserted
    """
    if not isinstance(text, str):
        return ""
    words = text.split()
    lines = [
        " ".join(words[i : i + words_per_line])
        for i in range(0, len(words), words_per_line)
    ]
    return "\n".join(lines)
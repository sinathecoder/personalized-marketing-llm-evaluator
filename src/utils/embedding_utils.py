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
Embedding utility functions for creating and processing embeddings.
"""

import time
import numpy as np
import pandas as pd
from typing import Optional, List


class EmbeddingGenerator:
    """Generate embeddings for text using OpenAI's embedding API."""
    
    def __init__(self, client, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model
        self._counter = 0
    
    def get_embedding(self, text: Optional[str], retries: int = 3) -> List[float]:
        """
        Get embedding vector for text.
        
        Args:
            text: Input text to embed
            retries: Number of retries on failure
        
        Returns:
            Embedding vector as list of floats
        """
        if text is None:
            text = ""
        
        for attempt in range(retries):
            try:
                self._counter += 1
                response = self.client.embeddings.create(
                    model=self.model, input=text
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(20)
                else:
                    print(f"Failed after {retries} attempts")
                    return [0.0] * 1536  # Return zero vector as fallback
    
    def get_counter(self) -> int:
        """Get the number of API calls made."""
        return self._counter


def build_X(df: pd.DataFrame, col_marketing: str) -> np.ndarray:
    """
    Build feature matrix from embedding column.
    
    Args:
        df: DataFrame containing embeddings
        col_marketing: Column name with embedding vectors
    
    Returns:
        NumPy array of shape (n_samples, embedding_dim)
    """
    return np.array([np.array(m) for m in df[col_marketing]])


def apply_embeddings(
    df: pd.DataFrame, 
    columns: List[str], 
    embedding_generator: EmbeddingGenerator
) -> pd.DataFrame:
    """
    Apply embeddings to specified columns in DataFrame.
    
    Args:
        df: DataFrame with text columns
        columns: List of column names to embed
        embedding_generator: EmbeddingGenerator instance
    
    Returns:
        DataFrame with new embedding columns added
    """
    for col in columns:
        embed_col_name = f"{col}_text_embedding_3_small"
        df[embed_col_name] = df[col].apply(embedding_generator.get_embedding)
    return df
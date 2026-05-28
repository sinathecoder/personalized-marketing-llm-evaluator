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
Configuration constants for the Personalized Marketing LLM Quality Control system.
"""

import openai

# === API Configuration ===
API_KEY = ""  # Set your OpenAI API key here
CLIENT = openai.OpenAI(api_key=API_KEY)

# === File Paths ===
RAW_DIR = "raw"
CUSTOMERS_FILE = f"{RAW_DIR}/customers_200_segments.csv"
PRODUCTS_FILE = f"{RAW_DIR}/products_1_segment.csv"
GENERATED_COPY_CSV = f"{RAW_DIR}/generated_marketing_copies.csv"
GENERATED_COPY_EXCEL = f"{RAW_DIR}/generated_marketing_labeled.xlsx"
LABELED_DATA_PARQUET = f"{RAW_DIR}/labeled_data_with_embeddings.parquet"

# === Quality Dimensions (0-10 scale) ===
QUALITY_DIMS = ["Relevance", "Creativity", "Personalization", "Clarity", "Completeness"]

# === Label Generation ===
HALF_LABELS_ZERO = False

# === LLM Configuration ===
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1000

# === Embedding Configuration ===
EMBEDDING_MODEL = "text-embedding-3-small"

# === EDA Column Mappings ===
NUMERIC_COLUMNS_MAPPING = {
    "total_spent": None,
    "age": None,
    "income": None,
    "purchase_frequency": None,
    "days_since_last_purchase": None,
    "total_purchases": None,
    "avg_order_value": None,
}
CATEGORICAL_COLUMNS_MAPPING = {
    "gender": None,
    "product_category_preference": None,
    "subscription_status": None,
    "communication_preference": None,
    "preferred_language": None,
    "location": None,
    "loyalty_tier": None,
    "device_preference": None,
    "promotion_response": None,
}

# === Experiment Groups ===
EXPERIMENTS = {
    "PromptOnly": {
        "only_prompt": True,
        "model": "svc",
        "group": "single_embedding",
    },
    "MarketingCopyOnly": {
        "only_prompt": False,
        "model": "svc",
        "group": "single_embedding",
    },
    "SVC_Averaged": {
        "only_prompt": None,
        "model": "svc",
        "group": "averaged",
    },
    "SVC_Combined": {
        "only_prompt": None,
        "model": "svc",
        "group": "combined",
    },
    "XGB_Averaged": {
        "only_prompt": None,
        "model": "xgb",
        "group": "averaged",
    },
    "XGB_Combined": {
        "only_prompt": None,
        "model": "xgb",
        "group": "combined",
    },
    "RF_Averaged": {
        "only_prompt": None,
        "model": "rf",
        "group": "averaged",
    },
    "RF_Combined": {
        "only_prompt": None,
        "model": "rf",
        "group": "combined",
    },
}
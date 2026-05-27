"""
Configuration constants for the Personalized Marketing LLM Quality Control system.
"""

import openai

# === API Configuration ===
API_KEY = ""  # Set your OpenAI API key here
CLIENT = openai.OpenAI(api_key=API_KEY)

# === Data Files ===
CUSTOMERS_FILE = "customers_200_segments.csv"
PRODUCTS_FILE = "ten_products_with_attributes.csv"
GENERATED_COPY_CSV = "llm_generated_marketing_copy.csv"
GENERATED_COPY_EXCEL = "llm_generated_marketing_copy.xlsx"
LABELED_DATA_PARQUET = "labled_data_2.parquet"

# === LLM Configuration ===
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.8
LLM_MAX_TOKENS = 120
EMBEDDING_MODEL = "text-embedding-3-small"

# === Embedding Columns Configuration ===
EMBEDDING_COLS = [
    ("marketing_copy_embedding", "text-embedding-3-large"),
    ("marketing_copy_embedding_256", "text-embedding-3-large_reduced_256"),
    ("marketing_text_embedding_ada", "text_embedding_ada"),
    ("marketing_text_embedding_3_small", "text_embedding_3_small"),
]

# === Strategies ===
STRATEGIES = ["oversample_1to1", "weighted_99to1"]

# === Experiments Configuration ===
EXPERIMENTS_CONFIG = [
    ("marketing_copy_embedding", "oversample_1to1", "3-large (OS)"),
    ("marketing_copy_embedding", "weighted_99to1", "3-large (W99)"),
    ("marketing_copy_embedding_256", "oversample_1to1", "3-large-256 (OS)"),
    ("marketing_copy_embedding_256", "weighted_99to1", "3-large-256 (W99)"),
    ("marketing_text_embedding_ada", "oversample_1to1", "Ada (OS)"),
    ("marketing_text_embedding_ada", "weighted_99to1", "Ada (W99)"),
    ("marketing_text_embedding_3_small", "oversample_1to1", "3-small (OS)"),
    ("marketing_text_embedding_3_small", "weighted_99to1", "3-small (W99)"),
]

# === Training Configuration ===
RANDOM_STATE = 42
TEST_SIZE = 0.3
VAL_SPLIT_FROM_TEMP = 0.80
EPOCHS = 200
BATCH_SIZE = 256
PATIENCE = 10
WEIGHTED_CLASS_WEIGHT = {0: 1, 1: 99}

# === Quality Dimensions ===
QUALITY_DIMS = ["P", "H", "C", "G", "L", "R"]

# === Quality Dimension Names ===
QUALITY_DIM_NAMES = {
    "P": "Privacy",
    "H": "Hallucination",
    "C": "Coherence",
    "G": "Grammar",
    "L": "Linguistic",
    "R": "Reject (any violation)",
}

# === Random Label Generation Config ===
HALF_LABELS_ZERO = True  # First half of labels are zeros, rest are random
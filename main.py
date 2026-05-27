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
Personalized Marketing LLM Quality Control
===========================================
Refactored from the original monolithic Jupyter notebook into modular files.

Usage:
    # Step 1: Data Exploration
    python main.py --eda

    # Step 2: Generate marketing copies (requires API key)
    python main.py --generate

    # Step 3: Generate labels
    python main.py --label

    # Step 4: Generate embeddings
    python main.py --embed

    # Step 5: Run model experiments
    python main.py --experiments

    # Step 6: Misclassification analysis
    python main.py --misclassification

    # Step 7: Embedding interpretability
    python main.py --interpretability

    # Step 8: FP/FN analysis
    python main.py --fpfn

    # Run all steps
    python main.py --all
"""

import argparse
import sys

from src.config import CLIENT, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, EMBEDDING_MODEL
from src.data.loader import (
    load_datasets,
    load_generated_copy,
    load_labeled_data,
    generate_random_labels,
    break_text_by_words,
)
from src.data.visualization import plot_numeric_distributions, plot_categorical_distributions
from src.data.prompt_builder import generate_marketing_copies
from src.utils.embedding_utils import EmbeddingGenerator, apply_embeddings
from src.models.experiment_runner import run_all_experiments, plot_roc_curves, plot_pr_curves
from src.analysis.misclassification_analysis import run_full_misclassification_analysis
from src.analysis.embedding_interpretability import (
    run_interpretability_for_selected_experiments,
    compare_embeddings_similarity,
)
from src.analysis.fp_fn_analysis import run_fp_fn_analysis


def eda_step():
    """Step 1: Exploratory Data Analysis - load and visualize data."""
    print("=" * 80)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    customers, products = load_datasets()
    print(f"\nCustomers shape: {customers.shape}")
    print(f"Products shape: {products.shape}")
    print(f"\nProducts columns: {list(products.columns)}")

    plot_numeric_distributions(customers)
    plot_categorical_distributions(customers)
    print("\n✅ EDA complete!")


def generate_step():
    """Step 2: Generate marketing copies using LLM."""
    print("=" * 80)
    print("STEP 2: GENERATE MARKETING COPIES")
    print("=" * 80)

    if not CLIENT.api_key:
        print("❌ OpenAI API key not set. Please set API_KEY in src/config.py")
        return

    customers, products = load_datasets()
    df_copy = generate_marketing_copies(
        customers, products, CLIENT, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
    )

    from src.config import GENERATED_COPY_CSV
    df_copy.to_csv(GENERATED_COPY_CSV, index=False)
    print(f"\n✅ Marketing copies saved to {GENERATED_COPY_CSV}")


def label_step():
    """Step 3: Generate random labels and save to Excel."""
    print("=" * 80)
    print("STEP 3: GENERATE LABELS")
    print("=" * 80)

    df = load_generated_copy()
    # Apply break_text_by_words to marketing_copy
    df["marketing_copy"] = df["marketing_copy"].apply(lambda x: break_text_by_words(x))
    df = generate_random_labels(df)

    from src.config import GENERATED_COPY_EXCEL
    df.to_excel(GENERATED_COPY_EXCEL, index=False)
    print(f"\n✅ Labeled data saved to {GENERATED_COPY_EXCEL}")
    print(f"   Label distribution: R=1 count = {df['R'].sum()}/{len(df)}")


def embed_step():
    """Step 4: Generate embeddings for prompt and marketing copy columns."""
    print("=" * 80)
    print("STEP 4: GENERATE EMBEDDINGS")
    print("=" * 80)

    if not CLIENT.api_key:
        print("❌ OpenAI API key not set. Please set API_KEY in src/config.py")
        return

    labled_data = load_labeled_data(source="excel")
    print(f"\nLoaded data shape: {labled_data.shape}")

    emb_gen = EmbeddingGenerator(CLIENT, model=EMBEDDING_MODEL)
    labled_data = apply_embeddings(labled_data, ["prompt", "marketing_copy"], emb_gen)
    print(f"Total API calls: {emb_gen.get_counter()}")

    from src.config import LABELED_DATA_PARQUET
    labled_data.to_parquet(LABELED_DATA_PARQUET)
    print(f"\n✅ Embeddings saved to {LABELED_DATA_PARQUET}")


def experiments_step():
    """Step 5: Run all model experiments and compare results."""
    print("=" * 80)
    print("STEP 5: RUN MODEL EXPERIMENTS")
    print("=" * 80)

    labled_data = load_labeled_data(source="parquet")
    print(f"\nLoaded data shape: {labled_data.shape}")

    results_df = run_all_experiments(labled_data)
    print("\n📊 RESULTS TABLE:")
    print(results_df.to_string(index=False))
    results_df.to_csv("experiment_results.csv", index=False)
    print("\n✅ Results saved to experiment_results.csv")


def misclassification_step():
    """Step 6: Run misclassification overlap analysis."""
    print("=" * 80)
    print("STEP 6: MISCLASSIFICATION OVERLAP ANALYSIS")
    print("=" * 80)

    labled_data = load_labeled_data(source="parquet")
    run_full_misclassification_analysis(labled_data)
    print("\n✅ Misclassification analysis complete!")


def interpretability_step():
    """Step 7: Run embedding interpretability analysis."""
    print("=" * 80)
    print("STEP 7: EMBEDDING INTERPRETABILITY")
    print("=" * 80)

    labled_data = load_labeled_data(source="parquet")
    interpretability_results = run_interpretability_for_selected_experiments(labled_data)
    comparison_df = compare_embeddings_similarity(labled_data)
    print("\n✅ Embedding interpretability complete!")


def fpfn_step():
    """Step 8: Run FP/FN analysis."""
    print("=" * 80)
    print("STEP 8: FALSE POSITIVE / FALSE NEGATIVE ANALYSIS")
    print("=" * 80)

    labled_data = load_labeled_data(source="parquet")
    run_fp_fn_analysis(labled_data)
    print("\n✅ FP/FN analysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Personalized Marketing LLM Quality Control"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all steps (1-8)"
    )
    parser.add_argument("--eda", action="store_true", help="Step 1: EDA")
    parser.add_argument("--generate", action="store_true", help="Step 2: Generate marketing copies")
    parser.add_argument("--label", action="store_true", help="Step 3: Generate labels")
    parser.add_argument("--embed", action="store_true", help="Step 4: Generate embeddings")
    parser.add_argument("--experiments", action="store_true", help="Step 5: Run experiments")
    parser.add_argument("--misclassification", action="store_true", help="Step 6: Misclassification analysis")
    parser.add_argument("--interpretability", action="store_true", help="Step 7: Embedding interpretability")
    parser.add_argument("--fpfn", action="store_true", help="Step 8: FP/FN analysis")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.all:
        eda_step()
        generate_step()
        label_step()
        embed_step()
        experiments_step()
        misclassification_step()
        interpretability_step()
        fpfn_step()
        return

    if args.eda:
        eda_step()
    if args.generate:
        generate_step()
    if args.label:
        label_step()
    if args.embed:
        embed_step()
    if args.experiments:
        experiments_step()
    if args.misclassification:
        misclassification_step()
    if args.interpretability:
        interpretability_step()
    if args.fpfn:
        fpfn_step()


if __name__ == "__main__":
    main()
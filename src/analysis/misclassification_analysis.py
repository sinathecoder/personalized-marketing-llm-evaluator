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
Misclassification overlap analysis across multiple experiments.
Analyzes which samples are misclassified by which models/strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping

from src.config import (
    EMBEDDING_COLS, STRATEGIES, WEIGHTED_CLASS_WEIGHT,
    RANDOM_STATE
)
from src.utils.embedding_utils import build_X
from src.models.model_utils import (
    build_model, split_data_with_indices, scale_features,
    oversample_1to1, train_model
)


def run_single_experiment_collect_errors(
    labled_data: pd.DataFrame, embedding_col: str, strategy: str,
    exp_name: str, random_state: int = RANDOM_STATE
) -> dict:
    """
    Run one experiment and return misclassification indices.
    
    Args:
        labled_data: DataFrame with embedding columns and 'R' label
        embedding_col: Column name for embeddings
        strategy: 'oversample_1to1' or 'weighted_99to1'
        exp_name: Display name for experiment
        random_state: Random seed
    
    Returns:
        Dictionary with experiment results including FP/FN indices
    """
    X = build_X(labled_data, embedding_col)
    y = labled_data["R"].astype(int).values
    indices = np.arange(len(y))

    # Split with indices
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, indices, test_size=0.3, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.80, stratify=y_temp, random_state=random_state
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Apply strategy
    if strategy == "oversample_1to1":
        classes, counts = np.unique(y_train, return_counts=True)
        max_count = counts.max()
        X_res_list, y_res_list = [], []
        for c in classes:
            X_c = X_train_scaled[y_train == c]
            y_c = y_train[y_train == c]
            X_up, y_up = resample(X_c, y_c, replace=True, n_samples=max_count, random_state=random_state)
            X_res_list.append(X_up)
            y_res_list.append(y_up)
        X_train_balanced = np.vstack(X_res_list)
        y_train_balanced = np.hstack(y_res_list)
        class_weight = None
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
        class_weight = WEIGHTED_CLASS_WEIGHT

    # Train model
    model = build_model(X_train_balanced.shape[1])
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
    model.fit(
        X_train_balanced, y_train_balanced,
        validation_data=(X_val_scaled, y_val),
        epochs=200, batch_size=256,
        class_weight=class_weight, callbacks=[es], verbose=0
    )

    # Predict
    y_prob = model.predict(X_test_scaled, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # Identify errors
    fp_mask = (y_test == 0) & (y_pred == 1)
    fn_mask = (y_test == 1) & (y_pred == 0)

    fp_indices = set(idx_test[fp_mask])
    fn_indices = set(idx_test[fn_mask])

    return {
        "experiment": exp_name,
        "embedding": embedding_col,
        "strategy": strategy,
        "fp_indices": fp_indices,
        "fn_indices": fn_indices,
        "n_fp": len(fp_indices),
        "n_fn": len(fn_indices),
        "test_size": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def run_all_experiments_and_collect_errors(labled_data: pd.DataFrame) -> list:
    """
    Run all 8 experiments (4 embeddings x 2 strategies) and collect errors.
    
    Args:
        labled_data: DataFrame with data
    
    Returns:
        List of result dictionaries for all experiments
    """
    all_results = []
    print("=" * 80)
    print("RUNNING 8 EXPERIMENTS (4 embeddings x 2 strategies)")
    print("=" * 80)

    for emb_col, exp_name in EMBEDDING_COLS:
        print(f"\n📌 Embedding: {exp_name} (column: {emb_col})")
        for strategy in STRATEGIES:
            full_name = f"{exp_name}_{strategy}"
            print(f"   Running: {full_name}...")
            result = run_single_experiment_collect_errors(
                labled_data, emb_col, strategy, full_name
            )
            all_results.append(result)
            print(f"   FP: {result['n_fp']}, FN: {result['n_fn']}, Acc: {result['accuracy']:.3f}, F1: {result['f1']:.3f}")

    print("\n✅ All experiments completed!")
    return all_results


def print_experiment_summary(all_results: list):
    """Print summary table of all experiments."""
    summary_data = []
    for r in all_results:
        summary_data.append({
            "Experiment": r["experiment"],
            "FP": r["n_fp"],
            "FN": r["n_fn"],
            "Total Errors": r["n_fp"] + r["n_fn"],
            "Accuracy": f"{r['accuracy']:.3f}",
            "Precision": f"{r['precision']:.3f}",
            "Recall": f"{r['recall']:.3f}",
            "F1": f"{r['f1']:.3f}",
        })
    summary_df = pd.DataFrame(summary_data)
    print("\n📊 EXPERIMENT SUMMARY:")
    print(summary_df.to_string(index=False))
    return summary_df


def analyze_misclassification_overlap(all_results: list, labled_data: pd.DataFrame):
    """
    Analyze overlap of misclassifications across all experiments.
    
    Args:
        all_results: List of experiment result dictionaries
        labled_data: Original DataFrame with marketing_copy and labels
    
    Returns:
        Tuple of (misclassified_sets, common_all)
    """
    experiment_names = [r["experiment"] for r in all_results]

    # Create sets of misclassified indices
    misclassified_sets = {}
    for r in all_results:
        misclassified_sets[r["experiment"]] = r["fp_indices"] | r["fn_indices"]

    # Print summary
    print("\n📊 EXPERIMENT ERROR SUMMARY:")
    print("-" * 60)
    print(f"{'Experiment':<40} {'Errors':<10} {'Error Rate':<10}")
    print("-" * 60)
    for name in experiment_names:
        errors = len(misclassified_sets[name])
        rate = errors / r["test_size"] * 100 if r["test_size"] > 0 else 0
        print(f"{name:<40} {errors:<10} {rate:<10.1f}%")

    # Consistently misclassified by ALL
    common_all = misclassified_sets[experiment_names[0]]
    for name in experiment_names[1:]:
        common_all = common_all & misclassified_sets[name]

    print(f"\n🔍 Consistently misclassified by ALL experiments: {len(common_all)} samples")
    if len(common_all) > 0:
        for i, idx in enumerate(list(common_all)[:5]):
            copy_text = labled_data.iloc[idx]["marketing_copy"][:200]
            violations = [d for d in ["P", "H", "C", "L"] if d in labled_data.columns and labled_data.iloc[idx][d] == 1]
            print(f"\n  Example {i+1} (Index: {idx}):")
            print(f"    Copy: {copy_text}...")
            if violations:
                print(f"    Violations: {', '.join(violations)}")

    return misclassified_sets, common_all


def plot_overlap_heatmap(misclassified_sets: dict, save_path: str = "overlap_heatmap.png"):
    """
    Plot heatmap of Jaccard similarity between experiment misclassifications.
    
    Args:
        misclassified_sets: Dictionary mapping experiment name to set of misclassified indices
        save_path: Path to save figure
    """
    experiment_names = list(misclassified_sets.keys())
    n_exp = len(experiment_names)
    jaccard_matrix = np.zeros((n_exp, n_exp))

    for i, name_i in enumerate(experiment_names):
        for j, name_j in enumerate(experiment_names):
            set_i = misclassified_sets[name_i]
            set_j = misclassified_sets[name_j]
            union = len(set_i | set_j)
            jaccard_matrix[i, j] = len(set_i & set_j) / union if union > 0 else 0

    # Short names
    short_names = []
    for name in experiment_names:
        strat = "OS" if "oversample" in name else "W99"
        if "text-embedding-3-large" in name and "reduced" not in name:
            emb = "Lg"
        elif "reduced" in name:
            emb = "Lg_256"
        elif "ada" in name:
            emb = "Ada"
        elif "small" in name:
            emb = "Sm"
        else:
            emb = name[:15]
        short_names.append(f"{emb}_{strat}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        jaccard_matrix, annot=True, fmt=".3f",
        xticklabels=short_names, yticklabels=short_names,
        cmap="YlOrRd", vmin=0, vmax=1,
    )
    plt.title("Jaccard Similarity of Misclassifications Across Experiments\n(OS=Oversample, W99=Weighted 99:1)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"\n✅ Heatmap saved to {save_path}")
    return jaccard_matrix


def plot_error_rates_comparison(all_results: list, save_path: str = "error_rates_comparison.png"):
    """
    Plot bar chart comparing error rates across experiments.
    
    Args:
        all_results: List of experiment result dictionaries
        save_path: Path to save figure
    """
    experiment_names = [r["experiment"] for r in all_results]
    n_test = all_results[0]["test_size"]
    error_rates = [
        (r["n_fp"] + r["n_fn"]) / n_test * 100 if n_test > 0 else 0
        for r in all_results
    ]

    # Short names
    short_names = []
    for name in experiment_names:
        strat = "OS" if "oversample" in name else "W99"
        if "text-embedding-3-large" in name and "reduced" not in name:
            emb = "Lg"
        elif "reduced" in name:
            emb = "Lg_256"
        elif "ada" in name:
            emb = "Ada"
        elif "small" in name:
            emb = "Sm"
        else:
            emb = name[:10]
        short_names.append(f"{emb}_{strat}")

    colors = ["steelblue" if "oversample" in n else "coral" for n in experiment_names]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(short_names, error_rates, color=colors, edgecolor="black")
    plt.xlabel("Experiment", fontsize=12)
    plt.ylabel("Error Rate (%)", fontsize=12)
    plt.title("Classification Error Rates Across All Experiments\n(Blue=Oversample, Orange=Weighted 99:1)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, max(error_rates) + 10)

    for bar, rate in zip(bars, error_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"\n✅ Error rates plot saved to {save_path}")


def analyze_by_group(misclassified_sets: dict, all_results: list):
    """
    Analyze overlap grouped by embedding and by strategy.
    
    Args:
        misclassified_sets: Dictionary of misclassified index sets
        all_results: List of experiment result dictionaries
    """
    experiment_names = list(misclassified_sets.keys())

    print("\n" + "=" * 80)
    print("OVERLAP ANALYSIS BY GROUP")
    print("=" * 80)

    # Same embedding, different strategy
    print("\n📊 SAME EMBEDDING, DIFFERENT STRATEGY:")
    embedding_groups = {
        "text-embedding-3-large": [],
        "text-embedding-3-large_reduced_256": [],
        "text_embedding_ada": [],
        "text_embedding_3_small": [],
    }
    for name in experiment_names:
        for emb in embedding_groups:
            if emb in name:
                embedding_groups[emb].append(name)

    for emb, names in embedding_groups.items():
        if len(names) == 2:
            set_os = misclassified_sets[names[0]]
            set_w = misclassified_sets[names[1]]
            intersection = len(set_os & set_w)
            union = len(set_os | set_w)
            jaccard = intersection / union if union > 0 else 0
            print(f"\n{emb}:")
            print(f"  OS errors: {len(set_os)}, W99 errors: {len(set_w)}")
            print(f"  Common: {intersection} ({jaccard * 100:.1f}% Jaccard)")

    # Same strategy, different embeddings
    print("\n\n📊 SAME STRATEGY, DIFFERENT EMBEDDINGS:")
    os_names = [n for n in experiment_names if "oversample" in n]
    w_names = [n for n in experiment_names if "weighted" in n]

    if os_names:
        common_os = misclassified_sets[os_names[0]]
        for name in os_names[1:]:
            common_os = common_os & misclassified_sets[name]
        print(f"\nAll Oversample models agree on: {len(common_os)} samples")

    if w_names:
        common_w = misclassified_sets[w_names[0]]
        for name in w_names[1:]:
            common_w = common_w & misclassified_sets[name]
        print(f"\nAll Weighted models agree on: {len(common_w)} samples")

    # Strategy-specific
    os_experiments = [r for r in all_results if r["strategy"] == "oversample_1to1"]
    w_experiments = [r for r in all_results if r["strategy"] == "weighted_99to1"]

    all_unique_indices = set()
    for s in misclassified_sets.values():
        all_unique_indices.update(s)

    os_only = set()
    for idx in all_unique_indices:
        mis_os = all(idx in r["fp_indices"] or idx in r["fn_indices"] for r in os_experiments)
        mis_w = any(idx in r["fp_indices"] or idx in r["fn_indices"] for r in w_experiments)
        if mis_os and not mis_w:
            os_only.add(idx)

    w_only = set()
    for idx in all_unique_indices:
        mis_os = any(idx in r["fp_indices"] or idx in r["fn_indices"] for r in os_experiments)
        mis_w = all(idx in r["fp_indices"] or idx in r["fn_indices"] for r in w_experiments)
        if mis_w and not mis_os:
            w_only.add(idx)

    print(f"\n📌 OS-only misclassifications: {len(os_only)}")
    print(f"📌 W99-only misclassifications: {len(w_only)}")


def run_full_misclassification_analysis(labled_data: pd.DataFrame):
    """
    Run the complete misclassification analysis pipeline.
    
    Args:
        labled_data: DataFrame with embedding columns and labels
    
    Returns:
        Tuple of (all_results, misclassified_sets, common_all)
    """
    # Step 1: Run experiments
    all_results = run_all_experiments_and_collect_errors(labled_data)

    # Step 2: Print summary
    print_experiment_summary(all_results)

    # Step 3: Analyze overlap
    misclassified_sets, common_all = analyze_misclassification_overlap(all_results, labled_data)

    # Step 4: Plot heatmap
    plot_overlap_heatmap(misclassified_sets)

    # Step 5: Plot error rates
    plot_error_rates_comparison(all_results)

    # Step 6: Analyze by group
    analyze_by_group(misclassified_sets, all_results)

    # Step 7: Save summary
    summary_data = []
    for r in all_results:
        if "oversample" in r["experiment"]:
            strategy = "Oversample 1:1"
        else:
            strategy = "Weighted 99:1"
        if "text-embedding-3-large_reduced" in r["experiment"]:
            embedding = "text-embedding-3-large (256 dim)"
        elif "text-embedding-3-large" in r["experiment"]:
            embedding = "text-embedding-3-large"
        elif "ada" in r["experiment"]:
            embedding = "text-embedding-ada"
        elif "small" in r["experiment"]:
            embedding = "text-embedding-3-small"
        else:
            embedding = r["experiment"]

        summary_data.append({
            "Embedding": embedding,
            "Strategy": strategy,
            "Test Samples": r["test_size"],
            "Errors": r["n_fp"] + r["n_fn"],
            "Error Rate (%)": f"{(r['n_fp'] + r['n_fn']) / r['test_size'] * 100:.1f}%" if r["test_size"] > 0 else "N/A",
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("misclassification_overlap_summary.csv", index=False)
    print("\n✅ Summary saved to 'misclassification_overlap_summary.csv'")

    return all_results, misclassified_sets, common_all
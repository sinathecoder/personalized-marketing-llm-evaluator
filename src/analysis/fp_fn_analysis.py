"""
False Positive (FP) and False Negative (FN) analysis across all experiments.
Identifies which samples are consistently misclassified and why.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping

from src.config import (
    EXPERIMENTS_CONFIG, WEIGHTED_CLASS_WEIGHT, RANDOM_STATE
)
from src.models.model_utils import build_model


def run_single_experiment(
    labled_data: pd.DataFrame, embedding_col: str, strategy: str,
    exp_name: str, random_state: int = RANDOM_STATE
) -> dict:
    """
    Run one experiment and return detailed results with FP/FN analysis.
    
    Args:
        labled_data: DataFrame with data
        embedding_col: Embedding column name
        strategy: 'oversample_1to1' or 'weighted_99to1'
        exp_name: Display name
        random_state: Random seed
    
    Returns:
        Dictionary with experiment results
    """
    X = np.array([np.array(x) for x in labled_data[embedding_col].values])
    y = labled_data["R"].astype(int).values
    indices = np.arange(len(y))

    # Split data
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
        X_res, y_res = [], []
        for c in classes:
            X_c = X_train_scaled[y_train == c]
            y_c = y_train[y_train == c]
            X_up, y_up = resample(X_c, y_c, replace=True, n_samples=max_count, random_state=random_state)
            X_res.append(X_up)
            y_res.append(y_up)
        X_train_balanced = np.vstack(X_res)
        y_train_balanced = np.hstack(y_res)
        class_weight = None
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
        class_weight = WEIGHTED_CLASS_WEIGHT

    # Train
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

    # Results DataFrame
    results = pd.DataFrame({
        "idx": idx_test,
        "true_label": y_test,
        "pred_label": y_pred,
        "probability": y_prob,
    })
    results["error_type"] = "correct"
    results.loc[(results["true_label"] == 0) & (results["pred_label"] == 1), "error_type"] = "false_positive"
    results.loc[(results["true_label"] == 1) & (results["pred_label"] == 0), "error_type"] = "false_negative"

    fp_indices = set(results[results["error_type"] == "false_positive"]["idx"].values)
    fn_indices = set(results[results["error_type"] == "false_negative"]["idx"].values)

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
        "predictions": results,
    }


def run_fp_fn_analysis(labled_data: pd.DataFrame) -> dict:
    """
    Run complete FP/FN analysis across all experiments.
    
    Args:
        labled_data: DataFrame with data
    
    Returns:
        Dictionary with all analysis results
    """
    print("=" * 80)
    print("FALSE POSITIVE AND FALSE NEGATIVE OVERLAP ANALYSIS")
    print("=" * 80)

    # Run all experiments
    all_results = []
    print("\n📊 Running all 8 experiments...")
    for emb_col, strategy, exp_name in EXPERIMENTS_CONFIG:
        print(f"   Running: {exp_name}...")
        result = run_single_experiment(labled_data, emb_col, strategy, exp_name)
        all_results.append(result)
        print(f"      FP: {result['n_fp']}, FN: {result['n_fn']}, Acc: {result['accuracy']:.3f}, F1: {result['f1']:.3f}")

    print("\n✅ All experiments completed!")

    # Summary DataFrame
    summary_data = []
    for r in all_results:
        strategy_label = "OS" if "OS" in r["experiment"] else "W99"
        summary_data.append({
            "Experiment": r["experiment"],
            "Strategy": strategy_label,
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

    # Collect all FP/FN indices
    all_fp_indices = set()
    all_fn_indices = set()
    for r in all_results:
        all_fp_indices.update(r["fp_indices"])
        all_fn_indices.update(r["fn_indices"])

    # FP frequency
    fp_sample_counts = {}
    for r in all_results:
        for idx in r["fp_indices"]:
            fp_sample_counts[idx] = fp_sample_counts.get(idx, 0) + 1
    sorted_fp = sorted(fp_sample_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n📊 Total unique FP samples: {len(all_fp_indices)}")
    fp_freq_dist = {}
    for _idx, count in sorted_fp:
        fp_freq_dist[count] = fp_freq_dist.get(count, 0) + 1
    for count in sorted(fp_freq_dist.keys(), reverse=True):
        print(f"   FP by {count}/8 models: {fp_freq_dist[count]} samples")

    # FN frequency
    fn_sample_counts = {}
    for r in all_results:
        for idx in r["fn_indices"]:
            fn_sample_counts[idx] = fn_sample_counts.get(idx, 0) + 1
    sorted_fn = sorted(fn_sample_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n📊 Total unique FN samples: {len(all_fn_indices)}")
    fn_freq_dist = {}
    for _idx, count in sorted_fn:
        fn_freq_dist[count] = fn_freq_dist.get(count, 0) + 1
    for count in sorted(fn_freq_dist.keys(), reverse=True):
        print(f"   FN by {count}/8 models: {fn_freq_dist[count]} samples")

    # Strategy overlap analysis
    print("\n" + "=" * 80)
    print("STRATEGY OVERLAP ANALYSIS")
    print("=" * 80)

    embeddings_map = {}
    for r in all_results:
        emb_key = r["embedding"]
        if emb_key not in embeddings_map:
            embeddings_map[emb_key] = {}
        embeddings_map[emb_key][r["strategy"]] = r

    overlap_rows = []
    for emb_key, strategies in embeddings_map.items():
        if "oversample_1to1" in strategies and "weighted_99to1" in strategies:
            os_r = strategies["oversample_1to1"]
            w_r = strategies["weighted_99to1"]

            fp_common = len(os_r["fp_indices"] & w_r["fp_indices"])
            fp_union = len(os_r["fp_indices"] | w_r["fp_indices"])
            fp_j = fp_common / fp_union if fp_union > 0 else 0

            fn_common = len(os_r["fn_indices"] & w_r["fn_indices"])
            fn_union = len(os_r["fn_indices"] | w_r["fn_indices"])
            fn_j = fn_common / fn_union if fn_union > 0 else 0

            if "3_small" in emb_key:
                emb_display = "text-embedding-3-small"
            elif "ada" in emb_key:
                emb_display = "text-embedding-ada"
            elif "256" in emb_key:
                emb_display = "text-embedding-3-large (256 dim)"
            else:
                emb_display = "text-embedding-3-large"

            overlap_rows.append({
                "Embedding": emb_display,
                "OS_FP": os_r["n_fp"],
                "W99_FP": w_r["n_fp"],
                "FP_Common": fp_common,
                "FP_Jaccard": f"{fp_j * 100:.1f}%",
                "OS_FN": os_r["n_fn"],
                "W99_FN": w_r["n_fn"],
                "FN_Common": fn_common,
                "FN_Jaccard": f"{fn_j * 100:.1f}%",
            })

    overlap_df = pd.DataFrame(overlap_rows)
    print("\n📊 Strategy Overlap:")
    print(overlap_df.to_string(index=False))

    # Consistent misclassifications
    all_indices = all_fp_indices | all_fn_indices
    consistent = []
    for idx in all_indices:
        fp_count = sum(1 for r in all_results if idx in r["fp_indices"])
        fn_count = sum(1 for r in all_results if idx in r["fn_indices"])
        total = fp_count + fn_count
        if total == len(all_results):
            consistent.append({"index": idx, "fp_count": fp_count, "fn_count": fn_count})

    print(f"\n🔍 Samples misclassified by ALL 8 models: {len(consistent)}")
    for item in consistent[:5]:
        idx = item["index"]
        copy_text = labled_data.iloc[idx]["marketing_copy"]
        preview = copy_text[:200] + "..." if len(copy_text) > 200 else copy_text
        violations = [d for d in ["P", "H", "C", "L"] if d in labled_data.columns and labled_data.iloc[idx][d] == 1]
        etype = "FP" if item["fp_count"] > item["fn_count"] else "FN"
        print(f"\n   Index {idx} (Mostly {etype}):")
        if violations:
            print(f"   Violations: {', '.join(violations)}")
        print(f"   Copy: {preview}")

    # Visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    exp_names = [r["experiment"] for r in all_results]
    x = np.arange(len(exp_names))
    width = 0.35
    fp_counts = [r["n_fp"] for r in all_results]
    fn_counts = [r["n_fn"] for r in all_results]

    bars1 = ax.bar(x - width / 2, fp_counts, width, label="False Positives", color="orange", edgecolor="black")
    bars2 = ax.bar(x + width / 2, fn_counts, width, label="False Negatives", color="red", edgecolor="black")

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Count")
    ax.set_title("FP and FN by Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.annotate(f"{int(bar.get_height())}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.annotate(f"{int(bar.get_height())}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("fp_fn_by_experiment.png", dpi=300)
    plt.show()

    # Error distribution histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fp_freqs = list(fp_sample_counts.values())
    axes[0].hist(fp_freqs, bins=range(1, len(all_results) + 2), align="left", rwidth=0.8, color="orange", edgecolor="black")
    axes[0].set_xlabel("Models that Flagged as FP")
    axes[0].set_ylabel("Samples")
    axes[0].set_title("FP Frequency Distribution")
    axes[0].set_xticks(range(1, len(all_results) + 1))

    fn_freqs = list(fn_sample_counts.values())
    axes[1].hist(fn_freqs, bins=range(1, len(all_results) + 2), align="left", rwidth=0.8, color="red", edgecolor="black")
    axes[1].set_xlabel("Models that Missed as FN")
    axes[1].set_ylabel("Samples")
    axes[1].set_title("FN Frequency Distribution")
    axes[1].set_xticks(range(1, len(all_results) + 1))

    plt.tight_layout()
    plt.savefig("misclassification_frequency.png", dpi=300)
    plt.show()

    # Export CSVs
    fp_summary = []
    for idx, count in sorted_fp:
        fp_summary.append({
            "sample_index": idx,
            "times_FP": count,
            "pct": f"{count / len(all_results) * 100:.1f}%",
            "copy": labled_data.iloc[idx]["marketing_copy"][:200],
        })
    pd.DataFrame(fp_summary).to_csv("false_positives_summary.csv", index=False)
    print("\n✅ Saved: false_positives_summary.csv")

    fn_summary = []
    for idx, count in sorted_fn:
        violations = [d for d in ["P", "H", "C", "L"] if d in labled_data.columns and labled_data.iloc[idx][d] == 1]
        fn_summary.append({
            "sample_index": idx,
            "times_FN": count,
            "pct": f"{count / len(all_results) * 100:.1f}%",
            "violations": ", ".join(violations) if violations else "None",
            "copy": labled_data.iloc[idx]["marketing_copy"][:200],
        })
    pd.DataFrame(fn_summary).to_csv("false_negatives_summary.csv", index=False)
    print("✅ Saved: false_negatives_summary.csv")

    overlap_df.to_csv("strategy_overlap_summary.csv", index=False)
    print("✅ Saved: strategy_overlap_summary.csv")
    summary_df.to_csv("experiment_summary.csv", index=False)
    print("✅ Saved: experiment_summary.csv")

    return {
        "all_results": all_results,
        "all_fp_indices": all_fp_indices,
        "all_fn_indices": all_fn_indices,
        "fp_sample_counts": fp_sample_counts,
        "fn_sample_counts": fn_sample_counts,
        "overlap_df": overlap_df,
        "consistent": consistent,
    }
"""
Embedding interpretability analysis - analyzing how embeddings distinguish
between high-quality and low-quality marketing content.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping

from src.config import EMBEDDING_COLS, STRATEGIES, WEIGHTED_CLASS_WEIGHT, RANDOM_STATE
from src.models.model_utils import build_model


def run_single_interpretability_experiment(
    labled_data: pd.DataFrame, embedding_col: str, strategy: str
) -> dict:
    """
    Train a model for a specific experiment and return data for interpretability analysis.
    
    Args:
        labled_data: DataFrame with data
        embedding_col: Column name for embeddings
        strategy: 'oversample_1to1' or 'weighted_99to1'
    
    Returns:
        Dictionary with test embeddings, labels, predictions, etc.
    """
    print(f"\n{'=' * 60}")
    print(f"Running interpretability: {embedding_col} - {strategy}")
    print(f"{'=' * 60}")

    X = np.array([np.array(x) for x in labled_data[embedding_col].values])
    y = labled_data["R"].astype(int).values
    indices = np.arange(len(y))

    # Split data
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, indices, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.80, stratify=y_temp, random_state=RANDOM_STATE
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
            X_up, y_up = resample(X_c, y_c, replace=True, n_samples=max_count, random_state=RANDOM_STATE)
            X_res.append(X_up)
            y_res.append(y_up)
        X_train_balanced = np.vstack(X_res)
        y_train_balanced = np.hstack(y_res)
        class_weight = None
        print(f"  Oversampled: {len(y_train)} -> {len(y_train_balanced)}")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
        class_weight = WEIGHTED_CLASS_WEIGHT
        print(f"  Weighted: class_weight = {class_weight}")

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

    return {
        "embedding_col": embedding_col,
        "strategy": strategy,
        "X_test": X_test_scaled,
        "X_test_original": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "idx_test": idx_test,
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def analyze_embedding_similarity_patterns(results_dict: dict, labled_data: pd.DataFrame) -> dict:
    """
    Analyze cosine similarity patterns for a specific experiment.
    Computes within-class and between-class similarity, PCA visualization, etc.
    
    Args:
        results_dict: Output from run_single_interpretability_experiment
        labled_data: Original DataFrame with marketing_copy
    
    Returns:
        Dictionary with similarity statistics
    """
    X_test = results_dict["X_test_original"]
    y_test = results_dict["y_test"]
    idx_test = results_dict["idx_test"]

    test_copies = labled_data.iloc[idx_test]["marketing_copy"].values

    print(f"\n{'=' * 70}")
    print(f"EMBEDDING INTERPRETABILITY ANALYSIS")
    print(f"Experiment: {results_dict['embedding_col']} - {results_dict['strategy']}")
    print(f"{'=' * 70}")

    # Cosine similarity matrix
    sim_matrix = cosine_similarity(X_test)

    # Within vs between class similarity
    within_high = []
    within_low = []
    between = []

    for i in range(len(y_test)):
        for j in range(i + 1, len(y_test)):
            sim = sim_matrix[i, j]
            if y_test[i] == 0 and y_test[j] == 0:
                within_high.append(sim)
            elif y_test[i] == 1 and y_test[j] == 1:
                within_low.append(sim)
            else:
                between.append(sim)

    within_high_mean = np.mean(within_high) if within_high else 0
    within_low_mean = np.mean(within_low) if within_low else 0
    between_mean = np.mean(between) if between else 0

    discriminative_ratio = (
        (within_high_mean + within_low_mean) / (2 * between_mean) if between_mean > 0 else 0
    )

    print(f"\n📈 Cosine Similarity Statistics:")
    print(f"   Within High:    mean={within_high_mean:.3f}, std={np.std(within_high):.3f}")
    print(f"   Within Low:     mean={within_low_mean:.3f}, std={np.std(within_low):.3f}")
    print(f"   Between:        mean={between_mean:.3f}, std={np.std(between):.3f}")
    print(f"\n🎯 Discriminative Ratio: {discriminative_ratio:.2f}")

    if discriminative_ratio > 1.5:
        print("   -> Excellent: distinct quality clusters")
    elif discriminative_ratio > 1.2:
        print("   -> Good: distinguishes quality classes well")
    elif discriminative_ratio > 1.0:
        print("   -> Moderate: some overlap")
    else:
        print("   -> Poor: does not clearly distinguish quality")

    # Visualize similarity distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(within_high, bins=30, alpha=0.5, label="Within High", color="green")
    axes[0].hist(within_low, bins=30, alpha=0.5, label="Within Low", color="red")
    axes[0].hist(between, bins=30, alpha=0.5, label="Between", color="gray")
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Similarity Distributions")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    bp = axes[1].boxplot(
        [within_high, within_low, between],
        labels=["Within High", "Within Low", "Between"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("lightgreen")
    bp["boxes"][1].set_facecolor("salmon")
    bp["boxes"][2].set_facecolor("lightgray")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("Similarity by Pair Type")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_name = (
        f"similarity_distributions_{results_dict['embedding_col']}_{results_dict['strategy']}.png"
    )
    plt.savefig(save_name, dpi=300)
    plt.show()
    print(f"✅ Saved: {save_name}")

    # PCA visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(
        embeddings_2d[y_test == 0, 0], embeddings_2d[y_test == 0, 1],
        c="green", alpha=0.6, s=30, label="High-Quality",
    )
    axes[0].scatter(
        embeddings_2d[y_test == 1, 0], embeddings_2d[y_test == 1, 1],
        c="red", alpha=0.6, s=30, label="Low-Quality",
    )
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[0].set_title("PCA: High vs Low Quality")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Color by Coherence if available
    if "C" in labled_data.columns:
        quality_col = labled_data.iloc[idx_test]["C"].values
        scatter = axes[1].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=quality_col, cmap="RdYlGn", alpha=0.6, s=30,
        )
        axes[1].set_title("PCA: Color = Coherence")
        plt.colorbar(scatter, ax=axes[1], label="Coherence (0=No, 1=Yes)")
    else:
        axes[1].hexbin(embeddings_2d[:, 0], embeddings_2d[:, 1], gridsize=30, cmap="Blues")
        axes[1].set_title("PCA: Density")
        plt.colorbar(label="Density")

    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_name = f"pca_visualization_{results_dict['embedding_col']}_{results_dict['strategy']}.png"
    plt.savefig(save_name, dpi=300)
    plt.show()
    print(f"✅ Saved: {save_name}")

    # Centroid analysis
    centroid_high = X_test[y_test == 0].mean(axis=0) if sum(y_test == 0) > 0 else None
    centroid_low = X_test[y_test == 1].mean(axis=0) if sum(y_test == 1) > 0 else None

    if centroid_high is not None and centroid_low is not None:
        centroid_dist = 1 - cosine_similarity(
            centroid_high.reshape(1, -1), centroid_low.reshape(1, -1)
        )[0][0]
        print(f"\n📐 Centroid distance: {centroid_dist:.3f}")

        # Prototypical examples
        dist_to_high = cosine_similarity(X_test, centroid_high.reshape(1, -1)).flatten()
        closest_to_high = np.argsort(dist_to_high)[-3:][::-1]

        print("\n⭐ PROTOTYPICAL HIGH-QUALITY:")
        for i, idx in enumerate(closest_to_high):
            print(f"  {i + 1}. (sim={dist_to_high[idx]:.3f}) {test_copies[idx][:120]}...")

    # Most similar pairs
    triu = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[triu]
    sorted_idx = np.argsort(similarities)[::-1]

    print("\n🔗 MOST SIMILAR PAIRS:")
    for i in range(min(3, len(sorted_idx))):
        idx1 = triu[0][sorted_idx[i]]
        idx2 = triu[1][sorted_idx[i]]
        sim = similarities[sorted_idx[i]]
        print(f"\n  Pair {i + 1} (sim={sim:.3f}):")
        print(f"    A: {test_copies[idx1][:100]}...")
        print(f"    B: {test_copies[idx2][:100]}...")
        print(f"    Labels: {y_test[idx1]} -> {y_test[idx2]}")

    return {
        "sim_matrix": sim_matrix,
        "within_high": within_high,
        "within_low": within_low,
        "between": between,
        "discriminative_ratio": discriminative_ratio,
        "pca_components": embeddings_2d,
        "pca_variance": pca.explained_variance_ratio_,
    }


def run_interpretability_for_selected_experiments(
    labled_data: pd.DataFrame, max_experiments: int = 2
) -> dict:
    """
    Run embedding interpretability for selected experiments.
    By default runs one oversample and one weighted for text-embedding-3-small.
    
    Args:
        labled_data: DataFrame with data
        max_experiments: Max experiments to run
    
    Returns:
        Dictionary with interpretability results
    """
    all_results = {}
    selected_strategies = []
    if "oversample_1to1" in STRATEGIES:
        selected_strategies.append("oversample_1to1")
    if "weighted_99to1" in STRATEGIES and len(selected_strategies) < max_experiments:
        selected_strategies.append("weighted_99to1")

    # Select one embedding
    selected_embedding = None
    for col, name in EMBEDDING_COLS:
        if "3_small" in col or "small" in name.lower():
            selected_embedding = (col, name)
            break
    if selected_embedding is None:
        selected_embedding = EMBEDDING_COLS[0]

    print("\n" + "🔍" * 30)
    print(f"Selected embedding: {selected_embedding[1]}")
    print(f"Selected strategies: {selected_strategies}")
    print("🔍" * 30)

    for strategy in selected_strategies:
        results = run_single_interpretability_experiment(labled_data, selected_embedding[0], strategy)
        similarity_stats = analyze_embedding_similarity_patterns(results, labled_data)
        all_results[f"{selected_embedding[1]}_{strategy}"] = {
            "results": results,
            "similarity_stats": similarity_stats,
        }

    return all_results


def compare_embeddings_similarity(labled_data: pd.DataFrame, strategy: str = "oversample_1to1") -> pd.DataFrame:
    """
    Compare cosine similarity patterns across different embedding models with the same strategy.
    
    Args:
        labled_data: DataFrame with data
        strategy: Training strategy to use
    
    Returns:
        DataFrame with comparison table
    """
    print("\n" + "=" * 70)
    print(f"COMPARING EMBEDDINGS (Strategy: {strategy})")
    print("=" * 70)

    comparison_data = []

    for col, name in EMBEDDING_COLS:
        print(f"\n   Processing: {name}")
        results = run_single_interpretability_experiment(labled_data, col, strategy)

        X_test = results["X_test_original"]
        y_test = results["y_test"]

        sim_matrix = cosine_similarity(X_test)

        within_high, within_low, between = [], [], []
        for i in range(len(y_test)):
            for j in range(i + 1, len(y_test)):
                sim = sim_matrix[i, j]
                if y_test[i] == 0 and y_test[j] == 0:
                    within_high.append(sim)
                elif y_test[i] == 1 and y_test[j] == 1:
                    within_low.append(sim)
                else:
                    between.append(sim)

        wh_mean = np.mean(within_high) if within_high else 0
        wl_mean = np.mean(within_low) if within_low else 0
        b_mean = np.mean(between) if between else 0
        dr = (wh_mean + wl_mean) / (2 * b_mean) if b_mean > 0 else 0

        comparison_data.append({
            "Embedding": name,
            "Within-High": f"{wh_mean:.3f}",
            "Within-Low": f"{wl_mean:.3f}",
            "Between": f"{b_mean:.3f}",
            "Disc. Ratio": f"{dr:.2f}",
            "Accuracy": f"{results['accuracy']:.3f}",
            "F1": f"{results['f1']:.3f}",
        })

    comp_df = pd.DataFrame(comparison_data)
    print("\n📊 EMBEDDING COMPARISON:")
    print(comp_df.to_string(index=False))
    comp_df.to_csv("embedding_comparison_similarity.csv", index=False)
    print("\n✅ Saved: embedding_comparison_similarity.csv")
    return comp_df
#!/usr/bin/env python
"""
K-Fold Cross-Validation Split for RNA Inverse Folding.

Creates K stratified splits with cluster-wise splitting (no data leakage)
and length-stratified distribution across folds.
"""

import os
import pandas as pd
import numpy as np
import torch

# Configuration
DATA_PATH = os.environ.get("DATA_PATH", "./")
RANDOM_SEED = 42
K_FOLDS = 5
SHORT_MAX = 100
MEDIUM_MAX = 200


def get_len_bin(length):
    """Assign length bin to a sequence."""
    if length <= SHORT_MAX:
        return "short"
    elif length <= MEDIUM_MAX:
        return "medium"
    return "long"


def stratified_kfold_clusters(cluster_stats, k=5, random_state=42):
    """Perform stratified K-fold split at the cluster level."""
    np.random.seed(random_state)
    folds = [[] for _ in range(k)]
    fold_seq_counts = [0] * k

    for bin_name in ["short", "medium", "long"]:
        bin_clusters = cluster_stats[cluster_stats["bin_label"] == bin_name].copy()
        if len(bin_clusters) == 0:
            continue
        bin_clusters = bin_clusters.sample(frac=1, random_state=random_state + hash(bin_name) % 1000)
        bin_clusters = bin_clusters.sort_values("seq_count", ascending=False)

        for _, row in bin_clusters.iterrows():
            min_fold_idx = np.argmin(fold_seq_counts)
            folds[min_fold_idx].append(row["safe_cluster"])
            fold_seq_counts[min_fold_idx] += row["seq_count"]

    return folds


def create_kfold_splits(df, cluster_stats, k=5, random_state=42):
    """Create K-fold cross-validation splits."""
    cluster_folds = stratified_kfold_clusters(cluster_stats, k, random_state)
    cluster_to_indices = df.groupby("safe_cluster").apply(lambda x: x.index.tolist()).to_dict()

    index_folds = []
    for fold_clusters in cluster_folds:
        fold_indices = []
        for cluster_id in fold_clusters:
            fold_indices.extend(cluster_to_indices[cluster_id])
        index_folds.append(fold_indices)

    splits = []
    for i in range(k):
        test_idx = index_folds[i]
        val_idx = index_folds[(i + 1) % k]
        train_idx = []
        for j in range(k):
            if j != i and j != (i + 1) % k:
                train_idx.extend(index_folds[j])
        splits.append((train_idx, val_idx, test_idx))

    return splits, cluster_folds


def validate_split(df, train_idx, val_idx, test_idx, fold_num):
    """Validate a single split for correctness."""
    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)

    assert len(train_set & val_set) == 0, f"Fold {fold_num}: Train/Val overlap"
    assert len(train_set & test_set) == 0, f"Fold {fold_num}: Train/Test overlap"
    assert len(val_set & test_set) == 0, f"Fold {fold_num}: Val/Test overlap"
    assert len(train_set) + len(val_set) + len(test_set) == len(df), f"Fold {fold_num}: Missing samples"

    train_clusters = set(df.loc[train_idx, "safe_cluster"])
    val_clusters = set(df.loc[val_idx, "safe_cluster"])
    test_clusters = set(df.loc[test_idx, "safe_cluster"])

    assert len(train_clusters & val_clusters) == 0, f"Fold {fold_num}: Cluster leakage Train/Val"
    assert len(train_clusters & test_clusters) == 0, f"Fold {fold_num}: Cluster leakage Train/Test"
    assert len(val_clusters & test_clusters) == 0, f"Fold {fold_num}: Cluster leakage Val/Test"


def print_split_statistics(df, splits, k):
    """Print statistics for all K splits."""
    total_n = len(df)
    print(f"\nK-Fold Statistics (K={k}), Total: {total_n} sequences, {df['safe_cluster'].nunique()} clusters")

    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        print(f"  Fold {i+1}: Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)}")


if __name__ == "__main__":
    print("Creating K-Fold Cross-Validation Splits...")

    # Load data
    df = pd.read_csv(os.path.join(DATA_PATH, "processed_df.csv"))
    print(f"Loaded {len(df)} sequences")

    # Process cluster IDs (handle NaN)
    df["safe_cluster"] = df["cluster_structsim0.45"].fillna(-1)
    nan_mask = df["safe_cluster"] == -1
    if nan_mask.sum() > 0:
        max_id = df["safe_cluster"].max()
        df.loc[nan_mask, "safe_cluster"] = range(int(max_id) + 1, int(max_id) + 1 + nan_mask.sum())

    # Length binning
    df["len_bin"] = df["length"].apply(get_len_bin)

    # Cluster statistics
    cluster_stats = df.groupby("safe_cluster").agg(
        seq_count=("sequence", "count"),
        max_len=("length", "max")
    ).reset_index()
    cluster_stats["bin_label"] = cluster_stats["max_len"].apply(get_len_bin)

    # Create and validate splits
    splits, cluster_folds = create_kfold_splits(df, cluster_stats, K_FOLDS, RANDOM_SEED)
    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        validate_split(df, train_idx, val_idx, test_idx, i + 1)
    print("All splits validated (no leakage)")

    print_split_statistics(df, splits, K_FOLDS)

    # Save splits
    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        torch.save((train_idx, val_idx, test_idx), os.path.join(DATA_PATH, f"kfold_{i+1}_split.pt"))

    # Save metadata
    metadata = {
        "k_folds": K_FOLDS,
        "random_seed": RANDOM_SEED,
        "total_sequences": len(df),
        "total_clusters": df["safe_cluster"].nunique(),
        "cluster_folds": cluster_folds,
        "split_sizes": [(len(t), len(v), len(te)) for t, v, te in splits]
    }
    torch.save(metadata, os.path.join(DATA_PATH, "kfold_metadata.pt"))
    print(f"Saved {K_FOLDS} split files to {DATA_PATH}")

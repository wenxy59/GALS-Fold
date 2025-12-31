#!/usr/bin/env python
"""
K-Fold Split Visualization and Export.

Generates visualizations for K-fold cross-validation splits and exports
test set PDB IDs to text files.
"""

import os
import ast
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Configuration
DATA_PATH = os.environ.get("DATA_PATH", "./")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./statistics/")
K_FOLDS = 5
SHORT_MAX = 100
MEDIUM_MAX = 200

COLORS = {
    'train': '#3498db', 'val': '#f39c12', 'test': '#e74c3c',
    'short': '#2ecc71', 'medium': '#9b59b6', 'long': '#e67e22',
}
FOLD_COLORS = ['#1abc9c', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']


def get_len_bin(length):
    """Assign length bin to a sequence."""
    if length <= SHORT_MAX:
        return "short"
    elif length <= MEDIUM_MAX:
        return "medium"
    return "long"


def load_all_splits(data_path, k_folds):
    """Load all K-fold split files."""
    splits = {}
    for i in range(1, k_folds + 1):
        split_path = os.path.join(data_path, f"kfold_{i}_split.pt")
        if os.path.exists(split_path):
            train_idx, val_idx, test_idx = torch.load(split_path, weights_only=False)
            splits[i] = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    return splits


def export_test_ids(df, splits, k_folds, output_dir):
    """Export test set PDB IDs to text files."""
    os.makedirs(output_dir, exist_ok=True)
    for fold_id in range(1, k_folds + 1):
        test_ids = []
        for idx in splits[fold_id]['test']:
            id_list = ast.literal_eval(df.loc[idx, 'id_list'])
            test_ids.append(id_list[0])
        output_path = os.path.join(output_dir, f"kfold_{fold_id}_test_index.txt")
        with open(output_path, "w") as f:
            f.write("\n".join(test_ids))
        print(f"Saved {output_path} ({len(test_ids)} IDs)")


def create_summary_table(df, splits, k_folds):
    """Create a summary DataFrame with statistics for each fold."""
    summary_data = []
    for fold_id in range(1, k_folds + 1):
        if fold_id not in splits:
            continue
        for split_name in ['train', 'val', 'test']:
            indices = splits[fold_id][split_name]
            split_df = df.loc[indices]
            summary_data.append({
                'Fold': fold_id,
                'Split': split_name.capitalize(),
                'Count': len(indices),
                'Percentage': len(indices) / len(df) * 100,
                'Short': (split_df['len_bin'] == 'short').sum(),
                'Medium': (split_df['len_bin'] == 'medium').sum(),
                'Long': (split_df['len_bin'] == 'long').sum(),
                'Mean_Length': split_df['length'].mean(),
                'Clusters': split_df['safe_cluster'].nunique()
            })
    return pd.DataFrame(summary_data)


def plot_fold_summary(df, splits, k_folds, save_path):
    """Plot comprehensive fold comparison summary."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Data quantity by fold
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(k_folds)
    width = 0.25
    for i, split_name in enumerate(['train', 'val', 'test']):
        counts = [len(splits[f][split_name]) for f in range(1, k_folds + 1)]
        ax1.bar(x + (i - 1) * width, counts, width, label=split_name.capitalize(), color=COLORS[split_name])
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Count')
    ax1.set_title('Data Quantity by Fold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in range(1, k_folds + 1)])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Test set length distributions
    ax2 = fig.add_subplot(gs[0, 1:])
    for fold_id in range(1, k_folds + 1):
        test_df = df.loc[splits[fold_id]['test']]
        ax2.hist(test_df['length'], bins=50, alpha=0.4, label=f'Fold {fold_id}', color=FOLD_COLORS[fold_id - 1])
    ax2.axvline(x=SHORT_MAX, color='gray', linestyle='--')
    ax2.axvline(x=MEDIUM_MAX, color='gray', linestyle='--')
    ax2.set_xlabel('Sequence Length (nt)')
    ax2.set_ylabel('Count')
    ax2.set_title('Test Set Length Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Cluster distribution
    ax3 = fig.add_subplot(gs[1, 0])
    for i, split_name in enumerate(['train', 'val', 'test']):
        clusters = [df.loc[splits[f][split_name]]['safe_cluster'].nunique() for f in range(1, k_folds + 1)]
        ax3.bar(x + (i - 1) * width, clusters, width, label=split_name.capitalize(), color=COLORS[split_name])
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Clusters')
    ax3.set_title('Cluster Distribution')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(i) for i in range(1, k_folds + 1)])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. Mean length by fold
    ax4 = fig.add_subplot(gs[1, 1])
    for split_name in ['train', 'val', 'test']:
        means = [df.loc[splits[f][split_name]]['length'].mean() for f in range(1, k_folds + 1)]
        ax4.plot(x, means, 'o-', label=split_name.capitalize(), color=COLORS[split_name], linewidth=2)
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Mean Length (nt)')
    ax4.set_title('Mean Sequence Length')
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(i) for i in range(1, k_folds + 1)])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Length category % in test sets
    ax5 = fig.add_subplot(gs[1, 2])
    for fold_id in range(1, k_folds + 1):
        test_df = df.loc[splits[fold_id]['test']]
        total = len(test_df)
        short_pct = (test_df['len_bin'] == 'short').sum() / total * 100
        med_pct = (test_df['len_bin'] == 'medium').sum() / total * 100
        long_pct = (test_df['len_bin'] == 'long').sum() / total * 100
        ax5.bar(fold_id - 1, short_pct, color=COLORS['short'], label='Short' if fold_id == 1 else '')
        ax5.bar(fold_id - 1, med_pct, bottom=short_pct, color=COLORS['medium'], label='Medium' if fold_id == 1 else '')
        ax5.bar(fold_id - 1, long_pct, bottom=short_pct + med_pct, color=COLORS['long'], label='Long' if fold_id == 1 else '')
    ax5.set_xlabel('Fold')
    ax5.set_ylabel('Percentage (%)')
    ax5.set_title('Test Set Length Categories')
    ax5.set_xticks(x)
    ax5.set_xticklabels([str(i) for i in range(1, k_folds + 1)])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


def print_statistics(df, splits, k_folds):
    """Print split statistics to console."""
    print(f"\nDataset: {len(df)} sequences, {df['safe_cluster'].nunique()} clusters")
    print(f"Length: {df['length'].min()}-{df['length'].max()} nt (mean: {df['length'].mean():.1f})")
    for fold_id in range(1, k_folds + 1):
        train_n, val_n, test_n = len(splits[fold_id]['train']), len(splits[fold_id]['val']), len(splits[fold_id]['test'])
        print(f"  Fold {fold_id}: Train={train_n} Val={val_n} Test={test_n}")


if __name__ == "__main__":
    print("K-Fold Split Visualization")

    # Load data
    df = pd.read_csv(os.path.join(DATA_PATH, "processed_df.csv"))
    df["safe_cluster"] = df["cluster_structsim0.45"].fillna(-1)
    max_cluster_id = df["safe_cluster"].max()
    nan_mask = df["safe_cluster"] == -1
    if nan_mask.sum() > 0:
        df.loc[nan_mask, "safe_cluster"] = range(int(max_cluster_id) + 1, int(max_cluster_id) + 1 + nan_mask.sum())
    df["len_bin"] = df["length"].apply(get_len_bin)
    print(f"Loaded {len(df)} sequences")

    # Load splits
    splits = load_all_splits(DATA_PATH, K_FOLDS)
    print(f"Loaded {len(splits)} fold splits")

    # Print statistics
    print_statistics(df, splits, K_FOLDS)

    # Export test IDs
    export_test_ids(df, splits, K_FOLDS, OUTPUT_DIR)

    # Generate visualization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_fold_summary(df, splits, K_FOLDS, os.path.join(OUTPUT_DIR, "kfold_summary.png"))

    # Save summary table
    summary_df = create_summary_table(df, splits, K_FOLDS)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "kfold_summary.csv"), index=False)
    print(f"Saved {OUTPUT_DIR}/kfold_summary.csv")

    print("Done!")

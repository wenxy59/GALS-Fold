#!/usr/bin/env python3
"""
Complexity evaluation for GALS-Fold.

Measures inference time and memory across length bins:
  - length < 1000: bin size 50
  - length >= 1000: bin size 500
"""

import os
import sys
import time
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace
from tqdm import tqdm

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(".env")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import GeometricLongShortRNA
from src.data.featurizer import RNAGraphFeaturizer
from src.data.data_utils import get_backbone_coords
from src.constants import DATA_PATH, RNA_ATOMS, FILL_VALUE


def preprocess_rna_data(rna_data):
    """Extract backbone coordinates for model input."""
    pyrimidine_bb_indices = [RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("N1")]
    purine_bb_indices = [RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("N9")]

    # Store original full-atom coordinates
    rna_data['coords_list_full'] = rna_data['coords_list'].copy()

    coords_list = []
    for coords in rna_data['coords_list']:
        # Check if already backbone coords (3 atoms) or full atom coords (27 atoms)
        if coords.shape[1] == 3:
            # Already backbone coords
            coords_list.append(coords)
        else:
            # Full atom coords, extract backbone
            coords = get_backbone_coords(
                coords, rna_data['sequence'],
                pyrimidine_bb_indices, purine_bb_indices
            )
            # Do not add structures with missing coordinates for ALL residues
            if not torch.all((coords == FILL_VALUE).sum(axis=(1, 2)) > 0):
                coords_list.append(coords)

    if len(coords_list) > 0:
        rna_data['coords_list'] = coords_list
        return rna_data
    return None


def load_config(model_name, split='kfold_1'):
    """Load model configuration."""
    config_path = 'configs/else.yaml'
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = {}
    for key, val in config_dict.items():
        if isinstance(val, dict) and 'value' in val:
            config[key] = val['value']
        else:
            config[key] = val
    
    # Override model and split
    config['model'] = model_name
    config['split'] = split
    config['model_path'] = f'./checkpoint/{model_name}_{split}_best_checkpoint.h5'
    
    return SimpleNamespace(**config)


def get_model(config):
    """Initialize model based on config."""
    model_class = GeometricLongShortRNA

    model_kwargs = {
        'node_in_dim': tuple(config.node_in_dim),
        'node_h_dim': tuple(config.node_h_dim),
        'edge_in_dim': tuple(config.edge_in_dim),
        'edge_h_dim': tuple(config.edge_h_dim),
        'num_layers': config.num_layers,
        'drop_rate': config.drop_rate,
        'out_dim': config.out_dim,
    }

    # GALS-specific parameters
    if config.model == 'GALS':
        model_kwargs['heads'] = getattr(config, 'heads', 4)
        model_kwargs['num_anchors'] = getattr(config, 'num_anchors', 32)
        model_kwargs['local_window'] = getattr(config, 'local_window', 10)
        model_kwargs['length_threshold'] = getattr(config, 'length_threshold', 150)
        model_kwargs['aux_loss_weight'] = getattr(config, 'aux_loss_weight', 0.3)

    return model_class(**model_kwargs)


def load_model(model_name, split, device):
    """Load model with checkpoint."""
    print(f"\n{'='*60}")
    print(f"Loading {model_name} model (split: {split})")
    print(f"{'='*60}")
    
    config = load_config(model_name, split)
    model = get_model(config).to(device)
    
    # Load checkpoint
    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=False))
        print(f"✓ Loaded checkpoint: {config.model_path}")
    else:
        print(f"✗ Checkpoint not found: {config.model_path}")
        return None, None
    
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    return model, config


def measure_inference_time(model, data, n_samples, temperature, device, n_repeats=1, warmup=1):
    """Measure inference time for generating n_samples sequences."""
    times = []

    with torch.no_grad():
        # Warm-up runs (not timed)
        for _ in range(max(0, warmup)):
            _ = model.sample(data, n_samples=1, temperature=temperature, return_logits=True)

        for _ in range(max(1, n_repeats)):
            start_time = time.time()

            # Sample sequences in a single batched call
            _ = model.sample(data, n_samples=n_samples, temperature=temperature, return_logits=True)

            end_time = time.time()
            elapsed = end_time - start_time

            times.append(elapsed / n_samples)

    avg_time = np.mean(times)
    total_time = avg_time * n_samples
    std_time = np.std(times)

    return avg_time, total_time, std_time


def measure_memory_usage(model, data, device):
    """Measure peak memory usage during inference."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        with torch.no_grad():
            _ = model.sample(data, n_samples=1, temperature=0.1)

        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB

        return peak_memory
    else:
        # CPU memory tracking is more complex, return 0 for now
        return 0.0


def build_length_bins(max_len, short_bin=50, long_bin=500, threshold=1000):
    """Build length bins with finer granularity below threshold."""
    bins = []

    # Short bins: [0, threshold) with step short_bin
    for start in range(0, min(threshold, max_len + 1), short_bin):
        end = min(start + short_bin, threshold)
        if end > start:
            label_end = end
            bins.append((start, end, f"{start}-{label_end}"))

    # Long bins: [threshold, max_len] with step long_bin
    if max_len >= threshold:
        for start in range(threshold, max_len + 1, long_bin):
            end = min(start + long_bin, max_len + 1)
            label_end = max_len if end == max_len + 1 else end
            bins.append((start, end, f"{start}-{label_end}"))

    return bins


def get_length_bin(length, bins):
    """Assign length bin label based on configured bins."""
    for start, end, label in bins:
        if start <= length < end:
            return label
    return None


def main():
    parser = argparse.ArgumentParser(description='Complexity Evaluation for RNA Inverse Folding')
    parser.add_argument('--split', type=str, default='kfold_1',
                        help='Data split to use')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, cpu)')
    parser.add_argument('--n_samples', type=int, default=20,
                        help='Number of samples to generate per RNA')
    parser.add_argument('--n_repeats', type=int, default=1,
                        help='Number of repeats per RNA for timing (batched)')
    parser.add_argument('--warmup', type=int, default=1,
                        help='Number of warm-up runs before timing')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature')
    parser.add_argument('--n_rnas_per_bin', type=int, default=1,
                        help='Number of RNAs to test per length bin')
    parser.add_argument('--output_dir', type=str, default='./complexity_results',
                        help='Output directory for results')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model_name = 'GALS'

    print("\n" + "="*60)
    print("RNA Inverse Folding Complexity Evaluation")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Samples per RNA: {args.n_samples}")
    print(f"Timing repeats: {args.n_repeats}, warmup: {args.warmup}")
    print(f"Temperature: {args.temperature}")
    print(f"RNAs per bin: {args.n_rnas_per_bin}")
    print("="*60 + "\n")

    # Load dataset
    print("Loading dataset...")
    data_dict = torch.load(os.path.join(DATA_PATH, 'processed.pt'), weights_only=False)
    data_list = list(data_dict.values())

    # Load split (returns tuple: train_idx, val_idx, test_idx)
    _, _, test_idx = torch.load(os.path.join(DATA_PATH, f'{args.split}_split.pt'), weights_only=False)

    # Preprocess: extract backbone coordinates
    print("Preprocessing: extracting backbone coordinates...")
    test_data = []
    for i in test_idx:
        rna_data = preprocess_rna_data(data_list[i])
        if rna_data is not None:
            # Add index as ID if not present
            if 'id' not in rna_data:
                rna_data['id'] = f"RNA_{i}"
            test_data.append(rna_data)

    print(f"✓ Loaded {len(test_data)} test samples (after preprocessing)\n")

    # Build bins from data length range
    max_len = max(len(d['sequence']) for d in test_data)
    bin_specs = build_length_bins(max_len)

    # Organize test data by length bins
    bins = {label: [] for _, _, label in bin_specs}
    for data in test_data:
        length = len(data['sequence'])
        bin_name = get_length_bin(length, bin_specs)
        if bin_name is not None:
            bins[bin_name].append((data, length))

    print("Length distribution:")
    for _, _, label in bin_specs:
        print(f"  {label} nt: {len(bins[label])} samples")
    print()

    # Select representative RNAs from each bin
    selected_rnas = {}
    for start, end, label in bin_specs:
        samples = bins[label]
        if len(samples) == 0:
            continue
        # Sort by length and select evenly spaced samples
        samples_sorted = sorted(samples, key=lambda x: x[1])
        n_available = len(samples_sorted)
        n_select = min(args.n_rnas_per_bin, n_available)

        if n_select == 1:
            indices = [n_available // 2]
        else:
            indices = np.linspace(0, n_available - 1, n_select, dtype=int)

        selected_rnas[label] = [samples_sorted[i] for i in indices]
        print(f"Selected {n_select} RNAs from {label} nt bin:")
        for data, length in selected_rnas[label]:
            print(f"  - Length: {length} nt (ID: {data['id']})")
    print()

    # Results storage
    results = []

    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}\n")

    model, config = load_model(model_name, args.split, device)
    if model is None:
        print(f"Skipping {model_name} due to loading error\n")
        return

    featurizer = RNAGraphFeaturizer(
        split='test',
        top_k=config.top_k,
        num_rbf=config.num_rbf,
        num_posenc=config.num_posenc,
        max_num_conformers=config.max_num_conformers,
        noise_scale=0.0
    )

    for bin_name, rna_list in selected_rnas.items():
        print(f"\nProcessing {bin_name} nt bin...")

        for data, length in tqdm(rna_list, desc=f"{model_name} - {bin_name}"):
            data_featurized = featurizer(data)
            data_featurized = data_featurized.to(device)

            avg_time, total_time, std_time = measure_inference_time(
                model, data_featurized, args.n_samples, args.temperature, device,
                n_repeats=args.n_repeats, warmup=args.warmup
            )

            peak_memory = measure_memory_usage(model, data_featurized, device)

            results.append({
                'model': model_name,
                'rna_id': data['id'],
                'length': length,
                'length_bin': bin_name,
                'n_samples': args.n_samples,
                'avg_time_per_sample': avg_time,
                'std_time_per_sample': std_time,
                'total_time': total_time,
                'total_time_all_samples': total_time,
                'n_repeats': args.n_repeats,
                'warmup': args.warmup,
                'peak_memory_mb': peak_memory,
                'throughput_samples_per_sec': 1.0 / avg_time if avg_time > 0 else 0,
            })

            print(f"  {data['id']} (L={length}): {avg_time:.4f}±{std_time:.4f}s/sample, "
                  f"{peak_memory:.1f} MB")
            print(f"    Total time for {args.n_samples} samples: {total_time:.4f}s")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    output_path = os.path.join(args.output_dir, f'complexity_{args.split}.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    model_df = df[df['model'] == model_name]
    if len(model_df) > 0:
        print(f"\n{model_name}:")
        for _, _, label in bin_specs:
            bin_df = model_df[model_df['length_bin'] == label]
            if len(bin_df) == 0:
                continue

            avg_time = bin_df['avg_time_per_sample'].mean()
            std_time = bin_df['avg_time_per_sample'].std()
            avg_memory = bin_df['peak_memory_mb'].mean()
            avg_throughput = bin_df['throughput_samples_per_sec'].mean()

            print(f"  {label} nt:")
            print(f"    Time: {avg_time:.4f}±{std_time:.4f} s/sample")
            print(f"    Memory: {avg_memory:.1f} MB")
            print(f"    Throughput: {avg_throughput:.2f} samples/s")

    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()

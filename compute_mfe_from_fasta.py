#!/usr/bin/env python3
"""
MFE calculation from existing FASTA files using ViennaRNA (RNAfold).
Reads from testfasta and runs in parallel across k-folds.
"""

import os
import glob
import subprocess
import argparse
import re
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time


def compute_mfe_rnafold(sequence: str) -> float:
    """Compute MFE using ViennaRNA RNAfold."""
    try:
        result = subprocess.run(
            ["RNAfold", "--noPS"],
            input=sequence,
            capture_output=True,
            text=True,
            timeout=60
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            # Format: "...((...))... (-12.34)"
            mfe_part = lines[1].split('(')[-1].rstrip(')')
            return float(mfe_part)
    except Exception as e:
        pass
    return None


def parse_fasta_file(fasta_path: str, n_samples: int = 5) -> tuple:
    """
    Parse fasta file and extract native sequence and multiple predicted sequences.

    Args:
        fasta_path: Path to fasta file
        n_samples: Number of predicted samples to extract (default: 5)

    Returns:
        (native_seq, list_of_pred_seqs)
    """
    with open(fasta_path, 'r') as f:
        lines = f.readlines()

    native_seq = ""
    pred_seqs = []
    current_seq = ""
    is_native = False
    current_sample_idx = -1

    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            # Save previous sequence
            if is_native and current_seq:
                native_seq = current_seq
            elif current_sample_idx >= 0 and current_seq:
                pred_seqs.append(current_seq)
                if len(pred_seqs) >= n_samples:
                    break  # Got enough samples

            current_seq = ""
            is_native = False

            if 'input_sequence' in line:
                is_native = True
                current_sample_idx = -1
            elif 'sample=' in line:
                # Extract sample index
                import re
                match = re.search(r'sample=(\d+)', line)
                if match:
                    current_sample_idx = int(match.group(1))
                    if current_sample_idx >= n_samples:
                        break  # Don't need more samples
                else:
                    current_sample_idx = -1
            else:
                current_sample_idx = -1
        else:
            if is_native or current_sample_idx >= 0:
                current_seq += line

    # Handle last sequence
    if is_native and current_seq:
        native_seq = current_seq
    elif current_sample_idx >= 0 and current_seq and len(pred_seqs) < n_samples:
        pred_seqs.append(current_seq)

    return native_seq, pred_seqs


def process_single_fasta(args: tuple) -> dict:
    """
    Process a single fasta file and return MFE data.
    Computes MFE for native sequence and median MFE of 5 predicted samples.
    """
    fasta_file, kfold = args
    try:
        native_seq, pred_seqs = parse_fasta_file(fasta_file, n_samples=5)
        if not native_seq or len(pred_seqs) == 0:
            return None

        # Compute native MFE
        native_mfe = compute_mfe_rnafold(native_seq)
        if native_mfe is None:
            return None

        # Compute MFE for each predicted sample
        pred_mfes = []
        for pred_seq in pred_seqs:
            mfe = compute_mfe_rnafold(pred_seq)
            if mfe is not None:
                pred_mfes.append(mfe)

        if len(pred_mfes) == 0:
            return None

        # Take median of predicted MFEs
        pred_mfe = float(np.median(pred_mfes))
        length = len(native_seq)

        return {
            'length': length,
            'native_mfe': native_mfe,
            'pred_mfe': pred_mfe,
            'native_mfe_per_nt': native_mfe / length,
            'pred_mfe_per_nt': pred_mfe / length,
            'kfold': kfold,
            'n_samples': len(pred_mfes),  # Record how many samples were used
        }
    except Exception:
        pass
    return None


def main(model_name: str, kfolds: list = [1, 2, 3, 4, 5], n_workers: int = 20):
    fasta_base_dir = "./testfasta"
    output_dir = "./mfe_data"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MFE Calculation using ViennaRNA (RNAfold) - Parallel Mode")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # Collect all fasta files from all folds
    all_tasks = []  # List of (fasta_file, kfold) tuples
    for kfold in kfolds:
        fasta_dir = os.path.join(fasta_base_dir, f"{model_name}_kfold_{kfold}")
        if not os.path.exists(fasta_dir):
            print(f"Warning: {fasta_dir} not found, skipping...")
            continue

        fasta_files = glob.glob(os.path.join(fasta_dir, "*.fasta"))
        print(f"  kfold_{kfold}: {len(fasta_files)} files")
        for f in fasta_files:
            all_tasks.append((f, kfold))

    print(f"\nTotal files to process: {len(all_tasks)}")
    print(f"Using {n_workers} parallel workers...")

    start_time = time.time()

    # Process all files in parallel using a single pool
    with Pool(processes=n_workers) as pool:
        all_results = list(tqdm(
            pool.imap(process_single_fasta, all_tasks),
            total=len(all_tasks),
            desc="Computing MFE"
        ))

    # Filter out None results
    results = [r for r in all_results if r is not None]

    elapsed = time.time() - start_time

    # Save results
    output_file = os.path.join(output_dir, f"mfe_{model_name}_all_kfolds.txt")
    with open(output_file, 'w') as f:
        f.write("length\tnative_mfe\tpred_mfe\tnative_mfe_per_nt\tpred_mfe_per_nt\tkfold\n")
        for r in results:
            f.write(f"{r['length']}\t{r['native_mfe']:.4f}\t{r['pred_mfe']:.4f}\t"
                    f"{r['native_mfe_per_nt']:.6f}\t{r['pred_mfe_per_nt']:.6f}\t{r['kfold']}\n")

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total samples: {len(results)}")
    print(f"Time elapsed: {elapsed:.2f}s ({len(results)/elapsed:.1f} samples/sec)")

    if results:
        native_mfe_avg = sum(r['native_mfe_per_nt'] for r in results) / len(results)
        pred_mfe_avg = sum(r['pred_mfe_per_nt'] for r in results) / len(results)
        print(f"Mean Native MFE/nt: {native_mfe_avg:.4f} kcal/mol/nt")
        print(f"Mean Pred MFE/nt:   {pred_mfe_avg:.4f} kcal/mol/nt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute MFE from FASTA files')
    parser.add_argument('--model', type=str, default='GALS', help='Model name')
    parser.add_argument('--kfolds', type=str, default='1,2,3,4,5', help='K-folds to process')
    parser.add_argument('--workers', type=int, default=5, help='Number of workers per fold')
    args = parser.parse_args()

    kfolds = [int(k) for k in args.kfolds.split(',')]
    main(args.model, kfolds, n_workers=args.workers)

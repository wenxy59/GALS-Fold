"""
3D structure evaluation for GALS-Fold.

Computes:
  - F1 score (secondary structure)
  - TM score (tertiary structure)
"""

import os
import yaml
import argparse
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(".env")

# Import model class
from src.models import GeometricLongShortRNA
from src.constants import NUM_TO_LETTER, DATA_PATH, PROJECT_PATH, RNA_ATOMS, FILL_VALUE
from src.data.sec_struct_utils import predict_sec_struct
from src.data.data_utils import get_backbone_coords

# RhoFold for 3D structure prediction
from tools.rhofold.rf import RhoFold
from tools.rhofold.config import rhofold_config


def load_config(config_path, overrides=None):
    """Load YAML config and apply overrides."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = {}
    for key, val in config_dict.items():
        if isinstance(val, dict) and 'value' in val:
            config[key] = val['value']
        else:
            config[key] = val

    if overrides:
        for key, val in overrides.items():
            config[key] = val

    # Auto-generate paths
    model = config.get('model', 'GALS')
    split = config.get('split', 'kfold_1')

    if not config.get('model_path'):
        config['model_path'] = f'./checkpoint/{model}_{split}_best_checkpoint.h5'

    return SimpleNamespace(**config)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(config):
    """Initialize model."""
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

    model_kwargs['heads'] = getattr(config, 'heads', 4)
    model_kwargs['num_anchors'] = getattr(config, 'num_anchors', 32)
    model_kwargs['local_window'] = getattr(config, 'local_window', 10)
    model_kwargs['length_threshold'] = getattr(config, 'length_threshold', 150)
    model_kwargs['aux_loss_weight'] = getattr(config, 'aux_loss_weight', 0.3)

    return model_class(**model_kwargs)


def preprocess_rna_data(rna_data):
    """Extract backbone coordinates and keep full-atom coords for GT PDB."""
    pyrimidine_bb_indices = [RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("N1")]
    purine_bb_indices = [RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("N9")]

    # Store original full-atom coordinates for GT PDB generation
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


def load_sample_indices_from_file(index_file):
    """Load sample indices (RNA IDs) from a file."""
    if not os.path.exists(index_file):
        return None

    rna_ids = []
    with open(index_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Handle TSV format (rna_id<TAB>length)
            if '\t' in line:
                rna_id = line.split('\t')[0]
            else:
                rna_id = line
            rna_ids.append(rna_id)

    return rna_ids


def filter_by_sample_indices(test_data, sample_indices):
    """Filter test data by sample indices."""
    # Create a mapping from ID to data
    id_to_data = {}
    for rna_data in test_data:
        if rna_data.get('id_list'):
            rna_id = rna_data['id_list'][0]
            id_to_data[rna_id] = rna_data

    # Filter and maintain order from sample_indices
    filtered_data = []
    missing_ids = []
    for rna_id in sample_indices:
        if rna_id in id_to_data:
            filtered_data.append(id_to_data[rna_id])
        else:
            missing_ids.append(rna_id)

    if missing_ids:
        print(f"  Warning: {len(missing_ids)} IDs from index file not found in test data")
        if len(missing_ids) <= 5:
            print(f"    Missing: {missing_ids}")

    return filtered_data


def load_test_data(split, sample_ratio=1.0, index_file=None):
    """Load and preprocess test data."""
    print(f"Loading data from {DATA_PATH}/processed.pt...")
    data_dict = torch.load(os.path.join(DATA_PATH, "processed.pt"), weights_only=False)
    data_list = list(data_dict.values())

    print(f"Loading split from {DATA_PATH}/{split}_split.pt...")
    _, _, test_idx = torch.load(
        os.path.join(DATA_PATH, f"{split}_split.pt"), weights_only=False
    )

    # Preprocess: extract backbone coordinates (but keep full coords for GT PDB)
    print("Preprocessing: extracting backbone coordinates...")
    test_data = []
    for i in test_idx:
        rna_data = preprocess_rna_data(data_list[i])
        if rna_data is not None:
            test_data.append(rna_data)

    print(f"Loaded {len(test_data)} test samples (after preprocessing)")

    # If index_file is provided, use it to filter samples
    if index_file is not None:
        sample_indices = load_sample_indices_from_file(index_file)
        if sample_indices is not None:
            print(f"\n{'='*60}")
            print(f"Loading sample indices from: {index_file}")
            print(f"{'='*60}")
            print(f"  Index file contains: {len(sample_indices)} RNA IDs")
            test_data = filter_by_sample_indices(test_data, sample_indices)
            print(f"  Filtered to: {len(test_data)} samples")
            print(f"{'='*60}\n")
            return test_data
        else:
            print(f"  Warning: Index file {index_file} not found, falling back to sampling")

    # Perform stratified sampling if sample_ratio < 1.0
    if sample_ratio < 1.0:
        test_data = stratified_sampling_by_length(test_data, split, sample_ratio)

    return test_data


def stratified_sampling_by_length(test_data, split, sample_ratio=0.2):
    """Perform stratified sampling by length ranges."""
    import random

    # Extract fold number from split name for consistent seeding
    # e.g., 'kfold_1' -> seed based on '1'
    fold_seed = hash(split) % 10000  # Use hash for consistent seed
    random.seed(fold_seed)
    np.random.seed(fold_seed)

    print(f"\n{'='*60}")
    print(f"Stratified Sampling by Length")
    print(f"Split: {split}, Seed: {fold_seed}")
    print(f"{'='*60}")

    # Categorize by length
    length_ranges = {
        '0-100': [],
        '100-200': [],
        '200-1000': [],
        '>1000': []
    }

    for rna_data in test_data:
        seq_len = len(rna_data['sequence'])
        if seq_len <= 100:
            length_ranges['0-100'].append(rna_data)
        elif seq_len <= 200:
            length_ranges['100-200'].append(rna_data)
        elif seq_len <= 1000:
            length_ranges['200-1000'].append(rna_data)
        else:
            length_ranges['>1000'].append(rna_data)

    # Define sampling ratios for each range
    sampling_ratios = {
        '0-100': sample_ratio,      # 20% by default
        '100-200': sample_ratio,    # 20% by default
        '200-1000': 0.5,            # 50% fixed (balance computation)
        '>1000': 0.0                # Skip entirely (RhoFold limitation)
    }

    # Sample from each range
    sampled_data = []
    total_original = len(test_data)

    for range_name, data_list in length_ranges.items():
        n_total = len(data_list)
        ratio = sampling_ratios[range_name]

        if ratio == 0.0:
            # Skip this range entirely
            print(f"  {range_name:>10} nt: SKIPPED ({n_total:4d} samples, RhoFold limitation)")
            continue

        n_sample = max(1, int(n_total * ratio)) if n_total > 0 else 0

        if n_total > 0:
            sampled = random.sample(data_list, min(n_sample, n_total))
            sampled_data.extend(sampled)
            print(f"  {range_name:>10} nt: {len(sampled):4d} / {n_total:4d} samples ({ratio*100:5.1f}%)")
        else:
            print(f"  {range_name:>10} nt: {0:4d} / {0:4d} samples ({ratio*100:5.1f}%)")

    print(f"{'='*60}")
    print(f"Total sampled: {len(sampled_data)} / {total_original} ({len(sampled_data)/total_original*100:.1f}%)")
    print(f"Skipped (>1000nt): {len(length_ranges['>1000'])} samples")
    print(f"{'='*60}\n")

    return sampled_data


def dotbracket_to_pairs(dotbracket):
    """Convert dot-bracket notation to base pairs."""
    pairs = set()

    # Define bracket pairs: opening -> closing
    bracket_pairs = {
        '(': ')',
        '[': ']',
        '{': '}',
        '<': '>'
    }

    # Create stacks for each bracket type
    stacks = {opening: [] for opening in bracket_pairs.keys()}

    for i, char in enumerate(dotbracket):
        if char in bracket_pairs:
            # Opening bracket
            stacks[char].append(i)
        elif char in bracket_pairs.values():
            # Closing bracket - find corresponding opening bracket
            for opening, closing in bracket_pairs.items():
                if char == closing and stacks[opening]:
                    j = stacks[opening].pop()
                    # Always store as (smaller_idx, larger_idx)
                    pairs.add((j, i))
                    break

    return pairs


def predict_secondary_structure(sequence):
    """Predict secondary structure."""
    try:
        # predict_sec_struct automatically selects EternaFold or LinearFold
        # based on sequence length (threshold = 1000 nt)
        result = predict_sec_struct(sequence, n_samples=1)
        return result[0]
    except Exception as e:
        print(f"Secondary structure prediction error: {e}")
        return '.' * len(sequence)


def extract_gt_pairs_from_annotation(sec_struct):
    """Extract ground truth base pairs from dataset annotations."""
    return dotbracket_to_pairs(sec_struct)


def compute_f1_score(gt_pairs, pred_pairs):
    """Compute F1 score for base pair prediction."""
    tp = len(gt_pairs & pred_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


def coords_to_pdb(coords, sequence, output_path):
    """Convert coordinates to PDB file."""
    # Create PDB structure
    structure = Structure.Structure("RNA")
    model = Model.Model(0)
    chain = Chain.Chain("A")

    # Determine if this is backbone-only or full-atom coordinates
    num_atoms = coords.shape[1]

    if num_atoms == 3:
        # Backbone only: P, C4', N1/N9
        for i, (res_coords, res_name) in enumerate(zip(coords, sequence)):
            res_id = (' ', i + 1, ' ')
            residue = Residue.Residue(res_id, res_name, ' ')

            atom_names = ["P", "C4'", "N1" if res_name in ['C', 'U'] else "N9"]

            for atom_name, atom_coord in zip(atom_names, res_coords):
                if torch.allclose(atom_coord, torch.tensor(FILL_VALUE)):
                    continue

                atom = Atom.Atom(
                    name=atom_name,
                    coord=atom_coord.cpu().numpy(),
                    bfactor=0.0,
                    occupancy=1.0,
                    altloc=' ',
                    fullname=atom_name,
                    serial_number=len(residue) + 1,
                    element=atom_name[0]
                )
                residue.add(atom)

            chain.add(residue)

    elif num_atoms == 27:
        # Full atom coordinates
        for i, (res_coords, res_name) in enumerate(zip(coords, sequence)):
            res_id = (' ', i + 1, ' ')
            residue = Residue.Residue(res_id, res_name, ' ')

            # Add all 27 atoms
            for atom_idx, atom_coord in enumerate(res_coords):
                # Skip missing atoms (FILL_VALUE)
                if torch.allclose(atom_coord, torch.tensor(FILL_VALUE)):
                    continue

                # Get atom name from RNA_ATOMS list
                atom_name = RNA_ATOMS[atom_idx]

                atom = Atom.Atom(
                    name=atom_name,
                    coord=atom_coord.cpu().numpy(),
                    bfactor=0.0,
                    occupancy=1.0,
                    altloc=' ',
                    fullname=atom_name,
                    serial_number=len(residue) + 1,
                    element=atom_name[0]
                )
                residue.add(atom)

            chain.add(residue)

    else:
        raise ValueError(f"Unexpected number of atoms per residue: {num_atoms}. Expected 3 or 27.")

    model.add(chain)
    structure.add(model)

    # Save to PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))


def run_usalign(pred_pdb, target_pdb):
    """Run US-align to compute TM score."""
    try:
        # Get US-align path from environment or use default (check tools/USalign first)
        default_usalign = os.path.join(PROJECT_PATH, "tools/USalign/USalign")
        usalign_path = os.environ.get("USALIGN_PATH", default_usalign)

        cmd = f"{usalign_path} {pred_pdb} {target_pdb}"
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode()

        # Parse TM-score from output
        # US-align output format: "TM-score= 0.xxxxx (normalized by length of Structure_1: L=xx, d0=x.xx)"
        for line in output.splitlines():
            if "TM-score=" in line and "Structure_1" in line:
                # Extract TM-score value (normalized by Structure_1, which is the predicted structure)
                parts = line.split("TM-score=")
                if len(parts) > 1:
                    tm_str = parts[1].split()[0]
                    return float(tm_str)

        return 0.0
    except Exception as e:
        print(f"US-align error: {e}")
        return 0.0


def featurize_rna(rna_data, featurizer):
    """Featurize RNA data for model input."""
    return featurizer(rna_data)



@torch.no_grad()
def evaluate_single_rna(model, rna_data, featurizer, device, rhofold, n_samples=5, max_length_for_tm=1000):
    """Evaluate a single RNA: generate samples and compute F1/TM."""
    # Featurize
    featurized_data = featurizer(rna_data).to(device)

    # Generate samples
    samples, _ = model.sample(featurized_data, n_samples, temperature=0.1, return_logits=True)

    # Get ground truth secondary structure
    gt_sec_struct = rna_data['sec_struct_list'][0]  # Use first structure from dataset annotation
    gt_pairs = extract_gt_pairs_from_annotation(gt_sec_struct)

    # Get ground truth 3D coordinates
    # Use full-atom coordinates (27 atoms) for GT PDB if available, otherwise use backbone
    if 'coords_list_full' in rna_data and len(rna_data['coords_list_full']) > 0:
        gt_coords = rna_data['coords_list_full'][0]  # Shape: (L, 27, 3) - full atoms
    else:
        gt_coords = rna_data['coords_list'][0]  # Shape: (L, 3, 3) - backbone only

    f1_scores = []
    tm_scores = []
    sequences = []

    # Check if sequence is too long for RhoFold TM-score calculation
    seq_length = len(rna_data['sequence'])
    skip_tm = seq_length > max_length_for_tm
    if skip_tm:
        print(f"    [INFO] Skipping TM-score for long sequence ({seq_length} > {max_length_for_tm} nt)")

    # Create temporary directory for PDB files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save ground truth PDB (using full atoms for US-align compatibility)
        # Only needed if we're computing TM scores
        gt_pdb_path = None
        if not skip_tm:
            gt_pdb_path = Path(tmpdir) / "gt.pdb"
            coords_to_pdb(gt_coords, rna_data['sequence'], gt_pdb_path)

        # Evaluate each sample
        for sample_idx, sample in enumerate(samples.cpu().numpy()):
            # Convert to sequence
            seq_str = "".join([NUM_TO_LETTER[num] for num in sample])
            sequences.append(seq_str)

            # ============ F1 Score (Secondary Structure) ============
            # Use EternaFold (<=1000nt) or LinearFold (>1000nt) for prediction
            pred_dotbracket = predict_secondary_structure(seq_str)
            pred_pairs = dotbracket_to_pairs(pred_dotbracket)
            f1 = compute_f1_score(gt_pairs, pred_pairs)
            f1_scores.append(f1)

            # ============ TM Score (Tertiary Structure) ============
            # Skip TM-score for long sequences to avoid CUDA OOM errors
            if skip_tm:
                tm_scores.append(None)  # Use None to indicate skipped
                continue

            # Use RhoFold to predict 3D structure from designed sequence
            try:
                # Save designed sequence to fasta file
                design_fasta_path = Path(tmpdir) / f"design_{sample_idx}.fasta"
                SeqIO.write(
                    SeqRecord(Seq(seq_str), id=f"sample_{sample_idx}", description=""),
                    str(design_fasta_path), "fasta"
                )

                # Predict 3D structure using RhoFold
                pred_pdb_path = Path(tmpdir) / f"pred_{sample_idx}.pdb"
                rhofold.predict(str(design_fasta_path), str(pred_pdb_path), use_relax=False)

                # Compute TM score using US-align
                tm = run_usalign(str(pred_pdb_path), str(gt_pdb_path))
                tm_scores.append(tm)

                # Clean up temporary fasta file
                if design_fasta_path.exists():
                    design_fasta_path.unlink()

            except Exception as e:
                print(f"    RhoFold prediction failed for sample {sample_idx}: {e}")
                tm_scores.append(0.0)  # Default to 0 on failure

    return f1_scores, tm_scores, sequences


def main(config, device, output_file, sample_ratio=1.0, index_file=None):
    """Main evaluation entry."""
    print(f"\n{'='*60}")
    print(f"3D Structure Evaluation for GALS-Fold")
    print(f"{'='*60}")
    print(f"Model: {config.model}")
    print(f"Split: {config.split}")
    print(f"Checkpoint: {config.model_path}")
    print(f"Output file: {output_file}")
    print(f"Device: {device}")
    if index_file:
        print(f"Index file: {index_file}")
    elif sample_ratio < 1.0:
        print(f"Sample ratio: {sample_ratio:.1%} (stratified by length)")
    print(f"{'='*60}\n")

    # Set seed
    set_seed(config.seed)

    # Load test data
    test_data = load_test_data(config.split, sample_ratio=sample_ratio, index_file=index_file)

    # Initialize featurizer
    print("Initializing featurizer...")
    from src.data.featurizer import RNAGraphFeaturizer
    featurizer = RNAGraphFeaturizer(
        split="test",
        top_k=config.top_k,
        num_rbf=config.num_rbf,
        num_posenc=config.num_posenc,
        max_num_conformers=config.max_num_conformers,
        noise_scale=0.0
    )

    # Initialize inverse folding model
    print(f"Initializing {config.model} model...")
    model = get_model(config).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {config.model_path}")
    model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=False))
    model.eval()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Initialize RhoFold for 3D structure prediction
    print("Initializing RhoFold for 3D structure prediction...")
    rhofold = RhoFold(rhofold_config, device)
    rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
    print(f"Loading RhoFold checkpoint: {rhofold_path}")
    rhofold.load_state_dict(torch.load(rhofold_path, map_location=torch.device('cpu'))['model'])
    rhofold = rhofold.to(device)
    rhofold.eval()
    print("RhoFold initialized successfully.\n")

    # Open output file
    results = []

    print(f"Processing {len(test_data)} test samples...\n")

    for i, rna_data in enumerate(test_data, 1):
        rna_id = rna_data['id_list'][0] if rna_data.get('id_list') else f"seq_{i}"
        seq_length = len(rna_data['sequence'])

        print(f"[{i}/{len(test_data)}] Processing {rna_id} (length={seq_length})...")

        try:
            # Generate 5 samples and compute metrics
            f1_scores, tm_scores, _ = evaluate_single_rna(
                model, rna_data, featurizer, device, rhofold, n_samples=5
            )

            # Compute median scores
            f1_median = np.median(f1_scores)

            # Handle TM scores (may contain None for skipped long sequences)
            valid_tm_scores = [tm for tm in tm_scores if tm is not None]
            if len(valid_tm_scores) > 0:
                tm_median = np.median(valid_tm_scores)
            else:
                tm_median = None  # All TM scores were skipped

            # Store result
            results.append({
                'length': seq_length,
                'f1': f1_median,
                'tm': tm_median
            })

            if tm_median is not None:
                print(f"  F1 median: {f1_median:.4f}, TM median: {tm_median:.4f}")
            else:
                print(f"  F1 median: {f1_median:.4f}, TM median: skipped (long sequence)")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Write results to file
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("# RNA 3D Structure Evaluation Results\n")
        f.write("# Column 1: Sequence length\n")
        f.write("# Column 2: F1 score (median of 5 samples)\n")
        f.write("# Column 3: TM score (median of 5 samples, 'NA' for skipped long sequences)\n")
        f.write("length\tf1_score\ttm_score\n")

        for result in results:
            tm_str = f"{result['tm']:.4f}" if result['tm'] is not None else "NA"
            f.write(f"{result['length']}\t{result['f1']:.4f}\t{tm_str}\n")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(test_data)}")
    print(f"Successfully evaluated: {len(results)}")

    if results:
        f1_values = [r['f1'] for r in results]
        tm_values = [r['tm'] for r in results if r['tm'] is not None]
        print(f"\nOverall Metrics:")
        print(f"  F1 score: {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}")
        if tm_values:
            print(f"  TM score: {np.mean(tm_values):.4f} ± {np.std(tm_values):.4f} (n={len(tm_values)})")
        else:
            print(f"  TM score: No valid TM scores (all sequences too long)")

    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='3D Structure Evaluation for GALS-Fold',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', default='configs/evaluate.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--output', default=None,
                        help='Output file for results (default: {model}_{split}_3d_metrics.txt)')
    parser.add_argument('--output_dir', default='./results_3d',
                        help='Output directory for results (default: ./results_3d)')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Ratio of samples to use (0-1). If < 1.0, performs stratified sampling '
                             'by length ranges (0-100nt, 100-200nt, 200nt+). Default: 1.0 (use all samples)')
    parser.add_argument('--index_file', type=str, default=None,
                        help='Path to a file containing RNA IDs to evaluate (one per line). '
                             'If provided, overrides --sample_ratio. Use this to ensure consistent '
                             'samples across different models.')
    args, unknown = parser.parse_known_args()

    # Parse key=value overrides
    overrides = {}
    for arg in unknown:
        if '=' in arg:
            k, v = arg.split('=', 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            overrides[k] = v

    # Load config
    config = load_config(args.config, overrides)

    # Auto-generate output filename based on model and split
    if args.output is None:
        os.makedirs(args.output_dir, exist_ok=True)
        # Add sample_ratio to filename if sampling is used
        if args.index_file:
            output_file = os.path.join(args.output_dir, f"{config.model}_{config.split}_3d_metrics_indexed.txt")
        elif args.sample_ratio < 1.0:
            output_file = os.path.join(args.output_dir, f"{config.model}_{config.split}_3d_metrics_sample{int(args.sample_ratio*100)}.txt")
        else:
            output_file = os.path.join(args.output_dir, f"{config.model}_{config.split}_3d_metrics.txt")
    else:
        output_file = args.output

    # Set device
    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")

    # Run evaluation
    main(config, device, output_file, sample_ratio=args.sample_ratio, index_file=args.index_file)

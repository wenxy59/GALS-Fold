"""
Unified evaluation script for GALS-Fold models.

Usage:
    python evaluate.py --config configs/evaluate.yaml

    # Test GALS model on kfold_1 (paths auto-generated):
    python evaluate.py --config configs/evaluate.yaml model=GALS split=kfold_1

    # Test GVPAtten model on kfold_2:
    python evaluate.py --config configs/evaluate.yaml model=GVPAtten split=kfold_2

    # Override auto-generated paths if needed:
    python evaluate.py --config configs/evaluate.yaml model=GALS split=kfold_1 \
        model_path=./custom/path/to/model.h5
"""

import dotenv
dotenv.load_dotenv(".env")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import yaml
import random
import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import torch_geometric
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.data.featurizer import RNAGraphFeaturizer
from src.data.data_utils import get_backbone_coords
from src.evaluator import edit_distance, self_consistency_score_eternafold
from src.models import GeometricLongShortRNA, GVPAttentionShortBranch
from src.constants import NUM_TO_LETTER, RNA_ATOMS, FILL_VALUE


def load_config(config_path, overrides=None):
    """Load YAML config and apply command line overrides."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Extract values from nested dict
    config = {}
    for key, val in config_dict.items():
        if isinstance(val, dict) and 'value' in val:
            config[key] = val['value']
        else:
            config[key] = val

    # Apply overrides
    if overrides:
        for key, val in overrides.items():
            config[key] = val

    # Auto-generate paths based on model and split if not specified
    model = config.get('model', 'GALS')
    split = config.get('split', 'kfold_1')

    # Auto-generate output_dir: ./statistics/testfasta/{model}_{split}
    if not config.get('output_dir'):
        config['output_dir'] = f'./statistics/testfasta/{model}_{split}'

    # Auto-generate model_path: ./trainedmodels/{model}_{split}_best_checkpoint.h5
    if not config.get('model_path'):
        config['model_path'] = f'./trainedmodels/{model}_{split}_best_checkpoint.h5'

    # Auto-generate test_index_file: ./statistics/{split}_test_index.txt
    if not config.get('test_index_file'):
        config['test_index_file'] = f'./statistics/{split}_test_index.txt'

    return SimpleNamespace(**config)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(config):
    """Initialize model based on config."""
    model_class = {
        'GALS': GeometricLongShortRNA,
        'GVPAtten': GVPAttentionShortBranch,
    }[config.model]
    
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


def read_test_index(index_file):
    """Read test index file and return PDB basenames (without .pdb)."""
    with open(index_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


@torch.no_grad()
def design_from_pdb(model, featurizer, pdb_path, config, device):
    """Design RNA sequences for a single PDB file."""
    # Featurize
    featurized_data, raw_data = featurizer.featurize_from_pdb_file(str(pdb_path))
    
    # Process coordinates
    if raw_data['coords_list'][0].shape[1] == len(RNA_ATOMS):
        coords_list = []
        for coords in raw_data['coords_list']:
            coords = get_backbone_coords(coords, raw_data['sequence'])
            if not torch.all((coords == FILL_VALUE).sum(axis=(1, 2)) > 0):
                coords_list.append(coords)
        if coords_list:
            raw_data['coords_list'] = coords_list
    
    # Transfer to device
    featurized_data = featurized_data.to(device)
    
    # Sample sequences
    n_samples = config.n_samples
    temperature = config.temperature
    samples, logits = model.sample(featurized_data, n_samples, temperature, None, return_logits=True)
    
    # Compute metrics
    n_nodes = logits.shape[1]
    perplexity = torch.exp(F.cross_entropy(
        logits.view(n_samples * n_nodes, model.out_dim),
        samples.view(n_samples * n_nodes).long(),
        reduction="none"
    ).view(n_samples, n_nodes).mean(dim=1)).cpu().numpy()
    
    recovery = samples.eq(featurized_data.seq).float().mean(dim=1).cpu().numpy()
    
    sc_score = self_consistency_score_eternafold(
        samples.cpu().numpy(),
        raw_data['sec_struct_list'],
        featurized_data.mask_coords.cpu().numpy()
    )
    
    # Format output
    sequences = [
        SeqRecord(
            Seq(raw_data["sequence"]),
            id="input_sequence,",
            description=f"model={config.model}, checkpoint={config.model_path}, seed={config.seed}"
        )
    ]
    
    for idx, (seq, perp, rec, sc) in enumerate(zip(
        samples.cpu().numpy(), perplexity, recovery, sc_score
    )):
        seq_str = "".join([NUM_TO_LETTER[num] for num in seq])
        edit_dist = edit_distance(seq_str, raw_data['sequence'])
        sequences.append(SeqRecord(
            Seq(seq_str),
            id=f"sample={idx},",
            description=f"seed={config.seed}, temperature={temperature}, perplexity={perp:.4f}, recovery={rec:.4f}, edit_dist={edit_dist}, sc_score={sc:.4f}"
        ))
    
    return sequences, samples, perplexity, recovery, sc_score


def main(config, device):
    """Main evaluation function."""
    print(f"\n{'='*60}")
    print(f"GALS-Fold Evaluation")
    print(f"{'='*60}")
    print(f"Model: {config.model}")
    print(f"Checkpoint: {config.model_path}")
    print(f"Test index: {config.test_index_file}")
    print(f"Output dir: {config.output_dir}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Set seed
    set_seed(config.seed)

    # Initialize featurizer
    print("Initializing featurizer...")
    featurizer = RNAGraphFeaturizer(
        split="test",
        top_k=config.top_k,
        num_rbf=config.num_rbf,
        num_posenc=config.num_posenc,
        max_num_conformers=config.max_num_conformers,
        noise_scale=0.0  # No noise for evaluation
    )

    # Initialize model
    print(f"Initializing {config.model} model...")
    model = get_model(config).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {config.model_path}")
    model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=False))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # Read test index
    pdb_basenames = read_test_index(config.test_index_file)
    print(f"Found {len(pdb_basenames)} PDB files to process\n")

    # Process each PDB
    success_count = 0
    failed_files = []
    all_recovery = []
    all_perplexity = []
    all_sc_score = []

    for i, pdb_basename in enumerate(pdb_basenames, 1):
        pdb_path = Path(config.pdb_dir) / f"{pdb_basename}.pdb"
        output_path = Path(config.output_dir) / f"{pdb_basename}.fasta"

        print(f"\n[{i}/{len(pdb_basenames)}] " + "="*50)
        print(f"Processing: {pdb_path}")

        if not pdb_path.exists():
            print(f"Warning: Input file does not exist, skipping...")
            failed_files.append(pdb_basename)
            continue

        try:
            sequences, samples, perplexity, recovery, sc_score = design_from_pdb(
                model, featurizer, pdb_path, config, device
            )
            SeqIO.write(sequences, str(output_path), "fasta")

            mean_rec = np.mean(recovery)
            mean_perp = np.mean(perplexity)
            mean_sc = np.mean(sc_score)

            all_recovery.append(mean_rec)
            all_perplexity.append(mean_perp)
            all_sc_score.append(mean_sc)

            print(f"Successfully processed {pdb_basename}")
            print(f"  Recovery: {mean_rec:.4f}, Perplexity: {mean_perp:.4f}, SC Score: {mean_sc:.4f}")
            print(f"Output file: {output_path}")
            success_count += 1

        except Exception as e:
            print(f"Failed to process {pdb_basename}")
            print(f"Error: {e}")
            failed_files.append(pdb_basename)

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total: {len(pdb_basenames)} files")
    print(f"Success: {success_count} files")
    print(f"Failed: {len(failed_files)} files")

    if all_recovery:
        print(f"\nOverall Metrics (mean ± std):")
        print(f"  Recovery:   {np.mean(all_recovery):.4f} ± {np.std(all_recovery):.4f}")
        print(f"  Perplexity: {np.mean(all_perplexity):.4f} ± {np.std(all_perplexity):.4f}")
        print(f"  SC Score:   {np.mean(all_sc_score):.4f} ± {np.std(all_sc_score):.4f}")

    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files[:10]:
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    print(f"\nOutput saved to: {config.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GALS-Fold Evaluation")
    parser.add_argument('--config', default='configs/evaluate.yaml', help='Path to config file')
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

    # Print config
    print("\nConfiguration:")
    for key, val in vars(config).items():
        print(f"  {key}: {val}")

    # Set device
    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")

    # Run evaluation
    main(config, device)


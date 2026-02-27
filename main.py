import dotenv
dotenv.load_dotenv(".env")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import random
import argparse
import ast
import wandb
import numpy as np

import torch
from torch_geometric.loader import DataLoader

from src.trainer import train
from src.data.dataset import RNADesignDataset, BatchSampler
from src.models import GeometricLongShortRNA
from src.constants import DATA_PATH


def main(config, device):
    """
    Main function for training and evaluating GALS-Fold.
    """
    # Set seed
    set_seed(config.seed, device.type)

    # Initialise model
    model = get_model(config).to(device)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'\nMODEL\n    {model}\n    Total parameters: {total_param}')
    wandb.run.summary["total_param"] = total_param

    # Load checkpoint
    if config.model_path != '':
        model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=False))
    
    # Get train, val, test data samples as lists
    train_list, val_list, test_list = get_data_splits(config, split_type=config.split)

    # Load datasets
    trainset = get_dataset(config, train_list, split="train")
    valset = get_dataset(config, val_list, split="val")
    testset = get_dataset(config, test_list, split="test")

    # Prepare dataloaders
    length_bins = getattr(config, 'length_bins', None)
    length_weights = getattr(config, 'length_weights', None)
    if not length_bins or not length_weights:
        length_bins = None
        length_weights = None

    train_loader = get_dataloader(
        config, trainset, shuffle=True,
        length_bins=length_bins, length_weights=length_weights
    )
    val_loader = get_dataloader(config, valset, shuffle=False)
    test_loader = get_dataloader(config, testset, shuffle=False)
        
    # Run trainer
    train(config, model, train_loader, val_loader, test_loader, device)

    # Post-training recovery evaluation by length bins (0-100/100-200/>200)
    post_eval = getattr(config, "post_eval_recovery", False)
    if post_eval:
        try:
            if hasattr(model, "sample"):
                _evaluate_recovery_bins(config, model, testset, device)
            else:
                print("Post-eval skipped: model has no sample() method.")
        except Exception as exc:
            print(f"Post-eval recovery bins failed: {exc}")
    else:
        print("Post-eval recovery bins skipped (post_eval_recovery=False).")


def _evaluate_recovery_bins(config, model, dataset, device):
    model.eval()
    n_samples = int(getattr(config, "n_samples", 16))
    temperature = float(getattr(config, "temperature", 0.1))

    save_dir = getattr(config, "save_dir", "./trainedmodels")
    ckpt_prefix = getattr(config, "checkpoint_prefix", "")
    ckpt_prefix = f"{ckpt_prefix}_" if ckpt_prefix else ""
    best_ckpt = os.path.join(save_dir, f"{ckpt_prefix}best_checkpoint.h5")

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=False))

    bins = {
        "0-100": [],
        "100-200": [],
        ">200": [],
    }

    with torch.no_grad():
        for raw_data in dataset.data_list:
            data = dataset.featurizer(raw_data).to(device)
            samples = model.sample(data, n_samples, temperature, return_logits=False)
            if isinstance(samples, tuple):
                samples = samples[0]
            if samples.dim() == 1:
                samples = samples.unsqueeze(0)

            rec = samples.eq(data.seq.unsqueeze(0)).float().mean().item()
            seq_len = int(data.seq.numel())
            if seq_len <= 100:
                bins["0-100"].append(rec)
            elif seq_len <= 200:
                bins["100-200"].append(rec)
            else:
                bins[">200"].append(rec)

    def summarize(values):
        if not values:
            return float("nan"), float("nan"), 0
        arr = np.array(values)
        return float(arr.mean()), float(arr.std()), len(values)

    print("\nPost-eval Recovery by Length (mean +/- std):")
    for key in ["0-100", "100-200", ">200"]:
        mean, std, count = summarize(bins[key])
        print(f"  {key}: {mean:.4f} +/- {std:.4f} (n={count})")


def get_data_splits(config, split_type="kfold_1"):
    """
    Returns train, val, test data splits as lists.
    """
    data_list = list(torch.load(os.path.join(DATA_PATH, "processed.pt"), weights_only=False).values())
    
    def index_list_by_indices(lst, indices):
        # return [lst[index] if 0 <= index < len(lst) else None for index in indices]
        return [lst[index] for index in indices]
    
    # Pre-compute using notebooks/split_{split_type}.ipynb
    train_idx_list, val_idx_list, test_idx_list = torch.load(
        os.path.join(DATA_PATH, f"{split_type}_split.pt"), weights_only=False) 
    train_list = index_list_by_indices(data_list, train_idx_list)
    val_list = index_list_by_indices(data_list, val_idx_list)
    test_list = index_list_by_indices(data_list, test_idx_list)

    return train_list, val_list, test_list


def get_dataset(config, data_list, split="train"):
    """
    Returns a Dataset for a given split.
    """
    return RNADesignDataset(
        data_list = data_list,
        split = split,
        top_k = config.top_k,
        num_rbf = config.num_rbf,
        num_posenc = config.num_posenc,
        max_num_conformers = config.max_num_conformers,
        noise_scale = config.noise_scale
    )


def get_dataloader(
        config, 
        dataset, 
        shuffle=True,
        pin_memory=True,
        exclude_keys=[],
        length_bins=None,
        length_weights=None,
    ):
    """
    Returns a DataLoader for a given Dataset.

    Args:
        dataset (RNADesignDataset): dataset object
        config (dict): wandb configuration dictionary
        shuffle (bool): whether to shuffle the dataset
        pin_memory (bool): whether to pin memory
        exclue_keys (list): list of keys to exclude during batching
    """
    return DataLoader(
        dataset, 
        num_workers = config.num_workers,
        batch_sampler = BatchSampler(
            node_counts = dataset.node_counts, 
            max_nodes_batch = config.max_nodes_batch,
            max_nodes_sample = config.max_nodes_sample,
            shuffle = shuffle,
            length_bins = length_bins,
            length_weights = length_weights,
        ),
        pin_memory = pin_memory,
        exclude_keys = exclude_keys
    )


def get_model(config):
    """
    Returns a Model for a given config.
    """
    model_class = GeometricLongShortRNA

    # Base parameters
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
    model_kwargs['aux_loss_weight'] = getattr(config, 'aux_loss_weight', 0.01)

    return model_class(**model_kwargs)


def set_seed(seed=0, device_type='cpu'):
    """
    Sets random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if device_type == 'xpu':
        import intel_extension_for_pytorch as ipex
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='GALS: Graph-based Autoregressive Language model for RNA Inverse Folding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --config configs/default.yaml
    python main.py --config configs/default.yaml split=kfold_2 epochs=50
    python main.py --config configs/default.yaml --expt_name my_experiment
        """)
    parser.add_argument('--config', dest='config', default='configs/default.yaml', type=str,
                        help='Path to YAML config file')
    parser.add_argument('--expt_name', dest='expt_name', default=None, type=str,
                        help='Experiment name for logging')
    parser.add_argument('--tags', nargs='+', dest='tags', default=[],
                        help='Tags for wandb logging')
    parser.add_argument('--wandb', action="store_true",
                        help='Enable wandb logging (disabled by default)')
    args, unknown = parser.parse_known_args()

    # Parse key=value args for config override
    config_overrides = {}
    for arg in unknown:
        if '=' in arg:
            k, v = arg.split('=', 1)
            # Parse list/tuple/dict literals passed from sweeps (e.g., "[15, 4]")
            if v.startswith('[') or v.startswith('(') or v.startswith('{'):
                try:
                    v = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    pass
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            config_overrides[k] = v

    # Initialise wandb (disabled by default)
    if args.wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            config=args.config,
            name=args.expt_name,
            tags=args.tags,
            mode='online'
        )
        # 设置 epoch 为默认 x 轴
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
    else:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            config=args.config,
            name=args.expt_name,
            mode='disabled'
        )
    config = wandb.config
    config.update(config_overrides, allow_val_change=True)

    # Log all hyperparameters explicitly to wandb for sweep visibility
    wandb.config.update({
        "lr": config.lr,
        "batch_size": config.batch_size,
        "drop_rate": config.drop_rate,
        "num_anchors": config.num_anchors,
        "length_threshold": config.length_threshold,
        "label_smoothing": config.label_smoothing,
        "noise_scale": config.noise_scale,
        "weight_decay": config.weight_decay,
        "aux_loss_weight": config.aux_loss_weight,
        "epochs": config.epochs,
        "patience": config.patience,
        "model": config.model,
        "split": config.split,
    }, allow_val_change=True)

    # Auto-generate checkpoint_prefix if not specified
    if not config.get('checkpoint_prefix'):
        if args.wandb and wandb.run.id:
            # 启用 wandb 时，使用 run.id 作为前缀（避免 sweep 多次运行覆盖）
            auto_prefix = f"{config.model}_{config.split}_{wandb.run.id}"
        elif config.split == "structsim":
            auto_prefix = f"{config.model}_{config.split}_{config.seed}"
        else:
            auto_prefix = f"{config.model}_{config.split}"
        config.update({'checkpoint_prefix': auto_prefix}, allow_val_change=True)

    config_str = "\nCONFIG"
    for key, val in config.items():
        config_str += f"\n    {key}: {val}"
    print(config_str)

    # Set device (GPU/CPU/XPU)
    if config.device == 'xpu':
        import intel_extension_for_pytorch as ipex
        [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())]
        device = torch.device("xpu:{}".format(config.gpu) if torch.xpu.is_available() else 'cpu')
    else:
        device = torch.device("cuda:{}".format(config.gpu) if torch.cuda.is_available() else "cpu")
    
    # Run main function
    main(config, device)

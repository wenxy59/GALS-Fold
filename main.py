import dotenv
dotenv.load_dotenv(".env")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import random
import argparse
import wandb
import numpy as np

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from src.trainer import train, evaluate
from src.data.dataset import RNADesignDataset, BatchSampler
from src.models import (
    GeometricLongShortRNA,
    GVPAttentionShortBranch,
)
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
    train_loader = get_dataloader(config, trainset, shuffle=True)
    val_loader = get_dataloader(config, valset, shuffle=False)
    test_loader = get_dataloader(config, testset, shuffle=False)
        
    # Run trainer
    train(config, model, train_loader, val_loader, test_loader, device)


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
        ),
        pin_memory = pin_memory,
        exclude_keys = exclude_keys
    )


def get_model(config):
    """
    Returns a Model for a given config.
    """
    model_class = {
        'GALS' : GeometricLongShortRNA,
        'GVPAtten': GVPAttentionShortBranch,
    }[config.model]

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

    if config.model == 'GALS':
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs/default.yaml', type=str)
    parser.add_argument('--expt_name', dest='expt_name', default=None, type=str)
    parser.add_argument('--tags', nargs='+', dest='tags', default=[])
    parser.add_argument('--no_wandb', action="store_true")
    args, unknown = parser.parse_known_args()

    # Parse key=value args for config override
    config_overrides = {}
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
            config_overrides[k] = v

    # Initialise wandb
    if args.no_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"), 
            entity=os.environ.get("WANDB_ENTITY"), 
            config=args.config, 
            name=args.expt_name, 
            mode='disabled'
        )
    else:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"), 
            entity=os.environ.get("WANDB_ENTITY"), 
            config=args.config, 
            name=args.expt_name, 
            tags=args.tags,
            mode='online'
        )
    config = wandb.config
    config.update(config_overrides, allow_val_change=True)

    # Auto-generate checkpoint_prefix if not specified: {model}_{split}
    if not config.get('checkpoint_prefix'):
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
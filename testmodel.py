
import dotenv
dotenv.load_dotenv(".env")
import sys
import os
import random
import numpy as np
from typing import Optional

import torch
import torch.nn.functional as F
import torch_geometric
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.data.featurizer import RNAGraphFeaturizer
from src.models import AutoregressiveMultiGNNv1
from src.data.data_utils import get_backbone_coords
from src.evaluator import edit_distance, self_consistency_score_eternafold
from src.constants import (
    NUM_TO_LETTER, 
    RNA_ATOMS, 
    FILL_VALUE,
    PROJECT_PATH
)

OUTPUT_DIR = "./statistics/testfasta/v13f3"
TrainedModelPATH = "./trainedmodel/best_checkpoint.h5"
IndexFilePATH = "./statistics/kfold_1_test_index.txt"
# Model checkpoint paths corresponding to data split and maximum no. of conformers
CHECKPOINT_PATH = {
    'all': {
        1: os.path.join(PROJECT_PATH, TrainedModelPATH),
        2: os.path.join(PROJECT_PATH, "checkpoints/gRNAde_ARv1_2state_all.h5"),
        3: os.path.join(PROJECT_PATH, "checkpoints/gRNAde_ARv1_3state_all.h5"),
        5: os.path.join(PROJECT_PATH, "checkpoints/gRNAde_ARv1_5state_all.h5"),
    },
    'das': {
        1: os.path.join(PROJECT_PATH, TrainedModelPATH),
        2: os.path.join(PROJECT_PATH, "checkpoints/gRNAde_ARv1_2state_das.h5"),
        3: os.path.join(PROJECT_PATH, "checkpoints/gRNAde_ARv1_3state_das.h5"),
        5: os.path.join(PROJECT_PATH, "checkpoints/gRNAde_ARv1_5state_das.h5"),
    },
    'multi': {
        1: os.path.join(PROJECT_PATH, "checkpoints/gRNAde_ARv1_1state_multi.h5"),
        2: os.path.join(PROJECT_PATH, "checkpoints/gRNAde_ARv1_2state_multi.h5"),
        3: os.path.join(PROJECT_PATH, "checkpoints/gRNAde_ARv1_3state_multi.h5"),
        5: os.path.join(PROJECT_PATH, "checkpoints/gRNAde_ARv1_5state_multi.h5"),
    }
}

# Default model hyperparameters (do not change)
VERSION = 0.1
RADIUS = 0.0
TOP_K = 32
NUM_RBF = 32
NUM_POSENC = 32
NOISE_SCALE = 0.1
NODE_IN_DIM = (15, 4)
NODE_H_DIM = (128, 16)
EDGE_IN_DIM = (131, 3)
EDGE_H_DIM = (64, 4)
NUM_LAYERS = 4
DROP_RATE = 0.5
OUT_DIM = 4
DEFAULT_N_SAMPLES = 16
DEFAULT_TEMPERATURE = 0.1


class gRNAde(object):
    """
    gRNAde: a Geometric Deep Learning pipeline for 3D RNA Inverse Design.

    This class loads a gRNAde inverse folding model checkpoint corresponding 
    to a maximum number of conformers and allows the user to perform fixed 
    backbone re-design of RNA structures.

    Args:
        split (str): data split used to train the model (all/das/multi)
        max_num_conformers (int): maximum number of conformers for an input RNA backbone
        gpu_id (int): GPU ID to use for inference (defaults to cpu if no GPU is available)
    """

    def __init__(
            self,
            split: Optional[str] = "all",
            max_num_conformers: Optional[int] = 1,
            gpu_id: Optional[int] = 0,
        ):

        # Set version
        self.version = VERSION
        print(f"Instantiating {TrainedModelPATH} v{self.version}\n")

        # Set maximum number of conformers
        if max_num_conformers > max(list(CHECKPOINT_PATH[split].keys())):
            max_num_conformers = max(list(CHECKPOINT_PATH[split].keys()))
            print(f"    Invalid max_num_conformers. Setting to maximum value: {max_num_conformers}")
        self.split = split
        self.max_num_conformers = max_num_conformers
        
        # Set device (GPU/CPU)
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
        print(f"    Using device: {device}")
        self.device = device

        # Define data featurizer
        print(f"    Creating RNA graph featurizer for max_num_conformers={max_num_conformers}")
        self.featurizer = RNAGraphFeaturizer(
            split = "test",  # set to 'train' to use noise augmentation
            radius = RADIUS,
            top_k = TOP_K,
            num_rbf = NUM_RBF,
            num_posenc = NUM_POSENC,
            max_num_conformers = max_num_conformers,
            noise_scale = NOISE_SCALE
        )

        # Initialise model
        print(f"    Initialising GNN encoder-decoder model")
        self.model = AutoregressiveMultiGNNv1(
            node_in_dim = NODE_IN_DIM,
            node_h_dim = NODE_H_DIM,
            edge_in_dim = EDGE_IN_DIM,
            edge_h_dim = EDGE_H_DIM,
            num_layers = NUM_LAYERS,
            drop_rate = DROP_RATE,
            out_dim = OUT_DIM,
            # V10 新增参数 (与 main.py 保持一致)
            heads = 4,                  # 注意力头数
            num_anchors = 32,           # 动态投影锚点数
            local_window = 10,          # 局部排斥窗口
            length_threshold = 150,     # 长度门控阈值
            aux_loss_weight = 0.3      # 辅助损失权重（与训练时的默认值一致）
        )
        # Load model checkpoint
        self.model_path = CHECKPOINT_PATH[split][max_num_conformers]
        print(f"    Loading model checkpoint: {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=False))  # 使用 CPU 加载模型

        # Transfer model to device in eval mode
        self.model = self.model.to(device)
        self.model.eval()

        print(f"Finished initialising {TrainedModelPATH} v{self.version}\n")

    def design_from_pdb_file(
            self, 
            pdb_filepath: str,
            output_filepath: Optional[str] = None, 
            n_samples: Optional[int] = DEFAULT_N_SAMPLES,
            temperature: Optional[float] = DEFAULT_TEMPERATURE,
            partial_seq: Optional[str] = None,
            seed: Optional[int] = 0
        ):
        """
        Design RNA sequences for a PDB file, i.e. fixed backbone re-design
        of the RNA structure.

        Args:
            pdb_filepath (str): filepath to PDB file
            output_filepath (str): filepath to write designed sequences to
            n_samples (int): number of samples to generate
            temperature (float): temperature for sampling
            partial_seq (str): partial sequence used to fix nucleotides in 
                designed sequences, provided as a string of nucleotides 
                and underscores (e.g. "AUG___") where letters are fixed 
                and underscores represent designable positions.
            seed (int): random seed for reproducibility
        
        Returns:
            sequences (List[SeqRecord]): designed sequences in fasta format
            samples (Tensor): designed sequences with shape `(n_samples, seq_len)`
            perplexity (Tensor): perplexity per sample with shape `(n_samples, 1)`
            recovery (Tensor): sequence recovery per sample with shape `(n_samples, 1)`
            sc_score (Tensor): global self consistency score per sample with shape `(n_samples, 1)`
        """
        featurized_data, raw_data = self.featurizer.featurize_from_pdb_file(pdb_filepath)
        return self.design(raw_data, featurized_data, output_filepath, n_samples, temperature, partial_seq, seed)

    def design_from_directory(
            self,
            directory_filepath: str,
            output_filepath: Optional[str] = None, 
            n_samples: Optional[int] = DEFAULT_N_SAMPLES,
            temperature: Optional[float] = DEFAULT_TEMPERATURE,
            partial_seq: Optional[str] = None,
            seed: Optional[int] = 0
        ):
        """
        Design RNA sequences for directory of PDB files corresponding to the 
        same RNA molecule, i.e. fixed backbone re-design given multiple
        conformations of the RNA structure.

        Args:
            directory_filepath (str): filepath to directory of PDB files
            output_filepath (str): filepath to write designed sequences to
            n_samples (int): number of samples to generate
            temperature (float): temperature for sampling
            partial_seq (str): partial sequence used to fix nucleotides in 
                designed sequences, provided as a string of nucleotides 
                and underscores (e.g. "AUG___") where letters are fixed 
                and underscores represent designable positions.
            seed (int): random seed for reproducibility
        
        Returns:
            sequences (List[SeqRecord]): designed sequences in fasta format
            samples (Tensor): designed sequences with shape `(n_samples, seq_len)`
            perplexity (Tensor): perplexity per sample with shape `(n_samples, 1)`
            recovery (Tensor): sequence recovery per sample with shape `(n_samples, 1)`
            sc_score (Tensor): global self consistency score per sample with shape `(n_samples, 1)`
        """
        pdb_filelist = []
        for pdb_filepath in os.listdir(directory_filepath):
            if pdb_filepath.endswith(".pdb"):
                pdb_filelist.append(os.path.join(directory_filepath, pdb_filepath))
        featurized_data, raw_data = self.featurizer.featurize_from_pdb_filelist(pdb_filelist)
        return self.design(raw_data, featurized_data, output_filepath, n_samples, temperature, partial_seq, seed)

    @torch.no_grad()
    def design(
        self, 
        raw_data: dict, 
        featurized_data: Optional[torch_geometric.data.Data] = None, 
        output_filepath: Optional[str] = None, 
        n_samples: Optional[int] = DEFAULT_N_SAMPLES,
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        partial_seq: Optional[str] = None,
        seed: Optional[int] = 0
    ):
        """
        Design RNA sequences from raw data.

        Args:
            raw_data (dict): Raw RNA data dictionary with keys:
                - sequence (str): RNA sequence of length `num_res`.
                - coords_list (Tensor): Backbone coordinates with shape
                    `(num_conf, num_res, num_bb_atoms, 3)`.
                - sec_struct_list (List[str]): Secondary structure for each
                    conformer in dotbracket notation.
            featurized_data (torch_geometric.data.Data): featurized RNA data
            output_filepath (str): filepath to write designed sequences to
            n_samples (int): number of samples to generate
            temperature (float): temperature for sampling
            partial_seq (str): partial sequence used to fix nucleotides in 
                designed sequences, provided as a string of nucleotides 
                and underscores (e.g. "AUG___") where letters are fixed 
                and underscores represent designable positions.
            seed (int): random seed for reproducibility
        
        Returns:
            sequences (List[SeqRecord]): designed sequences in fasta format
            samples (Tensor): designed sequences with shape `(n_samples, seq_len)`
            perplexity (Tensor): perplexity per sample with shape `(n_samples, 1)`
            recovery (Tensor): sequence recovery per sample with shape `(n_samples, 1)`
            sc_score (Tensor): global self consistency score per sample with shape `(n_samples, 1)`
        """
        # set random seed
        set_seed(seed)

        if raw_data['coords_list'][0].shape[1] == 3:
            # Expected input: num_conf x num_res x num_bb_atoms x 3
            # Backbone atoms: (P, C4', N1 or N9)
            pass
        elif raw_data['coords_list'][0].shape[1] == len(RNA_ATOMS):
            coords_list = []
            for coords in raw_data['coords_list']:
                # Only keep backbone atom coordinates: num_res x num_bb_atoms x 3
                coords = get_backbone_coords(coords, raw_data['sequence'])
                # Do not add structures with missing coordinates for ALL residues
                if not torch.all((coords == FILL_VALUE).sum(axis=(1,2)) > 0):
                    coords_list.append(coords)

            if len(coords_list) > 0:
                # Add processed coords_list to self.data_list
                raw_data['coords_list'] = coords_list
        else:
            raise ValueError(f"Invalid number of atoms per nucleotide in input data: {raw_data['coords_list'][0].shape[1]}")

        if featurized_data is None:
            # featurize raw data
            featurized_data = self.featurizer.featurize(raw_data)

        # transfer data to device
        featurized_data = featurized_data.to(self.device)

        # create logit bias matrix if partial sequence is provided
        if partial_seq is not None:
            # convert partial sequence to tensor
            _partial_seq = []
            for residue in partial_seq:
                if residue in self.featurizer.letter_to_num.keys():
                    # fixed nucleotide
                    _partial_seq.append(self.featurizer.letter_to_num[residue])
                else:
                    # designable position
                    _partial_seq.append(len(self.featurizer.letter_to_num.keys()))
            _partial_seq = torch.as_tensor(_partial_seq, device=self.device, dtype=torch.long)
            # convert to one-hot and create bias matrix used during sampling
            logit_bias = F.one_hot(_partial_seq, num_classes=self.model.out_dim+1).float()
            logit_bias = logit_bias[:, :-1] * 100.0
        else:
            logit_bias = None
        
        # sample n_samples from model for single data point: n_samples x seq_len
        samples, logits = self.model.sample(
            featurized_data, n_samples, temperature, logit_bias, return_logits=True)

        # perplexity per sample: n_samples x 1
        n_nodes = logits.shape[1]
        perplexity = torch.exp(F.cross_entropy(
            logits.view(n_samples * n_nodes, self.model.out_dim), 
            samples.view(n_samples * n_nodes).long(), 
            reduction="none"
        ).view(n_samples, n_nodes).mean(dim=1)).cpu().numpy()
        
        # sequence recovery per sample: n_samples x 1
        recovery = samples.eq(featurized_data.seq).float().mean(dim=1).cpu().numpy()

        # global self consistency score per sample: n_samples x 1
        sc_score = self_consistency_score_eternafold(
            samples.cpu().numpy(), 
            raw_data['sec_struct_list'], 
            featurized_data.mask_coords.cpu().numpy()
        )

        # collate designed sequences in fasta format
        sequences = [
            # first record: input sequence and model metadata
            SeqRecord(
                Seq(raw_data["sequence"]),
                id=f"input_sequence,",
                description=f"{TrainedModelPATH}_v{self.version}, model={self.model.__class__.__name__}, max_num_conformers={self.max_num_conformers}, checkpoint={self.model_path}, seed={seed}"
            )
        ]
        # remaining records: designed sequences and metrics
        for idx, zipped in enumerate(zip(
            samples.cpu().numpy(),
            perplexity,
            recovery,
            sc_score
        )):
            seq, perp, rec, sc = zipped
            seq = "".join([NUM_TO_LETTER[num] for num in seq])
            edit_dist = edit_distance(seq, raw_data['sequence'])
            sequences.append(SeqRecord(
                Seq(seq), 
                id=f"sample={idx},",
                description=f"seed={seed}, temperature={temperature}, perplexity={perp:.4f}, recovery={rec:.4f}, edit_dist={edit_dist}, sc_score={sc:.4f}"
            ))
        
        if output_filepath is not None:
            # write sequences to output filepath
            SeqIO.write(sequences, output_filepath, "fasta")

        return sequences, samples, perplexity, recovery, sc_score
    
    def perplexity_from_pdb_file(
            self, 
            seq: str,
            pdb_filepath: str,
            temperature: Optional[float] = 1.0,
            seed: Optional[int] = 0
        ):
        """
        Compute perplexity of an RNA sequences for a backbone from a PDB file,
        i.e. P (sequence | backbone structure)

        Args:
            seq (str): RNA sequence
            pdb_filepath (str): filepath to PDB file
            temperature (float): temperature for sampling
            seed (int): random seed for reproducibility
        
        Returns:
            perplexity (float): perplexity for RNA sequence
        """
        featurized_data, raw_data = self.featurizer.featurize_from_pdb_file(pdb_filepath)
        return self.perplexity(seq, raw_data, featurized_data, temperature, seed)
    
    def perplexity_from_directory(
            self,
            seq: str,
            directory_filepath: str,
            temperature: Optional[float] = 1.0,
            seed: Optional[int] = 0
        ):
        """
        Compute perplexity of an RNA sequences for a set of backbones 
        from a directory of PDB files,
        i.e. P (sequence | backbone conformational ensemble)

        Args:
            seq (str): RNA sequence
            directory_filepath (str): filepath to directory of PDB files
            temperature (float): temperature for sampling
            seed (int): random seed for reproducibility

        Returns:
            perplexity (float): perplexity for RNA sequence
        """
        pdb_filelist = []
        for pdb_filepath in os.listdir(directory_filepath):
            if pdb_filepath.endswith(".pdb"):
                pdb_filelist.append(os.path.join(directory_filepath, pdb_filepath))
        featurized_data, raw_data = self.featurizer.featurize_from_pdb_filelist(pdb_filelist)
        return self.perplexity(seq, raw_data, featurized_data, temperature, seed)
    
    @torch.no_grad()
    def perplexity(
        self, 
        seq: str,
        raw_data: dict, 
        featurized_data: Optional[torch_geometric.data.Data] = None, 
        temperature: Optional[float] = 1.0,
        seed: Optional[int] = 0
    ):
        """
        Compute perplexity of an RNA sequence conditioned on 
        one or more backbones from raw data.

        Args:
            seq (str): RNA sequence
            raw_data (dict): Raw RNA data dictionary with keys:
                - sequence (str): RNA sequence of length `num_res`.
                - coords_list (Tensor): Backbone coordinates with shape
                    `(num_conf, num_res, num_bb_atoms, 3)`.
                - sec_struct_list (List[str]): Secondary structure for each
                    conformer in dotbracket notation.
            featurized_data (torch_geometric.data.Data): featurized RNA data
            temperature (float): temperature for sampling
            seed (int): random seed for reproducibility
        
        Returns:
            perplexity (float): perplexity for RNA sequence
        """  
        # set random seed
        set_seed(seed)

        if raw_data['coords_list'][0].shape[1] == 3:
            # Expected input: num_conf x num_res x num_bb_atoms x 3
            # Backbone atoms: (P, C4', N1 or N9)
            pass
        elif raw_data['coords_list'][0].shape[1] == len(RNA_ATOMS):
            coords_list = []
            for coords in raw_data['coords_list']:
                # Only keep backbone atom coordinates: num_res x num_bb_atoms x 3
                coords = get_backbone_coords(coords, raw_data['sequence'])
                # Do not add structures with missing coordinates for ALL residues
                if not torch.all((coords == FILL_VALUE).sum(axis=(1,2)) > 0):
                    coords_list.append(coords)

            if len(coords_list) > 0:
                # Add processed coords_list to self.data_list
                raw_data['coords_list'] = coords_list
        else:
            raise ValueError(f"Invalid number of atoms per nucleotide in input data: {raw_data['coords_list'][0].shape[1]}")

        if featurized_data is None:
            # featurize raw data
            featurized_data = self.featurizer.featurize(raw_data)

        # transfer data to device
        featurized_data = featurized_data.to(self.device)

        # convert sequence to tensor
        _seq = torch.as_tensor(
            [self.featurizer.letter_to_num[residue] for residue in seq], 
            device=self.device, 
            dtype=torch.long
        )
        featurized_data.seq = _seq

        # raw logits for perplexity calculation: seq_len x out_dim
        logits = self.model.forward(featurized_data)

        # compute perplexity
        perplexity = torch.exp(F.cross_entropy(
            logits / temperature, 
            _seq,
            reduction="none"
        ).mean()).cpu().numpy()
        
        return perplexity


def set_seed(seed=0):
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

def read_test(index_file=IndexFilePATH):
    """
    Read test index file and return PDB basenames (without .pdb).
    """
    with open(index_file, 'r', encoding='utf-8') as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')]

def main():
    """
    Main function
    """
    print("=== RNA Test Script Started ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    MODEL_SPLIT = "kfold"
    MODEL_MAX_CONF = 1
    MODEL_N_SAMPLES = 16
    MODEL_TEMP = 0.5
    MODEL_GPU_ID = 0  # 假设使用 GPU 0
    
    print("\nInitializing gRNAde model...")
    g = gRNAde(
        split=MODEL_SPLIT,
        max_num_conformers=MODEL_MAX_CONF, 
        gpu_id=MODEL_GPU_ID
        )
    print("✓ Model initialized successfully.")
    
    # Read test index file
    pdb_basenames = read_test(IndexFilePATH)
    print(f"\nFound {len(pdb_basenames)} PDB files to process:")
    for i, name in enumerate(pdb_basenames, 1):
        print(f"  {i}. {name}")
    print("\nStarting processing...")
    
    success_count = 0
    failed_files = []
    
    # Process each PDB file
    for i, pdb_basename in enumerate(pdb_basenames, 1):
        print(f"\n[{i}/{len(pdb_basenames)}] " + "="*50)
        
        input_filename = f"{pdb_basename}.pdb"
        input_path = (Path(PROJECT_PATH) / "data/raw") / input_filename

        if not input_path.exists():
            print(f"Warning: Input file {input_path} does not exist, skipping...")
            failed_files.append(pdb_basename)
            continue
        
        output_file = f"{OUTPUT_DIR}/{pdb_basename}.fasta"
        print(f"Processing: {input_path}")
        
        try:
            # 直接调用模型方法
            g.design_from_pdb_file(
                pdb_filepath=str(input_path),
                output_filepath=output_file,
                n_samples=MODEL_N_SAMPLES,
                temperature=MODEL_TEMP,
                seed=42  # 你可以为所有运行使用同一个
            )
            
            print(f"Successfully processed {pdb_basename}")
            print(f"Output file: {output_file}")
            success_count += 1
            
        except Exception as e:
            # 捕获处理单个文件时可能发生的任何错误
            print(f"Failed to process {pdb_basename}")
            print(f"Error: {e}")
            failed_files.append(pdb_basename)

    
    # 打印总结 (这部分和你原来的一样)
    print("\n" + "="*60)
    print("=== Processing Completed ===")
    print(f"Total: {len(pdb_basenames)} files")
    print(f"Success: {success_count} files")
    print(f"Failed: {len(failed_files)} files")
    
    if failed_files:
        print("\nFailed files (basenames):")
        for failed_file in failed_files:
            print(f"  - {failed_file}")
    
    print(f"\nOutput files saved in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()



import os
import subprocess
import tempfile
from datetime import datetime
import numpy as np
import wandb
from typing import Any, List, Literal, Optional

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import biotite
from biotite.structure.io import load_structure
from biotite.structure import dot_bracket_from_structure

from src.constants import (
    PROJECT_PATH,
    X3DNA_PATH,
    ETERNAFOLD_PATH,
    LINEARFOLD_PATH,
    LINEARFOLD_LENGTH_THRESHOLD,
    DOTBRACKET_TO_NUM
)


def pdb_to_sec_struct(
        pdb_file_path: str,
        sequence: str,
        keep_pseudoknots: bool = False,
        x3dna_path: str = os.path.join(X3DNA_PATH, "bin/find_pair"),
        max_len_for_biotite: int = 1000,
    ) -> str:
    """
    Get secondary structure in dot-bracket notation from a PDB file.
    
    Args:
        pdb_file_path (str): Path to PDB file.
        sequence (str): Sequence of RNA molecule.
        keep_pseudoknots (bool, optional): Whether to keep pseudoknots in 
            secondary structure. Defaults to False.
        x3dna_path (str, optional): Path to x3dna find_pair tool.
        max_len_for_biotite (int, optional): Maximum length of sequence for
            which to use biotite. Otherwise use X3DNA Defaults to 1000.
    """
    if len(sequence) < max_len_for_biotite:
        try:
            # get secondary structure using biotite
            atom_array = load_structure(pdb_file_path)
            sec_struct = dot_bracket_from_structure(atom_array)[0]
            if not keep_pseudoknots:
                # replace all characters that are not '.', '(', ')' with '.'
                sec_struct = "".join([dotbrac if dotbrac in ['.', '(', ')'] else '.' for dotbrac in sec_struct])
        
        except Exception as e:
            # biotite fails for very short seqeunces
            if "out of bounds for array" not in str(e): raise e
            # get secondary structure using x3dna find_pair tool
            # does not support pseudoknots
            sec_struct = x3dna_to_sec_struct(
                pdb_to_x3dna(pdb_file_path, x3dna_path), 
                sequence
            )

    else:
        # get secondary structure using x3dna find_pair tool
        # does not support pseudoknots
        sec_struct = x3dna_to_sec_struct(
            pdb_to_x3dna(pdb_file_path, x3dna_path), 
            sequence
        )
    
    return sec_struct

def pdb_to_x3dna(
        pdb_file_path: str, 
        x3dna_path: str = os.path.join(X3DNA_PATH, "bin/find_pair")
    ) -> List[str]:
    # Run x3dna find_pair tool
    cmd = [
        x3dna_path,
        pdb_file_path,
    ]
    output = subprocess.run(cmd, check=True, capture_output=True).stdout.decode("utf-8")
    output = output.split("\n")

    # Delete temporary files
    # os.remove("./bestpairs.pdb")
    # os.remove("./bp_order.dat")
    # os.remove("./col_chains.scr")
    # os.remove("./col_helices.scr")
    # os.remove("./hel_regions.pdb")
    # os.remove("./ref_frames.dat")

    return output


def x3dna_to_sec_struct(output: List[str], sequence: str) -> str:
    # Secondary structure in dot-bracket notation
    num_base_pairs = int(output[3].split()[0])
    sec_struct = ["."] * len(sequence)
    for i in range(1, num_base_pairs+1):
        line = output[4 + i].split()
        start, end = int(line[0]), int(line[1])
        sec_struct[start-1] = "("
        sec_struct[end-1] = ")"
    return "".join(sec_struct)


def predict_sec_struct_linearfold(
        sequence: str,
        linearfold_path: str = os.path.join(LINEARFOLD_PATH, "linearfold"),
        use_vienna_params: bool = False,
        beam_size: int = 100,
    ) -> List[str]:
    """
    Predict secondary structure using LinearFold.

    LinearFold has O(N) time complexity, making it suitable for long sequences.

    Args:
        sequence (str): RNA sequence (can contain T, will be converted to U).
        linearfold_path (str): Path to LinearFold executable.
        use_vienna_params (bool): If True, use Vienna energy model; else use CONTRAfold model.
        beam_size (int): Beam size for beam search (larger = more accurate but slower).

    Returns:
        List[str]: List containing the predicted secondary structure in dot-bracket notation.
    """
    # Convert T to U (LinearFold uses RNA alphabet)
    sequence = sequence.upper().replace('T', 'U')

    # Build command
    cmd_args = [linearfold_path]
    if use_vienna_params:
        cmd_args.append("-V")
    cmd_args.extend(["-b", str(beam_size)])

    # Run LinearFold
    process = subprocess.run(
        cmd_args,
        input=sequence,
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse output
    # LinearFold output format:
    # Line 1: sequence
    # Line 2: structure (energy)
    output_lines = process.stdout.strip().split("\n")
    if len(output_lines) >= 2:
        # Extract structure from second line (format: "...((...))... (-1.23)")
        structure_line = output_lines[-1]
        # Remove energy part (the part with parentheses containing a number)
        structure = structure_line.split()[0]
    else:
        # Fallback: return all unpaired
        structure = '.' * len(sequence)

    return [structure]


def predict_sec_struct_eternafold(
        sequence: Optional[str] = None,
        fasta_file_path: Optional[str] = None,
        eternafold_path: str = os.path.join(ETERNAFOLD_PATH, "src/contrafold"),
        n_samples: int = 1,
    ) -> List[str]:
    """
    Predict secondary structure using EternaFold.

    Notes:
    - EternaFold does not support pseudoknots.
    - EternaFold only supports single chains in a fasta file.
    - When sampling multiple structures, EternaFold only supports nsamples=100.

    Args:
        sequence (str, optional): Sequence of RNA molecule. Defaults to None.
        fasta_file_path (str, optional): Path to fasta file. Defaults to None.
        eternafold_path (str, optional): Path to EternaFold. Defaults to ETERNAFOLD_PATH env variable.
        n_samples (int, optional): Number of samples to take. Defaults to 1.
    """
    if sequence is not None:
        assert fasta_file_path is None
        # Write sequence to temporary fasta file using tempfile for unique naming
        try:
            temp_dir = wandb.run.dir
        except AttributeError:
            temp_dir = PROJECT_PATH

        # Create a unique temporary file to avoid conflicts in concurrent execution
        fd, fasta_file_path = tempfile.mkstemp(suffix='.fasta', prefix='temp_', dir=temp_dir)
        os.close(fd)  # Close the file descriptor, we'll write using SeqIO
        SeqIO.write(
            SeqRecord(Seq(sequence), id="temp"),
            fasta_file_path, "fasta"
        )

    # Run EternaFold
    if n_samples > 1:
        assert n_samples == 100, "EternaFold using subprocess only supports nsamples=100"
        cmd = [
            eternafold_path, 
            "sample",
            fasta_file_path,
            # f" --nsamples {n_samples}",
            # It seems like EternaFold using subprocess can only sample the default nsamples=100...
            # Reason: unknown for now
        ]
    else:
        cmd = [
            eternafold_path, 
            "predict",
            fasta_file_path,
        ]

    output = subprocess.run(cmd, check=True, capture_output=True).stdout.decode("utf-8")

    # Delete temporary files
    if sequence is not None:
        os.remove(fasta_file_path)

    if n_samples > 1:
        return output.split("\n")[:-1]
    else:
        return [output.split("\n")[-2]]


def predict_sec_struct(
        sequence: Optional[str] = None,
        fasta_file_path: Optional[str] = None,
        n_samples: int = 1,
        length_threshold: int = LINEARFOLD_LENGTH_THRESHOLD,
        force_tool: Optional[Literal["eternafold", "linearfold"]] = None,
    ) -> List[str]:
    """
    Predict secondary structure using the appropriate tool based on sequence length.

    - For sequences <= length_threshold: use EternaFold (more accurate)
    - For sequences > length_threshold: use LinearFold (O(N) complexity, handles long sequences)

    Args:
        sequence (str, optional): Sequence of RNA molecule.
        fasta_file_path (str, optional): Path to fasta file (only for EternaFold).
        n_samples (int): Number of samples (only supported by EternaFold, must be 1 or 100).
        length_threshold (int): Length threshold for switching tools. Default: LINEARFOLD_LENGTH_THRESHOLD.
        force_tool (str, optional): Force use of specific tool ("eternafold" or "linearfold").

    Returns:
        List[str]: List of predicted secondary structures in dot-bracket notation.
    """
    # Determine sequence length
    if sequence is not None:
        seq_len = len(sequence)
    elif fasta_file_path is not None:
        # Read sequence from fasta to get length
        record = next(SeqIO.parse(fasta_file_path, "fasta"))
        seq_len = len(record.seq)
        sequence = str(record.seq)
    else:
        raise ValueError("Either sequence or fasta_file_path must be provided")

    # Determine which tool to use
    if force_tool == "linearfold":
        use_linearfold = True
    elif force_tool == "eternafold":
        use_linearfold = False
    else:
        # Auto-select based on length
        use_linearfold = seq_len > length_threshold

    # Use LinearFold for long sequences
    if use_linearfold:
        if n_samples > 1:
            # LinearFold doesn't support sampling, just return the same structure n times
            structure = predict_sec_struct_linearfold(sequence)[0]
            return [structure] * n_samples
        return predict_sec_struct_linearfold(sequence)

    # Use EternaFold for short sequences
    return predict_sec_struct_eternafold(
        sequence=sequence,
        fasta_file_path=None,  # We already have the sequence
        n_samples=n_samples,
    )


def dotbracket_to_paired(sec_struct: str) -> np.ndarray:
    """
    Return whether each residue is paired (1) or unpaired (0) given 
    secondary structure in dot-bracket notation.
    """
    is_paired = np.zeros(len(sec_struct), dtype=np.int8)
    for i, c in enumerate(sec_struct):
        if c == '(' or c == ')':
            is_paired[i] = 1
    return is_paired


def dotbracket_to_num(sec_struct: str) -> np.ndarray:
    """
    Convert secondary structure in dot-bracket notation to 
    numerical representation.
    """
    return np.array([DOTBRACKET_TO_NUM[c] for c in sec_struct])


def dotbracket_to_adjacency(
        sec_struct: str,
        keep_pseudoknots: bool = False,
    ) -> np.ndarray:
    """
    Convert secondary structure in dot-bracket notation to 
    adjacency matrix.
    """
    n = len(sec_struct)
    adj = np.zeros((n, n), dtype=np.int8)
        
    if keep_pseudoknots == False:
        stack = []
        for i, db_char in enumerate(sec_struct):
            if db_char == '(':
                stack.append(i)
            elif db_char == ')':
                j = stack.pop()
                adj[i, j] = 1
                adj[j, i] = 1
    else:
        stack={
            '(':[],
            '[':[],
            '<':[],
            '{':[]
        }
        pop={
            ')':'(',
            ']':'[',
            '>':"<",
            '}':'{'
        }
        for i, db_char in enumerate(sec_struct):
            if db_char in stack:
                stack[db_char].append((i, db_char))
            elif db_char in pop:
                forward_bracket = stack[pop[db_char]].pop()
                adj[forward_bracket[0], i] = 1
                adj[i, forward_bracket[0]] = 1    
    return adj

"""Deals with input/output operations."""

import mrcfile
import torch
import einops
import starfile
import sys
import yaml


def read_inputs(
        file_path: str
) -> dict:
    """
    Reads and parses the inputs from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML data as a Python dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit()
    except yaml.YAMLError as e:
        print(f"Error: Issue reading the YAML file. {e}")
        sys.exit()

def load_mrc_map(
        file_path: str
) -> torch.Tensor:
    """Load MRC file into a torch tensor."""

    with mrcfile.open(file_path) as mrc:
        return torch.tensor(mrc.data)
    
def load_mrc_micrographs(
        file_paths: list[str]
) -> torch.Tensor:
    """Load MRC micrographs into a torch tensor."""

    for i, file_path in enumerate(file_paths):
        with mrcfile.open(file_path) as mrc:
            if i == 0:
                mrc_data = torch.tensor(mrc.data, dtype=torch.float32)
                mrc_data = einops.rearrange(mrc_data, 'h w -> 1 h w')
            else:
                mrc_data = torch.cat((mrc_data, einops.rearrange(torch.tensor(mrc.data, dtype=torch.float32), 'h w -> 1 h w')), dim=0)
    return mrc_data


def load_relion_starfile(
        file_path: str # Path to relion ctffind output
) -> tuple  :
    """Load micrographs_ctf.star into dataframe"""

    star = starfile.read(file_path)
    # merge into one dataframe
    df = star['micrographs'].merge(star['optics'], on='rlnOpticsGroup')
    return df

# some functions here to convert parts of dict or df into tensors



"""Utility functions dealing with basic data I/O operations."""

import os
from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np
import torch


def read_mrc_to_numpy(mrc_path: str | os.PathLike | Path) -> np.ndarray:
    """Reads an MRC file and returns the data as a numpy array.

    Attributes
    ----------
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    np.ndarray
        The MRC data as a numpy array, copied.
    """
    with mrcfile.open(mrc_path) as mrc:
        return mrc.data.copy()


def read_mrc_to_tensor(mrc_path: str | os.PathLike | Path) -> torch.Tensor:
    """Reads an MRC file and returns the data as a torch tensor.

    Attributes
    ----------
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC data as a tensor, copied.
    """
    return read_mrc_to_numpy(mrc_path)


def write_mrc_from_numpy(
    data: np.ndarray,
    mrc_path: str | os.PathLike | Path,
    mrc_header: Optional[dict] = None,
) -> None:
    """Writes a numpy array to an MRC file.

    NOTE: Not currently implemented.

    Attributes
    ----------
    data : np.ndarray
        The data to write to the MRC file.
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.
    mrc_header : Optional[dict]
        Dictionary containing header information. Default is None.
    """
    # TODO: Figure out how to set info in the header
    raise NotImplementedError()


def write_mrc_from_tensor(
    data: torch.Tensor,
    mrc_path: str | os.PathLike | Path,
    mrc_header: Optional[dict] = None,
) -> None:
    """Writes a tensor array to an MRC file.

    NOTE: Not currently implemented.

    Attributes
    ----------
    data : np.ndarray
        The data to write to the MRC file.
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.
    mrc_header : Optional[dict]
        Dictionary containing header information. Default is None.
    """
    # TODO: Figure out how to set info in the header
    write_mrc_from_numpy(data.numpy(), mrc_path, mrc_header)


def load_mrc_image(file_path: str | os.PathLike | Path) -> torch.Tensor:
    """Helper function for loading an two-dimensional MRC image into a tensor.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC image as a tensor.

    Raises
    ------
    ValueError
        If the MRC file is not two-dimensional.
    """
    tensor = read_mrc_to_tensor(file_path)

    # Check that tensor is 2D, squeezing if necessary
    tensor = tensor.squeeze()
    if len(tensor.shape) != 2:
        raise ValueError("MRC file is not two-dimensional. Got shape: {tmp.shape}")

    return tensor


def load_mrc_volume(file_path: str | os.PathLike | Path) -> torch.Tensor:
    """Helper function for loading an three-dimensional MRC volume into a tensor.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC volume as a tensor.

    Raises
    ------
    ValueError
        If the MRC file is not three-dimensional.
    """
    tensor = read_mrc_to_tensor(file_path)

    # Check that tensor is 3D, squeezing if necessary
    tensor = tensor.squeeze()
    if len(tensor.shape) != 3:
        raise ValueError("MRC file is not three-dimensional. Got shape: {tmp.shape}")

    return tensor

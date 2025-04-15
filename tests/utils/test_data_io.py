"""Tests for the data_io module"""

import os
import pathlib
import tempfile

import mrcfile
import numpy as np
import pytest
import torch

from leopard_em.utils.data_io import (
    load_mrc_image,
    load_mrc_volume,
    read_mrc_to_numpy,
    read_mrc_to_tensor,
    write_mrc_from_numpy,
    write_mrc_from_tensor,
)

EXAMPLE_IMAGE_PATH = "data/test_image.mrc"
EXAMPLE_VOLUME_PATH = "data/test_volume.mrc"

# Evaluate the relative paths to this file
EXAMPLE_IMAGE_PATH = pathlib.Path(__file__).parent.parent / EXAMPLE_IMAGE_PATH
EXAMPLE_VOLUME_PATH = pathlib.Path(__file__).parent.parent / EXAMPLE_VOLUME_PATH


def create_test_mrc_file(data: np.ndarray, file_path: str) -> None:
    """Helper fn to create a test MRC file."""
    with mrcfile.new(file_path, overwrite=True) as mrc:
        mrc.set_data(data)


def test_read_mrc_to_numpy():
    """Test the read_mrc_to_numpy function."""
    result = read_mrc_to_numpy(EXAMPLE_IMAGE_PATH)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2


def test_read_mrc_to_tensor():
    """Test the read_mrc_to_tensor function."""
    result = read_mrc_to_tensor(EXAMPLE_IMAGE_PATH)

    assert isinstance(result, torch.Tensor)
    assert result.ndim == 2


def test_write_mrc_from_numpy():
    """Test writing an MRC file from a numpy array."""
    data = np.random.rand(10, 10).astype(np.float32)

    # Create a temporary file to write to
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as temp_file:
        write_mrc_from_numpy(data, temp_file.name, overwrite=True)
        with mrcfile.open(temp_file.name) as mrc:
            np.testing.assert_array_equal(mrc.data, data)

    # Finally, remove the temporary file
    os.remove(temp_file.name)


def test_write_mrc_from_tensor():
    """Test wriniting an MRC file from a torch tensor."""
    data = torch.rand(10, 10, dtype=torch.float32)

    # Create a temporary file to write to
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as temp_file:
        write_mrc_from_tensor(data, temp_file.name, overwrite=True)
        with mrcfile.open(temp_file.name) as mrc:
            np.testing.assert_array_equal(mrc.data, data.numpy())

    # Finally, remove the temporary file
    os.remove(temp_file.name)


def test_load_mrc_image():
    """Test loading an MRC image into a tensor."""
    result = load_mrc_image(EXAMPLE_IMAGE_PATH)

    assert isinstance(result, torch.Tensor)
    assert result.ndim == 2

    # Ensure the method raises a ValueError if the MRC file is not two-dimensional
    with pytest.raises(ValueError, match="MRC file is not two-dimensional"):
        load_mrc_image(EXAMPLE_VOLUME_PATH)


def test_load_mrc_volume():
    """Test loading an MRC volume into a tensor."""
    result = load_mrc_volume(EXAMPLE_VOLUME_PATH)

    assert isinstance(result, torch.Tensor)
    assert result.ndim == 3

    # Ensure the method raises a ValueError if the MRC file is not three-dimensional
    with pytest.raises(ValueError, match="MRC file is not three-dimensional"):
        load_mrc_volume(EXAMPLE_IMAGE_PATH)

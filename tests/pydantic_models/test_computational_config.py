"""Tests for the ComputationalConfig model"""

import pytest
import torch
from pydantic import ValidationError

from leopard_em.pydantic_models.computational_config import ComputationalConfig


def test_default_values():
    """
    Test the default values of the ComputationalConfig.

    Verifies that the default values for gpu_ids and num_cpus are set correctly.
    """
    config = ComputationalConfig()
    assert config.gpu_ids == [0]
    assert config.num_cpus == 1


def test_single_gpu_id():
    """
    Test that a single integer gpu_id is converted to a list.

    Verifies that when passing a single integer as gpu_ids, it's converted to a list
    containing that integer.
    """
    config = ComputationalConfig(gpu_ids=1)
    assert config.gpu_ids == [1]


def test_multiple_gpu_ids():
    """
    Test that multiple gpu_ids are correctly stored as a list.

    Verifies that when passing a list of gpu_ids, they are correctly stored in the
    config.
    """
    config = ComputationalConfig(gpu_ids=[0, 1, 2])
    assert config.gpu_ids == [0, 1, 2]


def test_all_gpus():
    """
    Test the special value -1 for gpu_ids.

    Verifies that when passing -1 as gpu_ids, it's stored correctly, which indicates
    using all available GPUs.
    """
    config = ComputationalConfig(gpu_ids=-1)
    assert config.gpu_ids == [-1]


def test_cpu_only():
    """
    Test the special value -2 for gpu_ids.

    Verifies that when passing -2 as gpu_ids, it's stored correctly, which indicates
    using CPU only.
    """
    config = ComputationalConfig(gpu_ids=-2)
    assert config.gpu_ids == [-2]


def test_invalid_gpu_ids():
    """
    Test that invalid combinations of gpu_ids raise errors.

    Verifies that special values -1 and -2 cannot be combined with other gpu IDs.
    """
    with pytest.raises(ValueError, match="If -1"):
        ComputationalConfig(gpu_ids=[-1, 0])

    with pytest.raises(ValueError, match="If -2"):
        ComputationalConfig(gpu_ids=[-2, 0])


def test_num_cpus():
    """
    Test that num_cpus is correctly stored.

    Verifies that the specified number of CPUs is correctly stored in the config.
    """
    config = ComputationalConfig(num_cpus=4)
    assert config.num_cpus == 4


def test_invalid_num_cpus():
    """
    Test that invalid num_cpus values raise errors.

    Verifies that specifying zero or negative values for num_cpus raises a
    ValidationError.
    """
    with pytest.raises(ValidationError):
        ComputationalConfig(num_cpus=0)


def test_gpu_devices_single_gpu():
    """
    Test the gpu_devices property for a single GPU configuration.

    Verifies that the correct torch.device object is created for a single GPU.
    """
    config = ComputationalConfig(gpu_ids=0)
    assert config.gpu_devices == [torch.device("cuda:0")]


def test_gpu_devices_multiple_gpus():
    """
    Test the gpu_devices property for multiple GPUs configuration.

    Verifies that the correct torch.device objects are created for multiple GPUs.
    """
    config = ComputationalConfig(gpu_ids=[0, 1])
    assert config.gpu_devices == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_gpu_devices_all_gpus(monkeypatch):
    """
    Test the gpu_devices property when using all available GPUs.

    Uses monkeypatch to set a fixed number of GPUs for testing, and verifies
    that the correct device objects are created for all available GPUs.
    """
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    config = ComputationalConfig(gpu_ids=-1)
    assert config.gpu_devices == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_gpu_devices_cpu():
    """
    Test the gpu_devices property in CPU-only mode.

    Verifies that the correct torch.device object for CPU is created when using
    CPU-only mode.
    """
    config = ComputationalConfig(gpu_ids=-2)
    assert config.gpu_devices == [torch.device("cpu")]

import pytest
import torch
from pydantic import ValidationError

from leopard_em.pydantic_models.computational_config import ComputationalConfig


def test_default_values():
    config = ComputationalConfig()
    assert config.gpu_ids == [0]
    assert config.num_cpus == 1


def test_single_gpu_id():
    config = ComputationalConfig(gpu_ids=1)
    assert config.gpu_ids == [1]


def test_multiple_gpu_ids():
    config = ComputationalConfig(gpu_ids=[0, 1, 2])
    assert config.gpu_ids == [0, 1, 2]


def test_all_gpus():
    config = ComputationalConfig(gpu_ids=-1)
    assert config.gpu_ids == [-1]


def test_cpu_only():
    config = ComputationalConfig(gpu_ids=-2)
    assert config.gpu_ids == [-2]


def test_invalid_gpu_ids():
    with pytest.raises(ValueError, match="If -1"):
        ComputationalConfig(gpu_ids=[-1, 0])

    with pytest.raises(ValueError, match="If -2"):
        ComputationalConfig(gpu_ids=[-2, 0])


def test_num_cpus():
    config = ComputationalConfig(num_cpus=4)
    assert config.num_cpus == 4


def test_invalid_num_cpus():
    with pytest.raises(ValidationError):
        ComputationalConfig(num_cpus=0)


def test_gpu_devices_single_gpu():
    config = ComputationalConfig(gpu_ids=0)
    assert config.gpu_devices == [torch.device("cuda:0")]


def test_gpu_devices_multiple_gpus():
    config = ComputationalConfig(gpu_ids=[0, 1])
    assert config.gpu_devices == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_gpu_devices_all_gpus(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    config = ComputationalConfig(gpu_ids=-1)
    assert config.gpu_devices == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_gpu_devices_cpu():
    config = ComputationalConfig(gpu_ids=-2)
    assert config.gpu_devices == [torch.device("cpu")]

import pytest
import torch
from pydantic import ValidationError

from leopard_em.pydantic_models.orientation_search import (
    OrientationSearchConfig,
    RefineOrientationConfig,
)


def test_orientation_search_default_values():
    config = OrientationSearchConfig()
    assert config.in_plane_step == 1.5
    assert config.out_of_plane_step == 2.5
    assert config.psi_min == 0.0
    assert config.psi_max == 360.0
    assert config.theta_min == 0.0
    assert config.theta_max == 180.0
    assert config.phi_min == 0.0
    assert config.phi_max == 360.0
    assert config.base_grid_method == "uniform"


def test_orientation_search_invalid_in_plane_step():
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        OrientationSearchConfig(in_plane_step=-1.0)


def test_orientation_search_invalid_out_of_plane_step():
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        OrientationSearchConfig(out_of_plane_step=-1.0)


def test_orientation_search_euler_angles():
    config = OrientationSearchConfig(in_plane_step=90.0, out_of_plane_step=90.0)
    euler_angles = config.euler_angles

    assert isinstance(euler_angles, torch.Tensor)
    assert euler_angles.shape == (24, 3)


def test_refine_orientation_default_values():
    config = RefineOrientationConfig()
    assert config.in_plane_angular_step_coarse == 1.5
    assert config.in_plane_angular_step_fine == 0.15
    assert config.out_of_plane_angular_step_coarse == 2.5
    assert config.out_of_plane_angular_step_fine == 0.25


def test_refine_orientation_invalid_in_plane_angular_step_coarse():
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(in_plane_angular_step_coarse=-1.0)


def test_refine_orientation_invalid_in_plane_angular_step_fine():
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(in_plane_angular_step_fine=-1.0)


def test_refine_orientation_invalid_out_of_plane_angular_step_coarse():
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(out_of_plane_angular_step_coarse=-1.0)


def test_refine_orientation_invalid_out_of_plane_angular_step_fine():
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(out_of_plane_angular_step_fine=-1.0)


def test_refine_orientation_euler_angles_offsets():
    config = RefineOrientationConfig(
        in_plane_angular_step_coarse=1.5,
        in_plane_angular_step_fine=0.5,
        out_of_plane_angular_step_coarse=2.5,
        out_of_plane_angular_step_fine=1.0,
    )
    euler_angles_offsets = config.euler_angles_offsets

    assert isinstance(euler_angles_offsets, torch.Tensor)
    assert euler_angles_offsets.shape == (126, 3)

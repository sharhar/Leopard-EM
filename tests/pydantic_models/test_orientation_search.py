"""Tests for the OrientationSearchConfig and RefineOrientationConfig models"""

import sys

import pytest
import torch
from pydantic import ValidationError

# Skip tests for only this file because Windows tests will generate warnings
# about healpy not being compatible with Windows. This is easier than having some more
# complex logic specific to only Windows platforms within the torch-so3 package.
if sys.platform.startswith("win"):
    pytestmark = pytest.mark.skipif(
        True,
        reason="Skip tests for Windows platform due to healpy warnings",
    )
else:
    from leopard_em.pydantic_models.config import (
        OrientationSearchConfig,
        RefineOrientationConfig,
    )


def test_orientation_search_default_values():
    """
    Test the default values of the OrientationSearchConfig.

    Verifies that all default properties, including angular ranges and steps,
    are correctly set to their expected values.
    """
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
    """
    Test that an error is raised for invalid in-plane step values.

    Verifies that a ValidationError is raised when in_plane_step is negative.
    """
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        OrientationSearchConfig(in_plane_step=-1.0)


def test_orientation_search_invalid_out_of_plane_step():
    """
    Test that an error is raised for invalid out-of-plane step values.

    Verifies that a ValidationError is raised when out_of_plane_step is negative.
    """
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        OrientationSearchConfig(out_of_plane_step=-1.0)


def test_orientation_search_euler_angles():
    """
    Test that the euler_angles property generates the correct tensor.

    Verifies that the generated Euler angles tensor has the expected shape and type
    when using specific in-plane and out-of-plane step values.
    """
    config = OrientationSearchConfig(in_plane_step=90.0, out_of_plane_step=90.0)
    euler_angles = config.euler_angles

    assert isinstance(euler_angles, torch.Tensor)
    assert euler_angles.shape == (24, 3)


def test_refine_orientation_default_values():
    """
    Test the default values of the RefineOrientationConfig.

    Verifies that all default angular step parameters for coarse and fine
    refinement are correctly set to their expected values.
    """
    config = RefineOrientationConfig()
    assert config.in_plane_angular_step_coarse == 1.5
    assert config.in_plane_angular_step_fine == 0.1
    assert config.out_of_plane_angular_step_coarse == 2.5
    assert config.out_of_plane_angular_step_fine == 0.25


def test_refine_orientation_invalid_in_plane_angular_step_coarse():
    """
    Test that an error is raised for invalid in-plane coarse step values.

    Verifies that a ValidationError is raised when in_plane_angular_step_coarse
    is negative.
    """
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(in_plane_angular_step_coarse=-1.0)


def test_refine_orientation_invalid_in_plane_angular_step_fine():
    """
    Test that an error is raised for invalid in-plane fine step values.

    Verifies that a ValidationError is raised when in_plane_angular_step_fine
    is negative.
    """
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(in_plane_angular_step_fine=-1.0)


def test_refine_orientation_invalid_out_of_plane_angular_step_coarse():
    """
    Test that an error is raised for invalid out-of-plane coarse step values.

    Verifies that a ValidationError is raised when out_of_plane_angular_step_coarse
    is negative.
    """
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(out_of_plane_angular_step_coarse=-1.0)


def test_refine_orientation_invalid_out_of_plane_angular_step_fine():
    """
    Test that an error is raised for invalid out-of-plane fine step values.

    Verifies that a ValidationError is raised when out_of_plane_angular_step_fine
    is negative.
    """
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(out_of_plane_angular_step_fine=-1.0)


def test_refine_orientation_euler_angles_offsets():
    """
    Test that the euler_angles_offsets property generates the correct tensor.

    Verifies that the generated Euler angles offsets tensor has the expected shape and
    type when using specific angular step values for both coarse and fine refinement.
    """
    config = RefineOrientationConfig(
        in_plane_angular_step_coarse=1.5,
        in_plane_angular_step_fine=0.5,
        out_of_plane_angular_step_coarse=2.5,
        out_of_plane_angular_step_fine=1.0,
    )
    euler_angles_offsets = config.euler_angles_offsets

    assert isinstance(euler_angles_offsets, torch.Tensor)
    assert euler_angles_offsets.shape == (126, 3)

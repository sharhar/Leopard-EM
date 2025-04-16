"""Tests for the DefocusSearchConfig model"""

import pytest
import torch

from leopard_em.pydantic_models.config import DefocusSearchConfig


def test_default_values():
    """
    Test the default values of the DefocusSearchConfig.

    Verifies that the 'enabled' property defaults to True and that the
    defocus parameters are correctly set.
    """
    config = DefocusSearchConfig(
        defocus_min=-1000.0, defocus_max=1000.0, defocus_step=100.0
    )
    assert config.enabled is True
    assert config.defocus_min == -1000.0
    assert config.defocus_max == 1000.0
    assert config.defocus_step == 100.0


def test_defocus_values():
    """
    Test that the defocus_values property generates the correct range of values.

    Ensures that the range from defocus_min to defocus_max with step size defocus_step
    is correctly generated as a tensor.
    """
    config = DefocusSearchConfig(
        defocus_min=-1000.0, defocus_max=1000.0, defocus_step=100.0
    )
    assert torch.allclose(
        config.defocus_values,
        torch.tensor(
            [
                -1000,
                -900,
                -800,
                -700,
                -600,
                -500,
                -400,
                -300,
                -200,
                -100,
                0,
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
            ],
            dtype=torch.float32,
        ),
    )


def test_defocus_values_disabled():
    """
    Test that when defocus search is disabled, only a single zero tensor is returned.

    Verifies that the defocus_values property returns [0.0] when enabled=False,
    regardless of the min/max/step parameters.
    """
    config = DefocusSearchConfig(
        enabled=False, defocus_min=-1000.0, defocus_max=1000.0, defocus_step=100.0
    )
    assert config.defocus_values == torch.tensor([0.0])


def test_defocus_values_no_values():
    """
    Test that an error is raised when the defocus parameters would result in no values.

    Verifies that a ValueError is raised when defocus_min > defocus_max, which would
    result in an empty range of values.
    """
    with pytest.raises(
        ValueError,
        match="Defocus search parameters result in no values to search over!",
    ):
        df_config = DefocusSearchConfig(
            defocus_min=1000.0, defocus_max=-1000.0, defocus_step=100.0
        )
        _ = df_config.defocus_values


def test_defocus_values_with_step():
    """
    Test that the defocus_values property correctly handles different step sizes.

    Verifies that the correct values are generated when using a larger step size.
    """
    config = DefocusSearchConfig(
        defocus_min=-400.0, defocus_max=400.0, defocus_step=200.0
    )
    assert torch.allclose(
        config.defocus_values,
        torch.tensor([-400, -200, 0, 200, 400], dtype=torch.float32),
    )

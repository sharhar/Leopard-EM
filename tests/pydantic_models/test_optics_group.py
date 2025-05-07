"""Tests for the OpticsGroup model"""

import pytest
from pydantic import ValidationError

from leopard_em.pydantic_models.data_structures import OpticsGroup


def test_default_values():
    """
    Test the default values of the OpticsGroup.

    Verifies that properties without explicit values in the constructor
    are set to their default values.
    """
    config = OpticsGroup(
        label="test",
        pixel_size=1.0,
        voltage=300.0,
        defocus_u=15000.0,
        defocus_v=15000.0,
        astigmatism_angle=0.0,
    )
    assert config.spherical_aberration == 2.7
    assert config.amplitude_contrast_ratio == 0.07
    assert config.phase_shift == 0.0
    assert config.ctf_B_factor == 0.0
    assert config.chromatic_aberration == 0.0


def test_invalid_pixel_size():
    """
    Test that an error is raised for invalid pixel size.

    Verifies that a ValidationError is raised when pixel_size is negative.
    """
    err_msg = "Input should be greater than or equal to 0"
    with pytest.raises(ValidationError, match=err_msg):
        OpticsGroup(
            label="test",
            pixel_size=-1.0,
            voltage=300.0,
            defocus_u=15000.0,
            defocus_v=15000.0,
            astigmatism_angle=0.0,
        )


def test_invalid_voltage():
    """
    Test that an error is raised for invalid voltage.

    Verifies that a ValidationError is raised when voltage is negative.
    """
    err_msg = "Input should be greater than or equal to 0"
    with pytest.raises(ValidationError, match=err_msg):
        OpticsGroup(
            label="test",
            pixel_size=1.0,
            voltage=-300.0,
            defocus_u=15000.0,
            defocus_v=15000.0,
            astigmatism_angle=0.0,
        )


def test_invalid_spherical_aberration():
    """
    Test that an error is raised for invalid spherical aberration.

    Verifies that a ValidationError is raised when spherical_aberration is negative.
    """
    err_msg = "Input should be greater than or equal to 0"
    with pytest.raises(ValidationError, match=err_msg):
        OpticsGroup(
            label="test",
            pixel_size=1.0,
            voltage=300.0,
            spherical_aberration=-2.7,
            defocus_u=15000.0,
            defocus_v=15000.0,
            astigmatism_angle=0.0,
        )


def test_invalid_amplitude_contrast_ratio():
    """
    Test that an error is raised for invalid amplitude contrast ratio.

    Verifies that a ValidationError is raised when amplitude_contrast_ratio is
    greater than 1 or less than 0, as it must be within the range [0, 1].
    """
    err_msg = "Input should be less than or equal to 1"
    with pytest.raises(ValidationError, match=err_msg):
        OpticsGroup(
            label="test",
            pixel_size=1.0,
            voltage=300.0,
            amplitude_contrast_ratio=1.1,
            defocus_u=15000.0,
            defocus_v=15000.0,
            astigmatism_angle=0.0,
        )

    err_msg = "Input should be greater than or equal to 0"
    with pytest.raises(ValidationError, match=err_msg):
        OpticsGroup(
            label="test",
            pixel_size=1.0,
            voltage=300.0,
            amplitude_contrast_ratio=-0.1,
            defocus_u=15000.0,
            defocus_v=15000.0,
            astigmatism_angle=0.0,
        )


def test_invalid_ctf_B_factor():
    """
    Test that an error is raised for invalid CTF B-factor.

    Verifies that a ValidationError is raised when ctf_B_factor is negative.
    """
    err_msg = "Input should be greater than or equal to 0"
    with pytest.raises(ValidationError, match=err_msg):
        OpticsGroup(
            label="test",
            pixel_size=1.0,
            voltage=300.0,
            ctf_B_factor=-0.1,
            defocus_u=15000.0,
            defocus_v=15000.0,
            astigmatism_angle=0.0,
        )


def test_invalid_chromatic_aberration():
    """
    Test that an error is raised for invalid chromatic aberration.

    Verifies that a ValidationError is raised when chromatic_aberration is negative.
    """
    err_msg = "Input should be greater than or equal to 0"
    with pytest.raises(ValidationError, match=err_msg):
        OpticsGroup(
            label="test",
            pixel_size=1.0,
            voltage=300.0,
            defocus_u=15000.0,
            defocus_v=15000.0,
            astigmatism_angle=0.0,
            chromatic_aberration=-0.1,
        )

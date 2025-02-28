import pytest
from pydantic import ValidationError

from leopard_em.pydantic_models.optics_group import OpticsGroup


def test_default_values():
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

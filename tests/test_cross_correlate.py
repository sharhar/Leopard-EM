"""Tests for the cross-correlation functions."""

import numpy as np
import pytest
import torch
from scipy.signal import fftconvolve
from skimage import data

from tt2dtm.utils.cross_correlation_core import cross_correlate


def fftcorrelate(in1, in2, **kwargs):
    """Wrapper function for np.fft.fftconvolve to run correlation."""
    return fftconvolve(in1, in2[::-1, ::-1], **kwargs)


#########################################
### Fixtures for generating test data ###
#########################################


@pytest.fixture
def skimage_camera_image() -> torch.Tensor:
    """Get the camera image from skimage."""
    img = data.camera().astype(np.float32)
    img = (img - img.mean()) / img.std()
    img = torch.tensor(img)

    return img


@pytest.fixture
def skimage_moon_image() -> torch.Tensor:
    """Get the moon image from skimage."""
    img = data.moon().astype(np.float32)
    img = (img - img.mean()) / img.std()
    img = torch.tensor(img)

    return img


@pytest.fixture
def skimage_camera_template() -> torch.Tensor:
    """Get the camera template from skimage."""
    img = data.camera().astype(np.float32)
    template = img[100:200, 100:200]
    template = (template - template.mean()) / template.std()
    template = torch.tensor(template)

    return template


@pytest.fixture
def skimage_moon_template() -> torch.Tensor:
    """Get the moon template from skimage."""
    img = data.moon().astype(np.float32)
    template = img[100:200, 100:200]
    template = (template - template.mean()) / template.std()
    template = torch.tensor(template)

    return template


######################################################
### Testing helper functions for cross-correlation ###
######################################################


def test_handle_correlation_mode():
    pass


def test_handle_template_padding_dft():
    pass


######################################################
### Testing cross-correlate function against scipy ###
######################################################


@pytest.mark.parametrize(
    "image_fixture,template_fixture",
    [
        ("skimage_camera_image", "skimage_camera_template"),
        ("skimage_moon_image", "skimage_moon_template"),
    ],
)
def test_cross_correlate(image_fixture, template_fixture, request):
    """Test cross-correlation function against scipy."""
    image = request.getfixturevalue(image_fixture)
    template = request.getfixturevalue(template_fixture)

    # NOTE: These more lenient tolerances are necessary because fft
    # implementations vary slightly between packages.
    atol = 5e-3
    rtol = 1e-4

    result = cross_correlate(image, template, mode="valid")
    expected = fftcorrelate(image.numpy(), template.numpy(), mode="valid")

    assert torch.allclose(result, torch.tensor(expected), atol=atol, rtol=rtol)

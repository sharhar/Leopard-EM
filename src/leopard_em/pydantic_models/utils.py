"""Utility functions shared between pydantic models."""

from typing import Any

import torch
from torch_fourier_filter.ctf import calculate_ctf_2d

from .optics_group import OpticsGroup
from .particle_stack import ParticleStack


def preprocess_image(
    image_rfft: torch.Tensor,
    cumulative_fourier_filters: torch.Tensor,
    bandpass_filter: torch.Tensor,
) -> torch.Tensor:
    """Preprocesses and normalizes the image based on the given filters.

    Parameters
    ----------
    image_rfft : torch.Tensor
        The real Fourier-transformed image (unshifted).
    cumulative_fourier_filters : torch.Tensor
        The cumulative Fourier filters. Multiplication of the whitening filter, phase
        randomization filter, bandpass filter, and arbitrary curve filter.
    bandpass_filter : torch.Tensor
        The bandpass filter used for the image. Used for dimensionality normalization.

    Returns
    -------
    torch.Tensor
        Preprocessed and normalized image in Fourier space
    """
    image_rfft = image_rfft * cumulative_fourier_filters

    # Normalize the image after filtering
    squared_image_rfft = torch.abs(image_rfft) ** 2
    squared_sum = torch.sum(squared_image_rfft, dim=(-2, -1), keepdim=True)
    squared_sum += torch.sum(
        squared_image_rfft[..., :, 1:-1], dim=(-2, -1), keepdim=True
    )
    image_rfft /= torch.sqrt(squared_sum)

    # NOTE: For two Gaussian random variables in d-dimensional space --  A and B --
    # each with mean 0 and variance 1 their correlation will have on average a
    # variance of d.
    # NOTE: Since we have the variance of the image and template projections each at
    # 1, we need to multiply the image by the square root of the number of pixels
    # so the cross-correlograms have a variance of 1 and not d.
    # NOTE: When applying the Fourier filters to the image and template, any
    # elements that get set to zero effectively reduce the dimensionality of our
    # cross-correlation. Therefore, instead of multiplying by the number of pixels,
    # we need to multiply tby the effective number of pixels that are non-zero.
    # Below, we calculate the dimensionality of our cross-correlation and divide
    # by the square root of that number to normalize the image.
    dimensionality = bandpass_filter.sum() + bandpass_filter[:, 1:-1].sum()
    image_rfft *= dimensionality**0.5

    return image_rfft


def calculate_ctf_filter_stack(
    template_shape: tuple[int, int],
    optics_group: OpticsGroup,
    defocus_offsets: torch.Tensor,  # in Angstrom, relative
    pixel_size_offsets: torch.Tensor,  # in Angstrom, relative
) -> torch.Tensor:
    """Calculate searched CTF filter values for a given shape and optics group.

    Parameters
    ----------
    template_shape : tuple[int, int]
        Desired output shape for the filter, in real space.
    optics_group : OpticsGroup
        OpticsGroup object containing the optics defining the CTF parameters.
    defocus_offsets : torch.Tensor
        Tensor of defocus offsets to search over, in Angstroms.
    pixel_size_offsets : torch.Tensor
        Tensor of pixel size offsets to search over, in Angstroms.

    Returns
    -------
    torch.Tensor
        Tensor of CTF filter values for the specified shape and optics group. Will have
        shape (num_pixel_sizes, num_defocus_offsets, h, w // 2 + 1)
    """
    return calculate_ctf_filter_stack_full_args(
        template_shape,
        optics_group.defocus_u,
        optics_group.defocus_v,
        defocus_offsets,
        pixel_size_offsets,
        astigmatism_angle=optics_group.astigmatism_angle,
        voltage=optics_group.voltage,
        spherical_aberration=optics_group.spherical_aberration,
        amplitude_contrast_ratio=optics_group.amplitude_contrast_ratio,
        ctf_B_factor=optics_group.ctf_B_factor,
        phase_shift=optics_group.phase_shift,
        pixel_size=optics_group.pixel_size,
    )


def calculate_ctf_filter_stack_full_args(
    template_shape: tuple[int, int],
    defocus_u: float,  # in Angstrom
    defocus_v: float,  # in Angstrom
    defocus_offsets: torch.Tensor,  # in Angstrom, relative
    pixel_size_offsets: torch.Tensor,  # in Angstrom, relative
    **kwargs: Any,
) -> torch.Tensor:
    """Calculate a CTF filter stack for a given set of parameters and search offsets.

    Parameters
    ----------
    template_shape : tuple[int, int]
        Desired output shape for the filter, in real space.
    defocus_u : float
        Defocus along the major axis, in Angstroms.
    defocus_v : float
        Defocus along the minor axis, in Angstroms.
    defocus_offsets : torch.Tensor
        Tensor of defocus offsets to search over, in Angstroms.
    pixel_size_offsets : torch.Tensor
        Tensor of pixel size offsets to search over, in Angstroms.
    **kwargs
        Additional keyword to pass to the calculate_ctf_2d function.

    Returns
    -------
    torch.Tensor
        Tensor of CTF filter values for the specified shape and parameters. Will have
        shape (num_pixel_sizes, num_defocus_offsets, h, w // 2 + 1)

    # Raises
    # ------
    # ValueError
    #     If not all the required parameters are passed as additional keyword arguments.
    """
    # Calculate the defocus values + offsets in terms of Angstrom
    defocus = defocus_offsets + ((defocus_u + defocus_v) / 2)
    astigmatism = abs(defocus_u - defocus_v) / 2

    # The different Cs values to search over as another dimension
    cs_values = get_cs_range(
        pixel_size=kwargs["pixel_size"],
        pixel_size_offsets=pixel_size_offsets,
        cs=kwargs["spherical_aberration"],
    )

    # Ensure defocus and astigmatism have a batch dimension so Cs and defocus can be
    # interleaved correctly
    if defocus.dim() == 1:
        defocus = defocus.unsqueeze(0)

    # Loop over spherical aberrations one at a time and collect results
    ctf_list = []
    for cs_val in cs_values:
        tmp = calculate_ctf_2d(
            defocus=defocus * 1e-4,  # Convert to um from Angstrom
            astigmatism=astigmatism * 1e-4,  # Convert to um from Angstrom
            astigmatism_angle=kwargs["astigmatism_angle"],
            voltage=kwargs["voltage"],
            spherical_aberration=cs_val,
            amplitude_contrast=kwargs["amplitude_contrast_ratio"],
            b_factor=kwargs["ctf_B_factor"],
            phase_shift=kwargs["phase_shift"],
            pixel_size=kwargs["pixel_size"],
            image_shape=template_shape,
            rfft=True,
            fftshift=False,
        )
        ctf_list.append(tmp)

    ctf = torch.stack(ctf_list, dim=0)

    return ctf


def get_cs_range(
    pixel_size: float,
    pixel_size_offsets: torch.Tensor,
    cs: float = 2.7,
) -> torch.Tensor:
    """Get the Cs values for a  range of pixel sizes.

    Parameters
    ----------
    pixel_size : float
        The nominal pixel size.
    pixel_size_offsets : torch.Tensor
        The pixel size offsets.
    cs : float, optional
        The Cs (spherical aberration) value, by default 2.7.

    Returns
    -------
    torch.Tensor
        The Cs values for the range of pixel sizes.
    """
    pixel_sizes = pixel_size + pixel_size_offsets
    cs_values = cs / torch.pow(pixel_sizes / pixel_size, 4)
    return cs_values


def cs_to_pixel_size(
    cs_vals: torch.Tensor,
    nominal_pixel_size: float,
    nominal_cs: float = 2.7,
) -> torch.Tensor:
    """Convert Cs values to pixel sizes.

    Parameters
    ----------
    cs_vals : torch.Tensor
        The Cs (spherical aberration) values.
    nominal_pixel_size : float
        The nominal pixel size.
    nominal_cs : float, optional
        The nominal Cs value, by default 2.7.

    Returns
    -------
    torch.Tensor
        The pixel sizes.
    """
    pixel_size = torch.pow(nominal_cs / cs_vals, 0.25) * nominal_pixel_size
    return pixel_size


def volume_to_rfft_fourier_slice(volume: torch.Tensor) -> torch.Tensor:
    """Prepares a 3D volume for Fourier slice extraction.

    Parameters
    ----------
    volume : torch.Tensor
        The input volume.

    Returns
    -------
    torch.Tensor
        The prepared volume in Fourier space ready for slice extraction.
    """
    assert volume.dim() == 3, "Volume must be 3D"

    # NOTE: There is an extra FFTshift step before the RFFT since, for some reason,
    # omitting this step will cause a 180 degree phase shift on odd (i, j, k)
    # structure factors in the Fourier domain. This just requires an extra
    # IFFTshift after converting a slice back to real-space (handled already).
    volume = torch.fft.fftshift(volume, dim=(0, 1, 2))  # pylint: disable=E1102
    volume_rfft = torch.fft.rfftn(volume, dim=(0, 1, 2))  # pylint: disable=E1102
    volume_rfft = torch.fft.fftshift(volume_rfft, dim=(0, 1))  # pylint: disable=E1102

    return volume_rfft


def _setup_ctf_kwargs_from_particle_stack(
    particle_stack: ParticleStack, template_shape: tuple[int, int]
) -> dict[str, Any]:
    """Helper function for per-particle CTF kwargs.

    Parameters
    ----------
    particle_stack : ParticleStack
        The particle stack to extract the CTF parameters from.
    template_shape : tuple[int, int]
        The shape of the template to use for the CTF calculation.

    Returns
    -------
    dict[str, Any]
        A dictionary of CTF parameters to pass to the CTF calculation function.
    """
    # Keyword arguments for the CTF filter calculation call
    # NOTE: We currently enforce the parameters (other than the defocus values) are
    # all the same. This could be updated in the future...
    assert particle_stack["voltage"].nunique() == 1
    assert particle_stack["spherical_aberration"].nunique() == 1
    assert particle_stack["amplitude_contrast_ratio"].nunique() == 1
    assert particle_stack["phase_shift"].nunique() == 1
    assert particle_stack["ctf_B_factor"].nunique() == 1

    return {
        "voltage": particle_stack["voltage"][0].item(),
        "spherical_aberration": particle_stack["spherical_aberration"][0].item(),
        "amplitude_contrast_ratio": particle_stack["amplitude_contrast_ratio"][
            0
        ].item(),
        "ctf_B_factor": particle_stack["ctf_B_factor"][0].item(),
        "phase_shift": particle_stack["phase_shift"][0].item(),
        "pixel_size": particle_stack["refined_pixel_size"].mean().item(),
        "template_shape": template_shape,
    }

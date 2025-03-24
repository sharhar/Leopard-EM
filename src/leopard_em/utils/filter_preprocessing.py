"""Helper functions for the CTF filter preprocessing."""

import einops
import torch
from torch_fourier_filter.ctf import calculate_ctf_2d


def get_Cs_range(
    pixel_size: float,
    pixel_size_offsets: torch.Tensor,
    Cs: float = 2.7,
) -> torch.Tensor:
    """Get the Cs values for a  range of pixel sizes.

    Parameters
    ----------
    pixel_size : float
        The nominal pixel size.
    pixel_size_offsets : torch.Tensor
        The pixel size offsets.
    Cs : float, optional
        The Cs value, by default 2.7.

    Returns
    -------
    torch.Tensor
        The Cs values for the range of pixel sizes.
    """
    pixel_sizes = pixel_size + pixel_size_offsets
    Cs_values = Cs / torch.pow(pixel_sizes / pixel_size, 4)
    return Cs_values


def Cs_to_pixel_size(
    Cs_vals: torch.Tensor,
    nominal_pixel_size: float,
    nominal_Cs: float = 2.7,
) -> torch.Tensor:
    """Convert Cs values to pixel sizes.

    Parameters
    ----------
    Cs_vals : torch.Tensor
        The Cs values.
    nominal_pixel_size : float
        The nominal pixel size.
    nominal_Cs : float, optional
        The nominal Cs value, by default 2.7.

    Returns
    -------
    torch.Tensor
        The pixel sizes.
    """
    pixel_size = torch.pow(nominal_Cs / Cs_vals, 0.25) * nominal_pixel_size
    return pixel_size


def calculate_ctf_filter_stack(
    pixel_size: float,
    template_shape: tuple[int, int],
    defocus_u: float,  # in um, *NOT* Angstrom
    defocus_v: float,  # in um, *NOT* Angstrom
    defocus_offsets: torch.Tensor,  # in um, *NOT* Angstrom
    astigmatism_angle: float,
    pixel_size_offsets: torch.Tensor,  # in Angstrom
    amplitude_contrast_ratio: float = 0.07,
    spherical_aberration: float = 2.7,
    phase_shift: float = 0.0,
    voltage: float = 300.0,
    ctf_B_factor: float = 60.0,
    rfft: bool = True,
    fftshift: bool = False,
) -> torch.Tensor:
    """Calculate stack (batch) of CTF filters to apply to projections during 2DTM.

    NOTE: While the defocus in the YAML and other Pydantic models is in Angstrom, this
    function expects defocus in um! Beware to convert accordingly.

    Parameters
    ----------
    pixel_size : float
        The pixel size of the images in Angstrom.
    template_shape : tuple[int, int]
        Desired output shape for the filter, in real space. Note that RFFT here cause
        resulting filter to *not* be this passed tuple, but (h, w // 2 + 1).
    defocus_u : float
        The defocus in the u direction, in um.
    defocus_v : float
        The defocus in the v direction, in um.
    defocus_offsets : torch.Tensor
        The offsets to apply to the defocus, in um.
    pixel_size_offsets : torch.Tensor
        The offsets to apply to the pixel size, in um.
    astigmatism_angle : float
        The angle of defocus astigmatism, in degrees.
    amplitude_contrast_ratio : float, optional
        The amplitude contrast ratio, by default 0.07.
    spherical_aberration : float, optional
        The spherical aberration constant, in mm, by default 2.7.
    phase_shift : float, optional
        The phase shift constant, by default 0.0.
    voltage : float, optional
        The voltage of the microscope, in kV, by default 300.0.
    ctf_B_factor : float, optional
        The additional B factor for the CTF, by default 60.0.
    rfft : bool, optional
        Whether to use RFFT, by default True.
    fftshift : bool, optional
        Whether to shift the FFT, by default False.
    """
    defocus = (defocus_u + defocus_v) / 2 + defocus_offsets
    astigmatism = abs(defocus_u - defocus_v) / 2

    Cs_vals = get_Cs_range(
        pixel_size=pixel_size,
        pixel_size_offsets=pixel_size_offsets,
        Cs=spherical_aberration,
    )

    # Loop over Cs_vals one at a time and collect results
    ctf_list = []
    for cs_val in Cs_vals:
        ctf_single = calculate_ctf_2d(
            defocus=defocus,
            astigmatism=astigmatism,
            astigmatism_angle=astigmatism_angle,
            voltage=voltage,
            spherical_aberration=cs_val,
            amplitude_contrast=amplitude_contrast_ratio,
            b_factor=ctf_B_factor,
            phase_shift=phase_shift,
            pixel_size=pixel_size,
            image_shape=template_shape,
            rfft=rfft,
            fftshift=fftshift,
        )

        # Ensure we have the defocus dimension
        if ctf_single.ndim == 2:  # (nx, ny)
            ctf_single = einops.rearrange(ctf_single, "nx ny -> 1 nx ny")

        ctf_list.append(ctf_single)

    # Stack along the Cs dimension
    ctf = torch.stack(ctf_list, dim=0)  # (nCs, n_defoc, nx, ny)
    # The CTF will have a shape of (n_Cs n_defoc, nx, ny)
    # These will catch any potential errors
    return ctf

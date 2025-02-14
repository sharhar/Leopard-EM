"""Utility functions for dealing with particle stack-like data."""

from typing import Literal

import numpy as np
import roma
import torch
from torch_fourier_slice import extract_central_slices_rfft_3d

from tt2dtm.backend import normalize_template_projection
from tt2dtm.utils.cross_correlation import handle_correlation_mode

TORCH_TO_NUMPY_PADDING_MODE = {
    "constant": "constant",
    "reflect": "reflect",
    "replicate": "edge",
}


def get_cropped_image_regions(
    image: torch.Tensor | np.ndarray,
    pos_y: torch.Tensor | np.ndarray,
    pos_x: torch.Tensor | np.ndarray,
    box_size: int | tuple[int, int],
    pos_reference: Literal["center", "top-left"] = "center",
    handle_bounds: Literal["pad", "error"] = "pad",
    padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
    padding_value: float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Extracts regions from an image into a stack of cropped images.

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        The input image from which to extract the regions.
    pos_y : torch.Tensor | np.ndarray
        The y positions of the regions to extract. Type must mach `image`
    pos_x : torch.Tensor | np.ndarray
        The x positions of the regions to extract. Type must mach `image`
    box_size : int | tuple[int, int]
        The size of the box to extract. If an integer is passed, the box will be square.
    pos_reference : Literal["center", "top-left"], optional
        The reference point for the positions, by default "center". If "center", the
        boxes extracted will be image[y - box_size // 2 : y + box_size // 2, ...]. If
        "top-left", the boxes will be image[y : y + box_size, ...].
    handle_bounds : Literal["pad", "clip", "error"], optional
        How to handle the bounds of the image, by default "pad". If "pad", the image
        will be padded with the padding value based on the padding mode. If "error", an
        error will be raised if any region exceeds the image bounds. Note clipping is
        not supported since returned stack may have inhomogeneous sizes.
    padding_mode : Literal["constant", "reflect", "replicate"], optional
        The padding mode to use when padding the image, by default "constant".
        "constant" pads with the value `padding_value`, "reflect" pads with the
        reflection of the image at the edge, and "replicate" pads with the last pixel
        of the image. These match the modes available in `torch.nn.functional.pad`.
    padding_value : float, optional
        The value to use for padding when `padding_mode` is "constant", by default 0.0.

    Returns
    -------
    torch.Tensor | np.ndarray
        The stack of cropped images extracted from the input image. Type will match the
        input image type.
    """
    if isinstance(box_size, int):
        box_size = (box_size, box_size)

    if pos_reference == "center":
        pos_y = pos_y - box_size[0] // 2
        pos_x = pos_x - box_size[1] // 2
    elif pos_reference == "top-left":
        pass
    else:
        raise ValueError(f"Unknown pos_reference: {pos_reference}")

    if isinstance(image, torch.Tensor):
        return _get_cropped_image_regions_torch(
            image=image,
            pos_y=pos_y,
            pos_x=pos_x,
            box_size=box_size,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode,
            padding_value=padding_value,
        )

    if isinstance(image, np.ndarray):
        padding_mode_np = TORCH_TO_NUMPY_PADDING_MODE[padding_mode]
        return _get_cropped_image_regions_numpy(
            image=image,
            pos_y=pos_y,
            pos_x=pos_x,
            box_size=box_size,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode_np,
            padding_value=padding_value,
        )

    raise ValueError(f"Unknown image type: {type(image)}")


def _get_cropped_image_regions_numpy(
    image: np.ndarray,
    pos_y: np.ndarray,
    pos_x: np.ndarray,
    box_size: tuple[int, int],
    handle_bounds: Literal["pad", "error"],
    padding_mode: str,
    padding_value: float,
) -> np.ndarray:
    """Helper function for extracting regions from a numpy array.

    NOTE: this function assumes that the position reference is the top-left corner.
    Reference value is handled by the user-exposed 'get_cropped_image_regions' function.
    """
    if handle_bounds == "pad":
        bs1 = box_size[1] // 2
        bs0 = box_size[0] // 2
        image = np.pad(
            image,
            pad_width=((bs0, bs0), (bs1, bs1)),
            mode=padding_mode,
            constant_values=padding_value,
        )

    cropped_images = np.stack(
        [image[y : y + box_size[0], x : x + box_size[1]] for y, x in zip(pos_y, pos_x)]
    )

    return cropped_images


def _get_cropped_image_regions_torch(
    image: torch.Tensor,
    pos_y: torch.Tensor,
    pos_x: torch.Tensor,
    box_size: tuple[int, int],
    handle_bounds: Literal["pad", "error"],
    padding_mode: Literal["constant", "reflect", "replicate"],
    padding_value: float,
) -> torch.Tensor:
    """Helper function for extracting regions from a torch tensor.

    NOTE: this function assumes that the position reference is the top-left corner.
    Reference value is handled by the user-exposed 'get_cropped_image_regions' function.
    """
    if handle_bounds == "pad":
        bs1 = box_size[1] // 2
        bs0 = box_size[0] // 2
        image = torch.nn.functional.pad(
            image,
            pad=(bs1, bs1, bs0, bs0),
            mode=padding_mode,
            value=padding_value,
        )

    cropped_images = torch.stack(
        [image[y : y + box_size[0], x : x + box_size[1]] for y, x in zip(pos_y, pos_x)]
    )

    return cropped_images


def cross_correlate_particle_stack(
    particle_stack_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    projective_filters: torch.Tensor,  # usually (ctf * whitening)
    mode: Literal["valid", "same"] = "valid",
    batch_size: int = 1024,
) -> torch.Tensor:
    """Cross-correlate a stack of particle images against a template.

    Parameters
    ----------
    particle_stack_dft : torch.Tensor
        The stack of particle real-Fourier transformed and un-fftshifted images.
        Shape of (N, H, W).
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted.
    euler_angles : torch.Tensor
        The Euler angles for each particle in the stack. Shape of (3, N).
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    mode : Literal["valid", "same"], optional
        Correlation mode to use, by default "valid". If "valid", the output will be
        the valid cross-correlation of the inputs. If "same", the output will be the
        same shape as the input particle stack.
    batch_size : int, optional
        The batch size to use when processing the particle stack, by default 1024.

    Returns
    -------
    torch.Tensor
        The cross-correlation of the particle stack with the template. Shape will depend
        on the mode used.
    """
    # Helpful constants for later use
    device = particle_stack_dft.device
    num_particles, H, W = particle_stack_dft.shape
    d, h, w = template_dft.shape
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)

    print(f"num_particles: {num_particles}")
    print(f"H: {H}")
    print(f"W: {W}")
    print(f"d: {d}")
    print(f"w: {w}")
    print(f"h: {h}")

    # # Flag for same dimensional output
    # _same_dim = (H == h) and (W == w)

    if mode == "valid":
        output_shape = (num_particles, H - h + 1, W - w + 1)
    elif mode == "same":
        output_shape = (num_particles, H, W)

    out_correlation = torch.zeros(output_shape, device=device)

    # Loop over the particle stack in batches
    for i in range(0, num_particles, batch_size):
        batch_particles_dft = particle_stack_dft[i : i + batch_size]
        batch_euler_angles = euler_angles[i : i + batch_size]
        batch_projective_filters = projective_filters[i : i + batch_size]

        # Convert the Euler angles into rotation matrices
        rot_matrix = roma.euler_to_rotmat(
            "ZYZ", batch_euler_angles, degrees=True, device=device
        )

        # Extract the Fourier slice and apply the projective filters
        fourier_slice = extract_central_slices_rfft_3d(
            volume_rfft=template_dft,
            image_shape=(h,) * 3,
            rotation_matrices=rot_matrix,
        )
        fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
        fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
        fourier_slice *= -1  # flip contrast
        fourier_slice *= batch_projective_filters

        # Inverse Fourier transform and normalize the projection
        projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
        projections = torch.fft.ifftshift(projections, dim=(-2, -1))
        projections = normalize_template_projection(projections, (h, w), (H, W))

        # Padded forward FFT and cross-correlate
        projections_dft = torch.fft.rfftn(projections, dim=(-2, -1), s=(H, W))
        projections_dft = batch_particles_dft * projections_dft.conj()
        cross_correlation = torch.fft.irfftn(projections_dft, dim=(-2, -1))

        # Handle the output shape
        cross_correlation = handle_correlation_mode(
            cross_correlation, output_shape, mode
        )

        out_correlation[i : i + batch_size] = cross_correlation

    return out_correlation

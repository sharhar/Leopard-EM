"""Backend functions related to correlating and refining particle stacks."""

from typing import Literal

import roma
import torch
import tqdm
from torch_fourier_filter.ctf import calculate_ctf_2d
from torch_fourier_slice import extract_central_slices_rfft_3d

from tt2dtm.backend.utils import normalize_template_projection
from tt2dtm.utils.cross_correlation import handle_correlation_mode


def core_refine_template(
    particle_stack_dft: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    euler_angles: torch.Tensor,  # (3, N)
    defocus_offsets: torch.Tensor,  # (l,)
    defocus_u: torch.Tensor,  # (N,)
    defocus_v: torch.Tensor,  # (N,)
    defocus_angle: torch.Tensor,  # (N,)
    ctf_kwargs: dict,
    projective_filters: torch.Tensor,  # (N, h, w)
    euler_angle_offsets: torch.Tensor,  # (3, k)
    batch_size: int = 1024,
    # TODO: additional arguments for cc --> z-score scaling
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core function to refine orientations and defoci of a set of particles.

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
    euler_angle_offsets : torch.Tensor
        The Euler angle offsets to apply to each particle. Shape of (3, k).
    defocus_u : torch.Tensor
        The defocus along the major axis for each particle in the stack. Shape of (N,).
    defocus_v : torch.Tensor
        The defocus along the minor for each particle in the stack. Shape of (N,).
    defocus_angle : torch.Tensor
        The defocus astigmatism angle for each particle in the stack. Shape of (N,).
        Is the same as the defocus for the micrograph the particle came from.
    defocus_offsets : torch.Tensor
        The defocus offsets to search over for each particle. Shape of (l,).
    ctf_kwargs : dict
        Keyword arguments to pass to the CTF calculation function.
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        Larger sizes will consume more memory. If -1, then the entire stack will be
        cross-correlated at once.
    """
    device = particle_stack_dft.device
    num_particles, H, W = particle_stack_dft.shape
    d, h, w = template_dft.shape
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)

    # Send other tensors to the same device
    euler_angles = euler_angles.to(device)
    defocus_u = defocus_u.to(device)
    defocus_v = defocus_v.to(device)
    defocus_angle = defocus_angle.to(device)
    defocus_offsets = torch.tensor(defocus_offsets)
    defocus_offsets = defocus_offsets.to(device)
    projective_filters = projective_filters.to(device)
    euler_angle_offsets = euler_angle_offsets.to(device)

    rot_matrix = roma.euler_to_rotmat("ZYZ", euler_angles, degrees=True, device=device)

    maximum_cross_correlation = torch.zeros(num_particles, device=device)
    best_euler_angle_offset = torch.zeros(num_particles, 3, device=device)
    best_defocus_offset = torch.zeros(num_particles, device=device)

    for i, delta_df in tqdm.tqdm(
        enumerate(defocus_offsets), total=len(defocus_offsets)
    ):
        _ = i
        # Recompute the CTF filters for each particle's absolute defocus
        defocus = (defocus_u + defocus_v) / 2 + delta_df
        astigmatism = defocus_v - delta_df
        ctf_filters = calculate_ctf_2d(
            defocus=defocus * 1e-4,  # to µm
            astigmatism=astigmatism * 1e-4,  # to µm
            astigmatism_angle=defocus_angle,
            **ctf_kwargs,
        )

        temp_projective_filters = projective_filters * ctf_filters

        # Iterate over the Euler angle offsets
        for j, delta_ea in tqdm.tqdm(
            enumerate(euler_angle_offsets), total=euler_angle_offsets.size(0)
        ):
            _ = j
            delta_rot_matrix = roma.euler_to_rotmat(
                "ZYZ", delta_ea, degrees=True, device=device
            )
            new_rot_matrix = roma.rotmat_composition((rot_matrix, delta_rot_matrix))
            new_rot_matrix = new_rot_matrix.to(torch.float32)

            cc_stack = cross_correlate_particle_stack(
                particle_stack_dft=particle_stack_dft,
                template_dft=template_dft,
                rotation_matrices=new_rot_matrix,
                projective_filters=temp_projective_filters,
                batch_size=batch_size,
            )

            # Update the best tracked statistics
            cc_stack_maximum = torch.amax(cc_stack, dim=(1, 2))

            update_mask = cc_stack_maximum > maximum_cross_correlation
            torch.where(update_mask, cc_stack_maximum, maximum_cross_correlation)
            torch.where(update_mask, delta_ea[0], best_euler_angle_offset[:, 0])
            torch.where(update_mask, delta_ea[1], best_euler_angle_offset[:, 1])
            torch.where(update_mask, delta_ea[2], best_euler_angle_offset[:, 2])
            torch.where(update_mask, delta_df, best_defocus_offset)

    return maximum_cross_correlation, best_euler_angle_offset, best_defocus_offset


def cross_correlate_particle_stack(
    particle_stack_dft: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
    projective_filters: torch.Tensor,  # (N, h, w)
    mode: Literal["valid", "same"] = "valid",
    batch_size: int = 1024,
) -> torch.Tensor:
    """Cross-correlate a stack of particle images against a template.

    Here, the argument 'particle_stack_dft' is a set of RFFT-ed particle images with
    necessary filtering already applied. The zeroth dimension corresponds to unique
    particles.

    Parameters
    ----------
    particle_stack_dft : torch.Tensor
        The stack of particle real-Fourier transformed and un-fftshifted images.
        Shape of (N, H, W).
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted.
    rotation_matrices : torch.Tensor
        The orientations of the particles to take the Fourier slices of, as a long
        list of rotation matrices. Shape of (N, 3, 3).
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    mode : Literal["valid", "same"], optional
        Correlation mode to use, by default "valid". If "valid", the output will be
        the valid cross-correlation of the inputs. If "same", the output will be the
        same shape as the input particle stack.
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        Larger sizes will consume more memory. If -1, then the entire stack will be
        cross-correlated at once.

    Returns
    -------
    torch.Tensor
        The cross-correlation of the particle stack with the template. Shape will depend
        on the mode used. If "valid", the output will be (N, H-h+1, W-w+1). If "same",
        the output will be (N, H, W).
    """
    # Helpful constants for later use
    device = particle_stack_dft.device
    num_particles, H, W = particle_stack_dft.shape
    d, h, w = template_dft.shape
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)

    if batch_size == -1:
        batch_size = num_particles

    if mode == "valid":
        output_shape = (num_particles, H - h + 1, W - w + 1)
    elif mode == "same":
        output_shape = (num_particles, H, W)

    out_correlation = torch.zeros(output_shape, device=device)

    # Loop over the particle stack in batches
    for i in range(0, num_particles, batch_size):
        batch_particles_dft = particle_stack_dft[i : i + batch_size]
        batch_rotation_matrices = rotation_matrices[i : i + batch_size]
        batch_projective_filters = projective_filters[i : i + batch_size]

        # Extract the Fourier slice and apply the projective filters
        fourier_slice = extract_central_slices_rfft_3d(
            volume_rfft=template_dft,
            image_shape=(h,) * 3,
            rotation_matrices=batch_rotation_matrices,
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

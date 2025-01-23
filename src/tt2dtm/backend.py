"""Pure PyTorch implementation of whole orientation search backend."""

import numpy as np
import roma
import torch
import torch.multiprocessing as mp
import tqdm
from torch_fourier_slice import extract_central_slices_rfft_3d

COMPILE_BACKEND = "inductor"
DEFAULT_STATISTIC_DTYPE = torch.float32

# Turn off gradient calculations by default
torch.set_grad_enabled(False)


def construct_multi_gpu_match_template_kwargs(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    H: int,
    W: int,
    h: int,
    w: int,
    projection_batch_size: int,
    devices: list[torch.device],
) -> list[dict[str, torch.Tensor]]:
    """Split orientations between requested devices.

    See the `core_match_template` function for further descriptions of the
    input parameters.

    Parameters
    ----------
    image_dft : torch.Tensor
        dft of image
    template_dft : torch.Tensor
        dft of template
    euler_angles : torch.Tensor
        euler angles to search
    projective_filters : torch.Tensor
        filters to apply to each projection
    defocus_values : torch.Tensor
        corresponding defocus values for each filter
    H : int
        height of image in real space
    W : int
        width of image in real space
    h : int
        height of projection in real space
    w : int
        width of projection in real space
    projection_batch_size : int
        number of projections to calculate at once
    devices : list[torch.device]
        list of devices to split the orientations across

    Returns
    -------
    list[dict[str, torch.Tensor]]
        List of dictionaries containing the kwargs to call the single-GPU
        function. Each element in the list corresponds to a different device,
        and all tensors in the dictionary have been allocated to that device.
    """
    num_devices = len(devices)
    num_orientations_per_device = euler_angles.shape[0] // num_devices + 1
    kwargs_per_device = []

    # Split the euler angles across devices
    euler_angles_split = euler_angles.split(num_orientations_per_device)

    for device, euler_angles_device in zip(devices, euler_angles_split):
        # Allocate all tensors to the device
        image_dft_device = image_dft.to(device)
        template_dft_device = template_dft.to(device)
        euler_angles_device = euler_angles_device.to(device)
        projective_filters_device = projective_filters.to(device)
        defocus_values_device = defocus_values.to(device)

        # Construct the kwargs dictionary
        kwargs = {
            "image_dft": image_dft_device,
            "template_dft": template_dft_device,
            "euler_angles": euler_angles_device,
            "projective_filters": projective_filters_device,
            "defocus_values": defocus_values_device,
            "H": H,
            "W": W,
            "h": h,
            "w": w,
            "projection_batch_size": projection_batch_size,
        }

        kwargs_per_device.append(kwargs)

    return kwargs_per_device


######################################################
### Helper functions called at the end of the loop ###
######################################################


def aggregate_distributed_results(
    results: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Combine the 2DTM results from multiple devices."""
    # Pass all tensors back to the CPU
    results = [
        {
            key: value.cpu()
            for key, value in result.items()
            if isinstance(value, torch.Tensor)
        }
        for result in results
    ]

    # Construct stack of MIPs and find the maximum
    mips = torch.stack([result["mip"] for result in results], dim=0)
    mip_max, mip_argmax = torch.max(mips, dim=0)

    # Construct stack of best angles and defocus values
    best_phi = torch.stack([result["best_phi"] for result in results], dim=0)
    best_phi = best_phi.gather(dim=0, index=mip_argmax[None, ...])

    best_theta = torch.stack([result["best_theta"] for result in results], dim=0)
    best_theta = best_theta.gather(dim=0, index=mip_argmax[None, ...])

    best_psi = torch.stack([result["best_psi"] for result in results], dim=0)
    best_psi = best_psi.gather(dim=0, index=mip_argmax[None, ...])

    best_defocus = torch.stack([result["best_defocus"] for result in results], dim=0)
    best_defocus = best_defocus.gather(dim=0, index=mip_argmax[None, ...])

    # Sum the sums and squared sums of the cross-correlation values
    correlation_sum = torch.stack(
        [result["correlation_sum"] for result in results], dim=0
    ).sum(dim=0)
    correlation_squared_sum = torch.stack(
        [result["correlation_squared_sum"] for result in results], dim=0
    ).sum(dim=0)

    total_projections = sum(result["total_projections"] for result in results)

    return {
        "mip": mip_max,
        "best_phi": best_phi,
        "best_theta": best_theta,
        "best_psi": best_psi,
        "best_defocus": best_defocus,
        "correlation_sum": correlation_sum,
        "correlation_squared_sum": correlation_squared_sum,
        "total_projections": total_projections,
    }


def sum_and_squared_sum_to_variance(
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    total_projections: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper function to calculate total variance from mean and squared sum."""
    raise NotImplementedError("Not implemented yet.")


def scale_mip(
    mip: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    total_correlation_positions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale the MIP by the mean and variance of the correlation values.

    Parameters
    ----------
    mip : torch.Tensor
        MIP of the correlation values.
    correlation_sum : torch.Tensor
        Sum of the correlation values.
    correlation_squared_sum : torch.Tensor
        Sum of the squared correlation values.
    total_correlation_positions : int
        Total number cross-correlograms calculated.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the MIP and scaled MIP
    """
    num_pixels = torch.tensor(mip.shape[0] * mip.shape[1])
    mip_mult = mip * num_pixels

    correlation_sum = correlation_sum / total_correlation_positions
    correlation_squared_sum = correlation_squared_sum / total_correlation_positions - (
        correlation_sum**2
    )
    correlation_squared_sum = torch.sqrt(
        torch.clamp(correlation_squared_sum, min=0)
    ) * torch.sqrt(num_pixels)
    correlation_sum = correlation_sum * torch.sqrt(num_pixels)

    # Calculate normalized MIP
    mip_normalized = mip_mult - correlation_sum
    mip_normalized = torch.where(
        correlation_squared_sum == 0,
        torch.zeros_like(mip_normalized),
        mip_normalized / correlation_squared_sum,
    )
    return mip_mult, mip_normalized


###########################################################################
### Helper functions called during the loop (passed into torch.compile) ###
###########################################################################


def normalize_template_projection(
    projections: torch.Tensor,  # shape (batch, h, w)
    small_shape: tuple[int, int],  # (h, w)
    large_shape: tuple[int, int],  # (H, W)
) -> torch.Tensor:
    """Subtract mean of edge values and set variance to 1 (in large shape).

    Parameters
    ----------
    projections : torch.Tensor
        Real-space projections of the template (in small space).
    small_shape : tuple[int, int]
        Shape of the template.
    large_shape : tuple[int, int]
        Shape of the image (in large space).

    Returns
    -------
    torch.Tensor
        Edge-mean subtracted projections, still in small space, but normalized
        so variance of zero-padded projection is 1.
    """
    # Extract edges while preserving batch dimensions
    top_edge = projections[..., 0, :]  # shape: (..., W)
    bottom_edge = projections[..., -1, :]  # shape: (..., W)
    left_edge = projections[..., :, 0]  # shape: (..., H)
    right_edge = projections[..., :, -1]  # shape: (..., H)

    # Stack all edges along a new dimension
    edge_pixels = torch.stack(
        [
            top_edge.flatten(-1),  # flatten last dim (W) to combine with H
            bottom_edge.flatten(-1),
            left_edge.flatten(-1),
            right_edge.flatten(-1),
        ],
        dim=-1,
    )  # shape: (..., num_edge_pixels, 4)

    edge_mean = torch.mean(edge_pixels, dim=(-2, -1), keepdim=True)
    projections -= edge_mean

    relative_size = float(small_shape[0] * small_shape[1])
    relative_size /= float(large_shape[0] * large_shape[1])

    squared_sum = torch.sum(projections**2, dim=(-1, -2), keepdim=True)
    variance = squared_sum * relative_size - (edge_mean * relative_size) ** 2

    return projections / torch.sqrt(variance)


def do_iteration_statistics_updates(
    cross_correlation: torch.Tensor,
    euler_angles: torch.Tensor,
    defocus_values: torch.Tensor,
    mip: torch.Tensor,
    best_phi: torch.Tensor,
    best_theta: torch.Tensor,
    best_psi: torch.Tensor,
    best_defocus: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    H: int,
    W: int,
) -> None:
    """Helper function for updating maxima and tracked statistics.

    NOTE: the batch dimensions are effectively unraveled since taking the
    maximum over a single batch dimensions is much faster than
    multi-dimensional maxima.

    NOTE: Updating the maxima was found to be fastest and least memory
    impactful when using torch.where directly. Other methods tested were
    boolean masking and torch.where with tuples of tensor indexes.

    Parameters
    ----------
    cross_correlation : torch.Tensor
        Cross-correlation values for the current iteration. Has either shape
        (batch, H, W) or (defocus, orientations, H, W).
    euler_angles : torch.Tensor
        Euler angles for the current iteration. Has shape (orientations, 3).
    defocus_values : torch.Tensor
        Defocus values for the current iteration. Has shape (defocus,).
    mip : torch.Tensor
        Maximum intensity projection of the cross-correlation values.
    best_phi : torch.Tensor
        Best phi angle for each pixel.
    best_theta : torch.Tensor
        Best theta angle for each pixel.
    best_psi : torch.Tensor
        Best psi angle for each pixel.
    best_defocus : torch.Tensor
        Best defocus value for each pixel.
    correlation_sum : torch.Tensor
        Sum of cross-correlation values for each pixel.
    correlation_squared_sum : torch.Tensor
        Sum of squared cross-correlation values for each pixel.
    H : int
        Height of the cross-correlation values.
    W : int
        Width of the cross-correlation values.
    """
    max_values, max_indices = torch.max(cross_correlation.view(-1, H, W), dim=0)
    max_defocus_idx = max_indices // euler_angles.shape[0]
    max_orientation_idx = max_indices % euler_angles.shape[0]

    # using torch.where directly
    update_mask = max_values > mip
    mip = torch.where(update_mask, max_values, mip)
    best_phi = torch.where(update_mask, euler_angles[max_orientation_idx, 0], best_phi)
    best_theta = torch.where(
        update_mask, euler_angles[max_orientation_idx, 1], best_theta
    )
    best_psi = torch.where(update_mask, euler_angles[max_orientation_idx, 2], best_psi)
    best_defocus = torch.where(
        update_mask, defocus_values[max_defocus_idx], best_defocus
    )

    correlation_sum += cross_correlation.view(-1, H, W).sum(dim=0)
    correlation_squared_sum += (cross_correlation.view(-1, H, W) ** 2).sum(dim=0)


#################################
### Compiled helper functions ###
#################################

normalize_template_projection_compiled = torch.compile(
    normalize_template_projection, backend=COMPILE_BACKEND
)
do_iteration_statistics_updates_compiled = torch.compile(
    do_iteration_statistics_updates, backend=COMPILE_BACKEND
)


def core_match_template(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,  # already fftshifted
    ctf_filters: torch.Tensor,
    whitening_filter_template: torch.Tensor,
    defocus_values: torch.Tensor,
    euler_angles: torch.Tensor,
    image_shape: tuple[int, int],
    template_shape: tuple[int, int],
    device: torch.device | list[torch.device],
    projection_batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    """Core function for performing the whole-orientation search.

    With the RFFT, the last dimension (fastest dimension) is half the width
    of the input, hence the shape of W // 2 + 1 instead of W for some of the
    input parameters.

    Parameters
    ----------
    image_dft : torch.Tensor
        Real-fourier transform (RFFT) of the image with large image filters
        already applied. Has shape (H, W // 2 + 1).
    template_dft : torch.Tensor
        Real-fourier transform (RFFT) of the template volume to take Fourier
        slices from. Has shape (l, h, w // 2 + 1). where l is the number of
        slices.
    ctf_filters : torch.Tensor
        Stack of CTF filters at different defocus values to use in the search.
        Has shape (defocus_batch, h, w // 2 + 1).
    whitening_filter_template : torch.Tensor
        Whitening filter for the template volume. Has shape (h, w // 2 + 1).
        Gets multiplied with the ctf filters to create a filter stack.
    euler_angles : torch.Tensor
        Euler angles (in 'zyz' convention) to search over. Has shape
        (orientations, 3).
    defocus_values : torch.Tensor
        What defoucs values correspond with the CTF filters. Has shape
        (defocus_batch,).
    image_shape : tuple[int, int]
        Shape of the image (H, W), in pixels.
    template_shape : tuple[int, int]
        Shape of the template (h, w), in pixels.
    device : torch.device | list[torch.device]
        Device or devices to split computation across.
    projection_batch_size : int, optional
        Number of projections to calculate at once, on each device

    Returns
    -------
    TODO
    """
    #####################################
    ### Constants for later reference ###
    #####################################

    h, w = template_shape
    H, W = image_shape

    # filter_batch_size = ctf_filters.shape[0]
    # batch_size = projection_batch_size * filter_batch_size

    #########################################
    ### Split orientations across devices ###
    #########################################

    if isinstance(device, torch.device):
        device = [device]

    kwargs_per_device = construct_multi_gpu_match_template_kwargs(
        image_dft=image_dft,
        template_dft=template_dft,
        euler_angles=euler_angles,
        projective_filters=ctf_filters,
        defocus_values=defocus_values,
        H=H,
        W=W,
        h=h,
        w=w,
        projection_batch_size=projection_batch_size,
        devices=device,
    )

    ##################################################
    ### Initialize and start multiprocessing queue ###
    ##################################################
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue()
    processes = []

    try:
        # Start processes
        for kwargs in kwargs_per_device:
            p = mp.Process(target=_core_match_template_single_gpu, kwargs=kwargs)
            p.start()
            processes.append(p)

        # Collect results
        partial_results = []
        for _ in range(len(kwargs_per_device)):
            result = queue.get()  # Blocking call, waits for result
            if result is None:
                raise RuntimeError("GPU process reported failure")

            # Convert numpy arrays back to torch tensors
            result = {
                k: torch.from_numpy(v)
                for k, v in result.items()
                if isinstance(v, np.ndarray)
            }
            partial_results.append(result)

    except Exception as e:
        print(f"Error in multi-GPU processing: {e!s}")
        raise

    finally:
        # Cleanup
        for p in processes:
            p.terminate() if p.is_alive() else None
            p.join()  # Ensure all processes have finished

    # Verify we got all results
    if len(partial_results) != len(kwargs_per_device):
        raise RuntimeError("Not all GPU processes returned results")

    # Get the aggregated results
    aggregated_results = aggregate_distributed_results(partial_results)
    mip = aggregated_results["mip"]
    best_phi = aggregated_results["best_phi"]
    best_theta = aggregated_results["best_theta"]
    best_psi = aggregated_results["best_psi"]
    best_defocus = aggregated_results["best_defocus"]
    correlation_sum = aggregated_results["correlation_sum"]
    correlation_squared_sum = aggregated_results["correlation_squared_sum"]
    total_projections = aggregated_results["total_projections"]

    # Find the correlation mean and variance
    # mean, variance = sum_and_squared_sum_to_variance(
    #    correlation_sum, correlation_squared_sum, total_projections
    # )

    # scaled_mip = (mip - mean) / torch.sqrt(variance)

    mip_mult, scaled_mip = scale_mip(
        mip=mip,
        correlation_sum=correlation_sum,
        correlation_squared_sum=correlation_squared_sum,
        total_correlation_positions=total_projections,
    )

    return {
        "mip": mip_mult,
        "scaled_mip": scaled_mip,
        "best_phi": best_phi,
        "best_theta": best_theta,
        "best_psi": best_psi,
        "best_defocus": best_defocus,
        "correlation_sum": correlation_sum,
        "correlation_squared_sum": correlation_squared_sum,
        "total_projections": total_projections,
    }


def _core_match_template_single_gpu(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    H: int,
    W: int,
    h: int,
    w: int,
    projection_batch_size: int,
) -> dict[str, torch.Tensor]:
    """Single-GPU call for template matching.

    NOTE: All tensors *must* be allocated on the same device. By calling the
    user-facing `core_match_template` function this is handled automatically.

    Parameters
    ----------
    image_dft : torch.Tensor
        Real-fourier transform (RFFT) of the image with large image filters
        already applied. Has shape (H, W // 2 + 1).
    template_dft : torch.Tensor
        Real-fourier transform (RFFT) of the template volume to take Fourier
        slices from. Has shape (l, h, w // 2 + 1). where l is the number of
        slices.
    euler_angles : torch.Tensor
        Euler angles (in 'zyz' convention) to search over. Has shape
        (orientations // n_devices, 3). This has already been split (e.g.
        4 devices has shape (orientations // 4, 3).
    projective_filters : torch.Tensor
        Multiplied 'ctf_filters' with 'whitening_filter_template'. Has shape
        (defocus_batch, h, w // 2 + 1). Is RFFT and not fftshifted.
    defocus_values : torch.Tensor
        What defoucs values correspond with the CTF filters. Has shape
        (defocus_batch,).
    H : int
        Height of the image in real space.
    W : int
        Width of the image in real space.
    h : int
        Height of the template in real space.
    w : int
        Width of the template in real space.
    projection_batch_size : int
        The number of projections to calculate the correlation for at once.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing the following (keys, value) pairs:
          - 'mip': Maximum intensity projection of the cross-correlation.
          - 'best_phi': Best phi angle for each pixel.
          - 'best_theta': Best theta angle for each pixel.
          - 'best_psi': Best psi angle for each pixel.
          - 'best_defocus': Best defocus value for each pixel.
          - 'correlation_sum': Sum of cross-correlation values for each pixel.
          - 'correlation_squared_sum': Sum of squared cross-correlation values
            for each pixel.
          - 'total_projections': Total number of projections calculated.

        These values will be contained on the same device as the input tensors.
    """
    device = image_dft.device

    ################################################
    ### Initialize the tracked output statistics ###
    ################################################

    mip = torch.full(
        size=(H, W),
        fill_value=-float("inf"),
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_phi = torch.full(
        size=(H, W),
        fill_value=-1.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_theta = torch.full(
        size=(H, W),
        fill_value=-1.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_psi = torch.full(
        size=(H, W),
        fill_value=-1.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_defocus = torch.full(
        size=(H, W),
        fill_value=0.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    correlation_sum = torch.zeros(
        size=(H, W), dtype=DEFAULT_STATISTIC_DTYPE, device=device
    )
    correlation_squared_sum = torch.zeros(
        size=(H, W), dtype=DEFAULT_STATISTIC_DTYPE, device=device
    )

    ########################################################
    ### Setup iterator object with tqdm for progress bar ###
    ########################################################

    num_batches = euler_angles.shape[0] // projection_batch_size + 1
    orientation_batch_iterator = tqdm.tqdm(
        range(num_batches),
        desc=f"search on dev. {device.index}",
        leave=True,
        total=num_batches,
        dynamic_ncols=True,
        position=device.index,
    )

    total_projections = euler_angles.shape[0] * defocus_values.shape[0]
    ##################################
    ### Start the orientation loop ###
    ##################################

    for i in orientation_batch_iterator:
        euler_angles_batch = euler_angles[
            i * projection_batch_size : (i + 1) * projection_batch_size
        ]
        rot_matrix = roma.euler_to_rotmat(
            "zyz", euler_angles, degrees=True, device=device
        )

        # Extract central slice(s) from the template volume
        fourier_slice = extract_central_slices_rfft_3d(
            volume_rfft=template_dft,
            image_shape=(h,) * 3,  # NOTE: requires cubic template
            rotation_matrices=rot_matrix,
        )
        fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
        fourier_slice *= -1  # flip contrast

        # Apply the projective filters on a new batch dimension
        fourier_slice = fourier_slice[None, ...] * projective_filters[:, None, ...]

        # NOTE: This is reshaping into a single batch dimension (not used)
        # fourier_slice = fourier_slice.reshape(
        #     -1, fourier_slice.shape[-2], fourier_slice.shape[-1]
        # )

        # Inverse Fourier transform into real space and normalize
        projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
        projections = torch.fft.ifftshift(projections, dim=(-2, -1))
        projections = normalize_template_projection_compiled(
            projections, (h, w), (H, W)
        )

        # Padded forward Fourier transform for cross-correlation
        projections_dft = torch.fft.rfftn(projections, dim=(-2, -1), s=(H, W))

        ### Cross correlation step by element-wise multiplication ###
        projections_dft = image_dft[None, None, ...] * projections_dft
        cross_correlation = torch.fft.irfftn(projections_dft, dim=(-2, -1))

        # Update the tracked statistics through compiled function
        do_iteration_statistics_updates_compiled(
            cross_correlation,
            euler_angles_batch,
            defocus_values,
            mip,
            best_phi,
            best_theta,
            best_psi,
            best_defocus,
            correlation_sum,
            correlation_squared_sum,
            projection_batch_size,
            H,
            W,
        )

    return {
        "mip": mip.cpu().numpy(),
        "best_phi": best_phi.cpu().numpy(),
        "best_theta": best_theta.cpu().numpy(),
        "best_psi": best_psi.cpu().numpy(),
        "best_defocus": best_defocus.cpu().numpy(),
        "correlation_sum": correlation_sum.cpu().numpy(),
        "correlation_squared_sum": correlation_squared_sum.cpu().numpy(),
        "total_projections": total_projections,
    }

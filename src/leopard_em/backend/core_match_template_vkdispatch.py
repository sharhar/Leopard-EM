import torch
import tqdm
import numpy as np

def euler_angles_to_rotation_matricies(angles: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles in ZYZ convention to rotation matrices.
    The angles are assumed to be in degrees.

    Parameters
    ----------
    angles : np.ndarray
        The Euler angles in ZYZ convention. Shape (n, 3) where n is the number of angles.

    Returns
    -------
    np.ndarray
        The rotation matrices corresponding to the Euler angles. Shape (n, 4, 4).
    """

    m = np.zeros(shape=(4, 4, angles.shape[0]), dtype=np.float32)

    cos_phi   = np.cos(np.deg2rad(angles[:, 0]))
    sin_phi   = np.sin(np.deg2rad(angles[:, 0]))
    cos_theta = np.cos(np.deg2rad(angles[:, 1]))
    sin_theta = np.sin(np.deg2rad(angles[:, 1]))
    cos_psi   = np.cos(np.deg2rad(angles[:, 2]))
    sin_psi   = np.sin(np.deg2rad(angles[:, 2]))
    m[0][0]   = cos_phi * cos_theta * cos_psi - sin_phi * sin_psi
    m[1][0]   = sin_phi * cos_theta * cos_psi + cos_phi * sin_psi
    m[2][0]   = -sin_theta * cos_psi

    m[0][1]   = -cos_phi * cos_theta * sin_psi - sin_phi * cos_psi
    m[1][1]   = -sin_phi * cos_theta * sin_psi + cos_phi * cos_psi
    m[2][1]   = sin_theta * sin_psi

    m[0][2]   = sin_theta * cos_phi
    m[1][2]   = sin_theta * sin_phi
    m[2][2]   = cos_theta

    return m.T

def decompose_best_indicies(
        best_indicies: np.ndarray,
        euler_angles: np.ndarray,
        defocus_values: np.ndarray,
        pixel_values: np.ndarray):
    """
    Decomposes the best indicies into the corresponding Euler angles,
    defocus values, and pixel sizes.

    Parameters
    ----------
    best_indicies : np.ndarray
        The best indicies from the cross-correlation.
    euler_angles : np.ndarray
        The Euler angles used for the cross-correlation.
    defocus_values : np.ndarray
        The defocus values used for the cross-correlation.
    pixel_values : np.ndarray
        The pixel sizes used for the cross-correlation.
    
    Returns
    -------
    best_phi : np.ndarray
        The best phi angles.
    best_theta : np.ndarray
        The best theta angles.
    best_psi : np.ndarray
        The best psi angles.
    best_defocus : np.ndarray
        The best defocus values.
    best_pixel_size : np.ndarray
        The best pixel sizes.
    """

    total_projections = defocus_values.shape[0] * pixel_values.shape[0]

    best_phi = euler_angles[best_indicies // total_projections, 0]
    best_theta = euler_angles[best_indicies // total_projections, 1]
    best_psi = euler_angles[best_indicies // total_projections, 2]

    best_defocus = defocus_values[best_indicies % defocus_values.shape[0]]
    best_pixel_size = pixel_values[(best_indicies % total_projections) // defocus_values.shape[0]]

    return best_phi, best_theta, best_psi, best_defocus, best_pixel_size

def _core_match_template_vkdispatch_single_gpu(
    result_dict: dict,
    device_id: int,
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    pixel_values: torch.Tensor,
    orientation_batch_size: int,
) -> None:
    """Single-GPU call for template matching using the vkdispatch backend. This
    function is an experimental implementation based on a pre-alpha build of the
    vkdispatch GPU-acceleration library. 

    NOTE: All tensors *must* be allocated on the same device. By calling the
    user-facing `core_match_template` function this is handled automatically.

    NOTE: The result_dict is a shared dictionary between processes and updated in-place
    with this processes's results under the 'device_id' key.

    Parameters
    ----------
    result_dict : dict
        Dictionary to store the results in.
    device_id : int
        ID of the device which computation is running on. Results will be stored
        in the dictionary with this key.
    image_dft : torch.Tensor
        Real-fourier transform (RFFT) of the image with large image filters
        already applied. Has shape (H, W // 2 + 1).
    template_dft : torch.Tensor
        Real-fourier transform (RFFT) of the template volume to take Fourier
        slices from. Has shape (l, h, w // 2 + 1). where l is the number of
        slices.
    euler_angles : torch.Tensor
        Euler angles (in 'ZYZ' convention) to search over. Has shape
        (orientations // n_devices, 3). This has already been split (e.g.
        4 devices has shape (orientations // 4, 3).
    projective_filters : torch.Tensor
        Multiplied 'ctf_filters' with 'whitening_filter_template'. Has shape
        (defocus_batch, h, w // 2 + 1). Is RFFT and not fftshifted.
    defocus_values : torch.Tensor
        What defoucs values correspond with the CTF filters. Has shape
        (defocus_batch,).
    orientation_batch_size : int
        The number of projections to calculate the correlation for at once.

    Returns
    -------
    None
    """

    #############################
    ### Initialize vkdispatch ###
    #############################

    # We import vkdispatch here to avoid requiring it unless the user opts to use it.
    # This ensures the code remains functional without vkdispatch unless explicitly needed.

    try:
        import vkdispatch as vd

        from .vkdispatch_utils import (
            extract_fft_slices,
            fftshift,
            get_template_sums,
            normalize_templates,
            transpose_kernel,
            do_padded_cross_correlation,
            accumulate_per_pixel,
        )
    except ImportError as exp:
        raise ImportError(
            "The 'vkdispatch' must be installed to use the enable_vkdispatch_experimental flag. "
            "Please install it as such: pip3 install leopard-em[vkdispatch]"
        ) from exp

    device_index = image_dft.device.index

    vd.initialize(debug_mode=True)
    vd.make_context(devices=[device_index]) # select the device we want to use

    ########################################################################
    ### Calculate full density volume FFT (this is faster in vkdispatch) ###
    ########################################################################

    template_dft_temp: torch.Tensor = torch.fft.ifftshift(template_dft, dim=(0, 1))
    density_volume: torch.Tensor = torch.fft.irfftn(template_dft_temp, dim=(0, 1, 2))
    density_volume_fft: torch.Tensor = torch.fft.fftn(density_volume, dim=(0, 1, 2))

    ####################################
    ### Copy Tensor data back to CPU ###
    ####################################

    # Accounting for RFFT shape
    projection_shape_real = (template_dft.shape[1], template_dft.shape[2] * 2 - 2)
    image_shape_real = (image_dft.shape[0], image_dft.shape[1] * 2 - 2)

    # Copy all tensors to numpy arrays
    image_dft_cpu = image_dft.cpu().numpy()
    density_volume_fft_cpu = density_volume_fft.cpu().numpy()
    euler_angles_cpu = euler_angles.cpu().numpy()
    projective_filters_cpu = projective_filters.cpu().numpy()
    pixel_values_cpu = pixel_values.cpu().numpy()
    defocus_values_cpu = defocus_values.cpu().numpy()

    # vkdispatch can only handle 1-3D arrays, so we need to flatten the projection filer tensor
    projective_filters_cpu = projective_filters_cpu.reshape(
        -1, projective_filters_cpu.shape[-2], projective_filters_cpu.shape[-1]
    )

    ########################################
    ### Upload Tensor data to vkdispatch ###
    ########################################

    image_dft_buffer = vd.RFFTBuffer(
        (image_dft_cpu.shape[0],
        (image_dft_cpu.shape[1] - 1) * 2)
    )

    image_dft_buffer.write_fourier(image_dft_cpu)

    image_dft_buffer_transposed = vd.RFFTBuffer(
        (image_dft_cpu.shape[0],
        (image_dft_cpu.shape[1] - 1) * 2)
    )

    projective_filters_buffer = vd.asbuffer(projective_filters_cpu)

    template_buffer = vd.RFFTBuffer(
        (projective_filters_cpu.shape[0],
         density_volume_fft_cpu.shape[0],
         density_volume_fft_cpu.shape[0])
    )
    
    template_buffer.write(np.zeros(shape=template_buffer.shape, dtype=np.complex64))

    template_buffer2 = vd.RFFTBuffer(
        (projective_filters_cpu.shape[0],
         density_volume_fft_cpu.shape[0],
         density_volume_fft_cpu.shape[0])
    )

    template_buffer2.write(np.zeros(shape=template_buffer2.shape, dtype=np.complex64))

    correlation_buffer = vd.RFFTBuffer(
        (projective_filters_cpu.shape[0],
         image_dft_cpu.shape[0],
         (image_dft_cpu.shape[1] - 1) * 2)
    )

    correlation_buffer.write(np.zeros(shape=correlation_buffer.shape, dtype=np.complex64))

    best_values_buffer = vd.Buffer(
        (correlation_buffer.shape[1], correlation_buffer.shape[1], 2),
        vd.float32
    )

    best_initial_values = np.zeros(shape=best_values_buffer.shape, dtype=np.float32)
    best_initial_values[:, :, 0] = -float("inf")
    best_initial_values[:, :, 1] = -1

    best_values_buffer.write(best_initial_values)

    sum_buffer = vd.asbuffer(np.zeros(
        shape=(correlation_buffer.shape[1], correlation_buffer.shape[1], 2),
        dtype=np.complex64)
    )

    template_image = vd.Image3D(density_volume_fft_cpu.shape, vd.float32, 2)
    template_image.write(density_volume_fft_cpu)

    transpose_kernel(
        correlation_buffer,
        image_dft_buffer,
        image_dft_buffer_transposed
    )

    # transposed_data = image_dft_buffer_transposed.read(0)

    # np.save(
    #     f"transposed_data_{device_id}.npy",
    #     transposed_data
    # )

    # exit()

    ########################################################
    ### Setup iterator object with tqdm for progress bar ###
    ########################################################

    num_batches = euler_angles.shape[0] // orientation_batch_size
    orientation_batch_iterator = tqdm.tqdm(
        range(num_batches),
        desc=f"Progress on device: {device_index}",
        leave=True,
        total=num_batches,
        dynamic_ncols=True,
        position=device_index,
    )

    total_projections = euler_angles.shape[0] * defocus_values.shape[0]

    ###################################################
    ### Create the CommandStream for later playback ###
    ###################################################

    cmd_stream = vd.CommandStream()

    vd.set_global_cmd_stream(cmd_stream)

    extract_fft_slices(
        template_buffer,
        projective_filters_buffer,
        template_image,
        density_volume_fft_cpu.shape,
        cmd_stream.bind_var("rotation_matrix"),
    )

    vd.fft.irfft2(template_buffer)

    fftshift(template_buffer2, template_buffer)

    # Now, we normalize the templates
    sums = get_template_sums(template_buffer2)
    normalize_templates(template_buffer2, sums, projection_shape_real, image_shape_real)

    do_padded_cross_correlation(
        template_buffer2,
        correlation_buffer,
        image_dft_buffer_transposed
    )

    accumulate_per_pixel(
        best_values_buffer,
        sum_buffer,
        correlation_buffer,
        cmd_stream.bind_var("index")
    )

    ##################################
    ### Start the orientation loop ###
    ##################################

    for i in orientation_batch_iterator:
        start_idx = i * orientation_batch_size
        end_idx = (i + 1) * orientation_batch_size

        euler_angles_batch = euler_angles_cpu[start_idx:end_idx]
        rotation_matricies = euler_angles_to_rotation_matricies(euler_angles_batch)

        cmd_stream.set_var("rotation_matrix", rotation_matricies)
        cmd_stream.set_var("index", list(range(start_idx, end_idx)))
        cmd_stream.submit(rotation_matricies.shape[0])

        # corrs = correlation_buffer.read(0)

        # for j in range(correlation_buffer.shape[0]):
        #     np.save(
        #         f"corr_{device_id}_{j}.npy",
        #         corrs[j]
        #     )
        
        # exit()

    accumulation = best_values_buffer.read(0)

    best_phi, best_theta, best_psi, best_defocus, best_pixel_size = decompose_best_indicies(
        accumulation[:, :, 1].astype(np.int32),
        euler_angles_cpu,
        defocus_values_cpu,
        pixel_values_cpu,
    )

    sums_result = sum_buffer.read(0)

    result = {
        "mip": accumulation[:, :, 0],
        "best_phi": best_phi,
        "best_theta": best_theta,
        "best_psi": best_psi,
        "best_defocus": best_defocus,
        "best_pixel_size": best_pixel_size,
        "correlation_sum": sums_result[:, :, 0].real, # accumulation[:, :, 2],
        "correlation_squared_sum": sums_result[:, :, 1].real,
        "total_projections": total_projections,
    }

    # Place the results in the shared multi-process manager dictionary so accessible
    # by the main process.
    result_dict[device_id] = result

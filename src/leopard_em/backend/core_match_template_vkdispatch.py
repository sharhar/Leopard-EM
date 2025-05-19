import torch
import tqdm
import time

import numpy as np

def euler_angles_to_rotation_matricies(angles: np.ndarray) -> np.ndarray:
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
    #m[3][0]   = offsets[0]

    m[0][1]   = -cos_phi * cos_theta * sin_psi - sin_phi * cos_psi
    m[1][1]   = -sin_phi * cos_theta * sin_psi + cos_phi * cos_psi
    m[2][1]   = sin_theta * sin_psi
    #m[3][1]   = offsets[1]    
    
    m[0][2]   = sin_theta * cos_phi
    m[1][2]   = sin_theta * sin_phi
    m[2][2]   = cos_theta
    #m[3][2]   = offsets[2]

    return m.T

def decompose_best_indicies(
        best_indicies: np.ndarray,
        euler_angles: np.ndarray,
        defocus_values: np.ndarray,
        pixel_values: np.ndarray,):

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
    """Single-GPU call for template matching.

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

    try:
        import vkdispatch as vd
        import vkdispatch.codegen as vc

        from .vkdispatch_utils import extract_fft_slices, fftshift, get_template_sums, normalize_templates, pad_templates
        from .vkdispatch_utils import accumulate_per_pixel
    except ImportError:
        raise ImportError("The 'vkdispatch' package must be installed to use the vkdispatch backend.")
    
    vd.initialize(debug_mode=True)

    vd.make_context(devices=[device_id]) #, queue_families=[[0, 2]])

    #################################################################################
    ### Inverse FFT shift the template, since we need to redo the FFT calculation ###
    #################################################################################

    template_dft_temp = torch.fft.ifftshift(template_dft, dim=(0, 1))
    density_volume = torch.fft.irfftn(template_dft_temp, dim=(0, 1, 2))
    density_volume_fft = torch.fft.fftn(density_volume, dim=(0, 1, 2))
    #density_volume_fft: torch.Tensor = torch.fft.fftshift(density_volume_fft, dim=(0, 1, 2))

    ###############################################################
    ### Copy Tensor data back to CPU and then to vkdispatch GPU ###
    ###############################################################

    # Accounting for RFFT shape
    projection_shape_real = (template_dft.shape[1], template_dft.shape[2] * 2 - 2)
    image_shape_real = (image_dft.shape[0], image_dft.shape[1] * 2 - 2)

    image_dft_cpu = image_dft.cpu().numpy()
    density_volume_fft_cpu = density_volume_fft.cpu().numpy()
    euler_angles_cpu = euler_angles.cpu().numpy()
    projective_filters_cpu = projective_filters.cpu().numpy()
    pixel_values_cpu = pixel_values.cpu().numpy()
    defocus_values_cpu = defocus_values.cpu().numpy()

    projective_filters_cpu = projective_filters_cpu.reshape(
        -1, projective_filters_cpu.shape[-2], projective_filters_cpu.shape[-1]
    )
    
    image_dft_buffer = vd.RFFTBuffer((image_dft_cpu.shape[0], (image_dft_cpu.shape[1] - 1) * 2)) #vd.asbuffer(image_dft_cpu)
    image_dft_buffer.write_fourier(image_dft_cpu)
    
    projective_filters_buffer = vd.asbuffer(projective_filters_cpu)

    template_buffer = vd.RFFTBuffer((projective_filters_cpu.shape[0], density_volume_fft_cpu.shape[0], density_volume_fft_cpu.shape[0]))
    template_buffer.write(np.zeros(shape=template_buffer.shape, dtype=np.complex64))
    
    template_buffer2 = vd.RFFTBuffer((projective_filters_cpu.shape[0], density_volume_fft_cpu.shape[0], density_volume_fft_cpu.shape[0]))
    template_buffer2.write(np.zeros(shape=template_buffer2.shape, dtype=np.complex64))
    
    correlation_buffer = vd.RFFTBuffer((projective_filters_cpu.shape[0], image_dft_cpu.shape[0], (image_dft_cpu.shape[1] - 1) * 2))
    correlation_buffer.write(np.zeros(shape=correlation_buffer.shape, dtype=np.complex64))

    accumulation_buffer = vd.Buffer((correlation_buffer.shape[1], correlation_buffer.shape[1], 4), vd.float32)

    accumulation_initial_values = np.zeros(shape=accumulation_buffer.shape, dtype=np.float32)
    accumulation_initial_values[:, :, 0] = -100000000 # -float("inf")
    accumulation_initial_values[:, :, 1] = -1

    accumulation_buffer.write(accumulation_initial_values)

    # density_volume_padding_size = 2
    # padded_density = np.zeros(
    #     shape=(
    #         density_volume_fft_cpu.shape[0] + density_volume_padding_size * 2, 
    #         density_volume_fft_cpu.shape[1] + density_volume_padding_size * 2,
    #         density_volume_fft_cpu.shape[2] + density_volume_padding_size * 2,),
    #     dtype=np.complex64)
    
    # padded_density[
    #     density_volume_padding_size : density_volume_padding_size + density_volume_fft_cpu.shape[0],
    #     density_volume_padding_size : density_volume_padding_size + density_volume_fft_cpu.shape[1],
    #     density_volume_padding_size : density_volume_padding_size + density_volume_fft_cpu.shape[2],
    # ] = density_volume_fft_cpu
    
    
    template_image = vd.Image3D(density_volume_fft_cpu.shape, vd.float32, 2)
    template_image.write(density_volume_fft_cpu)

    ########################################################
    ### Setup iterator object with tqdm for progress bar ###
    ########################################################

    num_batches = euler_angles.shape[0] // orientation_batch_size
    orientation_batch_iterator = tqdm.tqdm(
        range(num_batches),
        desc=f"Progress on device: {device_id}",
        leave=True,
        total=num_batches,
        dynamic_ncols=True,
        position=device_id,
    )

    total_projections = euler_angles.shape[0] * defocus_values.shape[0]

    ###################################################
    ### Create the CommandStream for later playback ###
    ###################################################

    cmd_stream = vd.CommandStream()

    vd.set_global_cmd_stream(cmd_stream)

    # This part of the TM algorithm has been correctly implemented

    extract_fft_slices(
        template_buffer,
        projective_filters_buffer,
        template_image,
        density_volume_fft_cpu.shape,
        cmd_stream.bind_var("rotation_matrix"),
    )

    vd.fft.irfft2(template_buffer)

    @vd.shader("buff.size")
    def clear_buffer(buff: vc.Buff[vc.c64]):
        buff[vc.global_invocation().x] = "vec2(0)"

    # #clear_buffer(template_buffer2)
    
    fftshift(template_buffer2, template_buffer)

    # Now, we normalize the templates
    sums = get_template_sums(template_buffer2)
    normalize_templates(template_buffer2, sums, projection_shape_real, image_shape_real)

    clear_buffer(correlation_buffer)

    # Now we pad the templates to the size of the image
    pad_templates(correlation_buffer, template_buffer2)
    
    # Do fused convolution of padded templates with reference image
    @vd.map_registers([vc.c64])
    def kernel_mapping(kernel_buffer: vc.Buffer[vc.c64]):
        img_val = vc.mapping_registers()[0]
        read_register = vc.mapping_registers()[1]

        read_register[:] = kernel_buffer[vc.mapping_index() % (image_dft_buffer.shape[0] * image_dft_buffer.shape[1])]
        img_val[:] = vc.mult_conj_c64(read_register, img_val)

    vd.fft.convolve2DR(correlation_buffer, image_dft_buffer, kernel_map=kernel_mapping)

    accumulate_per_pixel(
        accumulation_buffer,
        correlation_buffer,            
        cmd_stream.bind_var("index")
    )

    ##################################
    ### Start the orientation loop ###
    ##################################

    for i in orientation_batch_iterator:
        euler_angles_batch = euler_angles_cpu[
            i * orientation_batch_size : (i + 1) * orientation_batch_size
        ]

        rotation_matricies = euler_angles_to_rotation_matricies(euler_angles_batch)

        cmd_stream.set_var("rotation_matrix", rotation_matricies)
        cmd_stream.set_var("index", list(range(i * orientation_batch_size, (i + 1) * orientation_batch_size)))
        cmd_stream.submit(rotation_matricies.shape[0])

        # accumulation = accumulation_buffer.read(0)

        # np.save(f"mip_{device_id}.npy", accumulation[:, :, 0])
        # np.save(f"best_index_{device_id}.npy", accumulation[:, :, 1])
        # np.save(f"correlation_sum_{device_id}.npy", accumulation[:, :, 2])
        # np.save(f"correlation_sum2_{device_id}.npy", accumulation[:, :, 3])

        # corrs = accumulation_buffer.read(0)

        # for ii, corr in enumerate(corrs):
        #     np.save(f"test_data/corr_{device_id}_{ii}.npy", corr[:, :, 0])

        # time.sleep(1)

        # exit()
    
    accumulation = accumulation_buffer.read(0)

    best_phi, best_theta, best_psi, best_defocus, best_pixel_size = decompose_best_indicies(
        accumulation[:, :, 1].astype(np.int32),
        euler_angles_cpu,
        defocus_values_cpu,
        pixel_values_cpu,
    )

    result = {
        "mip": accumulation[:, :, 0],
        "best_phi": best_phi,
        "best_theta": best_theta,
        "best_psi": best_psi,
        "best_defocus": best_defocus,
        "best_pixel_size": best_pixel_size,
        "correlation_sum": accumulation[:, :, 2],
        "correlation_squared_sum": accumulation[:, :,3],
        "total_projections": total_projections,
    }

    # Place the results in the shared multi-process manager dictionary so accessible
    # by the main process.
    result_dict[device_id] = result



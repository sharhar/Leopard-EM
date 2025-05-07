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

def _core_match_template_vkdispatch_single_gpu(
    result_dict: dict,
    device_id: int,
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
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
    except ImportError:
        raise ImportError("The 'vkdispatch' package must be installed to use the vkdispatch backend.")
    
    vd.make_context(devices=[device_id]) #, all_queues=True)

    ###############################################################
    ### Copy Tensor data back to CPU and then to vkdispatch GPU ###
    ###############################################################

    image_dft_cpu = image_dft.cpu().numpy()
    template_dft_cpu = template_dft.cpu().numpy()
    euler_angles_cpu = euler_angles.cpu().numpy()
    projective_filters_cpu = projective_filters.cpu().numpy()
    defocus_values_cpu = defocus_values.cpu().numpy()

    image_dft_buffer = vd.asbuffer(image_dft_cpu)
    projective_filters_buffer = vd.asbuffer(projective_filters_cpu)
    defocus_values_buffer = vd.asbuffer(defocus_values_cpu)

    template_image = vd.Image3D(template_dft_cpu.shape, dtype=vd.float32)
    template_image.write(template_dft_cpu)

    print("image_dft_cpu", image_dft_cpu.shape)
    print("template_dft_cpu", template_dft_cpu.shape)
    print("euler_angles_cpu", euler_angles_cpu.shape)
    print("projective_filters_cpu", projective_filters_cpu.shape)
    print("defocus_values_cpu", defocus_values_cpu.shape)

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

    @vd.shader(exec_size=lambda args: args.buff.size)
    def extract_fft_slices(
        buff: vc.Buff[vc.c64], 
        img: vc.Img3[vc.f32], 
        img_shape: vc.Const[vc.iv4], 
        rotation: vc.Var[vc.m4]):

        ind = vc.global_invocation().x.cast_to(vc.i32).copy()
        
        # calculate the planar position of the current buffer pixel
        my_pos = vc.new_vec4(0, 0, 0, 1)
        my_pos.xy[:] = vc.unravel_index(ind, buff.shape).xy
        my_pos.xy += buff.shape.xy / 2
        my_pos.xy[:] = vc.mod(my_pos.xy, buff.shape.xy)
        my_pos.xy -= buff.shape.xy / 2

        # rotate the position to 3D template space
        my_pos[:] = rotation * my_pos
        my_pos.xyz += img_shape.xyz.cast_to(vc.v3) / 2
        
        # sample the 3D image at the current position
        buff[ind] = img.sample(my_pos.xyz).xy

    ###################################################
    ### Create the CommandStream for later playback ###
    ###################################################

    cmd_stream = vd.CommandStream()

    vd.set_global_cmd_stream(cmd_stream)

    extract_fft_slices(
        image_dft_buffer,
        template_image,
        (*template_image.shape, 0),
        cmd_stream.bind_var("rotation_matrix"),
    )

    ##################################
    ### Start the orientation loop ###
    ##################################

    for i in orientation_batch_iterator:
        euler_angles_batch = euler_angles_cpu[
            i * orientation_batch_size : (i + 1) * orientation_batch_size
        ]

        rotation_matricies = euler_angles_to_rotation_matricies(euler_angles_batch)

        print(euler_angles_batch)

        time.sleep(1)  # Sleep to allow the progress bar to update

        exit()



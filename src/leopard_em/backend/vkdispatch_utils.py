import builtins
from contextlib import contextmanager

import vkdispatch as vd
import vkdispatch.codegen as vc

def extract_fft_slices(
        template_buffer: vd.Buffer,
        projection_filters: vd.Buffer,
        image: vd.Image3D,
        image_shape: tuple[int, int, int],
        rotation: vc.Var[vc.m4]):
    """
    This function extracts the FFT slices from the image and stores them in the
    template buffer. The FFT slices are extracted by rotating the image and
    projecting it onto the template buffer. The projection filters are used to
    filter the FFT slices before they are stored in the template buffer.

    Parameters
    ----------
    template_buffer : vd.Buffer
        The buffer to store the FFT slices in.
    projection_filters : vd.Buffer
        The filters to apply to the FFT slices.
    image : vd.Image3D
        The image to extract the FFT slices from.
    rotation : vc.Var[vc.m4]
        The rotation matrix to apply to the image before extracting the FFT slices.
    """

    # We generate the shader source inside this function because we want to hardcode
    # information about the buffer size into the source code of the shader.
    @vd.shader(exec_size=lambda args: args.buff.shape[1] * args.buff.shape[2])
    def extract_fft_slices_shader(
        buff: vc.Buff[vc.c64],
        projections: vc.Buff[vc.f32],
        img: vc.Img3[vc.f32], 
        img_shape: vc.Const[vc.iv4], 
        rotation: vc.Var[vc.m4]):
        """
        This is the shader that performs the sampling of the volume's FFT and
        the application of the projection filters.

        Parameters
        ----------
        buff : vc.Buff[vc.c64]
            The buffer to store the FFT slices in.
        projections : vc.Buff[vc.f32]
            The filters to apply to the FFT slices.
        img : vc.Img3[vc.f32]
            The image to extract the FFT slices from.
        img_shape : vc.Const[vc.iv4]
            The shape of the image.
        rotation : vc.Var[vc.m4]
            The rotation matrix to apply to the image before extracting the FFT slices.
        """

        ind = vc.global_invocation().x.cast_to(vc.i32).copy()
        
        # calculate the planar position of the current buffer pixel
        my_pos = vc.new_vec4(0, 0, 0, 1)
        my_pos.x = ind % template_buffer.shape[2]
        my_pos.y = ind / template_buffer.shape[2]
        
        # my_pos.x += img_shape.y / 2
        # my_pos.x[:] = vc.mod(my_pos.x, img_shape.y)
        # my_pos.x -= img_shape.y / 2

        my_pos.y += img_shape.y / 2
        my_pos.y[:] = vc.mod(my_pos.y, img_shape.y)
        my_pos.y -= img_shape.y / 2
        
        # rotate the position to 3D template space
        my_pos[:] = rotation * my_pos

        vc.if_any(my_pos.x < -256, my_pos.x > 256, my_pos.y < -256, my_pos.y > 256, my_pos.z < -256, my_pos.z > 256)
        for i in range(projection_filters.shape[0]):
            index = ind + i * template_buffer.shape[1] * template_buffer.shape[2]
            buff[index].x = 0
            buff[index].y = 0
        vc.return_statement()
        vc.end()

        my_pos.xy[:] = -1 * img.sample(my_pos.xyz).xy
        
        for i in range(projection_filters.shape[0]):
            index = ind + i * template_buffer.shape[1] * template_buffer.shape[2]
            buff[index] = my_pos.xy * projections[index]

    extract_fft_slices_shader(
        template_buffer,
        projection_filters,
        image.sample(
            address_mode=vd.AddressMode.REPEAT,
        ),
        (*image_shape, 0),
        rotation
    )

@vd.shader(exec_size=lambda args: args.input.size * 2)
def fftshift(output: vc.Buff[vc.f32], input: vc.Buff[vc.f32]):
    """
    This function performs an FFT shift on the input buffer and stores the result
    in the output buffer.

    Parameters
    ----------
    output : vc.Buff[vc.f32]
        The buffer to store the FFT shift result in.
    input : vc.Buff[vc.f32]
        The buffer to perform the FFT shift on.
    """

    ind = vc.global_invocation().x.cast_to(vd.int32).copy()

    image_ind = vc.new_int()
    image_ind[:] = ind % (input.shape.y * input.shape.z * 2)

    out_x = (image_ind / (2 * input.shape.z)).copy()
    out_y = (image_ind % (2 * input.shape.z)).copy()

    vc.if_statement(out_y >= 2 * input.shape.z - 2)
    output[ind].x = 0
    output[ind].y = 0
    vc.return_statement()
    vc.end()

    image_ind[:] = ind / (input.shape.y * input.shape.z * 2)

    image_ind[:] = image_ind * (input.shape.y * input.shape.z * 2)

    in_x = ((out_x + input.shape.y / 2) % output.shape.y).copy()
    in_y = ((out_y + input.shape.y / 2) % output.shape.y).copy()

    image_ind += in_x * 2 * input.shape.z + in_y

    output[ind] = input[image_ind]


@contextmanager
def suppress_print():
    """
    Context manager to suppress print statements in vkdispatch shaders.
    """

    original_print = builtins.print
    builtins.print = lambda *args, **kwargs: None
    try:
        yield
    finally:
        builtins.print = original_print

def get_template_sums(templates: vd.Buffer):
    """
    This function calculates the sums of the templates in the batch. It does this by
    mapping the templates to a 4-vector which contains:
    1) The sum of the two real-space pixels
    2) The sum of the squares of the two real-space pixels
    3) The sum of only edge pixels
    4) Reserved as padding

    These vectors then get added together per template such that in the end we get the sum
    vector for each of out templates in this batch.

    Parameters
    ----------
    templates : vd.Buffer
        The buffer to store the sums in.
    """

    @vd.map_reduce(vd.SubgroupAdd, axes=[1, 2])
    def calculate_sums(buff: vc.Buff[vc.v2]) -> vc.v4:
        """
        This function is a mapping function that preprocesses each "couplet" of real-space pixels
        in the projected template into a 4-vector which contains:
        1) The sum of the two real-space pixels
        2) The sum of the squares of the two real-space pixels
        3) The sum of only edge pixels
        4) Reserved as padding

        These vectors then get added together per template such that in the end we get the sum
        vector for each of out templates in this batch.

        Parameters
        ----------
        buff : vc.Buff[vc.v2]
            The buffer to store the sums in.
        """

        ind = vc.mapping_index()

        result = vc.new_vec4()

        result.zw[:] = buff[ind]

        result.x = result.z + result.w
        result.y = result.z * result.z + result.w * result.w

        in_batch_index = ind % (templates.shape[1] * templates.shape[2])

        x_index = in_batch_index % templates.shape[2]
        y_index = in_batch_index / templates.shape[2]

        vc.if_any(y_index == 0, y_index == templates.shape[1] - 1)
        result.z = result.x
        vc.else_if_statement(x_index == 0)
        vc.else_if_statement(x_index == templates.shape[2] - 1)
        result.x = 0
        result.y = 0
        result.z = 0

        buff[ind].x = 0
        buff[ind].y = 0

        vc.else_if_statement(x_index == templates.shape[2] - 2)
        result.z = result.w
        vc.else_statement()
        result.z = 0
        vc.end()

        result.w = 0

        return result

    # I accidentally left some print statements in the reduction code in vkdispatch
    # so for now we will suppress the print statements when calling this function,
    # and I will fix this in the next release.
    with suppress_print():
        sums = calculate_sums(templates)

    return sums

@vd.shader(exec_size=lambda args: args.buff.size * 2)
def normalize_templates_shader(
    buff: vc.Buff[vc.f32],
    sums: vc.Buff[vc.v4],
    relative_size: vc.Const[vc.v4]):
    """
    This is the shader for 'normalize_templates'.

    Parameters
    ----------
    buff : vc.Buff[vc.f32]
        The buffer to normalize.
    sums : vc.Buff[vc.v4]
        The buffer containing the sums of the templates.
    relative_size : vc.Const[vc.v4]
        The relative size of the small and large templates.
    """

    ind = vc.global_invocation().x.copy()

    vc.if_statement(ind % (2 * buff.shape.z) >= 2 * buff.shape.z - 2)
    vc.return_statement()
    vc.end()

    template_index = ind / (buff.shape.y * buff.shape.z * 2)

    template_sums = sums[template_index].copy()

    N = vc.new_int(buff.shape.y * buff.shape.y)
    mean = vc.new_float(template_sums.x / N)
    variance = vc.new_float(template_sums.y)
    edge_mean = vc.new_float(template_sums.z / (4 * buff.shape.y - 4))

    # This block calculates the "mean" and "variance" values such that they will
    # equal the same values in the "normalize_template_projection" function after 
    # we subtract the edge mean.
    variance[:] = variance - 2 * N * edge_mean * mean
    mean[:] = (mean - edge_mean) * relative_size[0] * relative_size[0]
    variance[:] = variance + mean * mean * (1 - 2 * N / (relative_size[0] * relative_size[0]))

    # These lines then match the same lines of code in the "normalize_template_projection" function
    variance[:] = variance + relative_size[1] * mean * mean
    variance[:] = variance / relative_size[2]

    buff[ind] = (buff[ind] - edge_mean) / vc.sqrt(variance)


def normalize_templates(
        templates_buffer: vd.Buffer,
        sums_buffer: vd.Buffer,
        small_shape: tuple[int, int],
        large_shape: tuple[int, int]):
    """
    This function normalizes the templates in the buffer by subtracting the mean
    of the edge pixels and dividing by the standard deviation. The standard deviation
    is calculated such that the variance of the zero-padded projection is 1.
    The mean of the edge pixels is subtracted from the template.

    Parameters
    ----------
    templates_buffer : vd.Buffer
        The buffer to normalize.
    sums_buffer : vd.Buffer
        The buffer containing the sums of the templates.
    small_shape : tuple[int, int]
        The shape of the small template.
    large_shape : tuple[int, int]
        The shape of the large template.
    """

    relative_size_array = [
        (small_shape[0] * small_shape[1]) / (large_shape[0] * large_shape[1]),
        (large_shape[0] - small_shape[0]) * (large_shape[1] - small_shape[1]),
        large_shape[0] * large_shape[1],
        0 # Padding
    ]

    normalize_templates_shader(
        templates_buffer,
        sums_buffer,
        relative_size_array)

@vd.shader("input.size")
def pad_templates(output: vc.Buff[vc.c64], input: vc.Buff[vc.c64]):
    """
    This function pads the input buffer with zeros to match the size of the output buffer.
    The input buffer is assumed to be in the small space and the output buffer is
    assumed to be in the large space.

    Parameters
    ----------
    output : vc.Buff[vc.c64]
        The buffer to put the padded data in.
    input : vc.Buff[vc.c64]
        The buffer to read from.
    """

    ind = vc.global_invocation().x.copy()

    ind_0 = ind % input.shape.z
    ind_1 = (ind / input.shape.z) % input.shape.y
    ind_2 = ind / (input.shape.y * input.shape.z)

    new_ind = vc.new_uint(0)
    new_ind[:] = new_ind + ind_0
    new_ind[:] = new_ind + ind_1 * output.shape.z
    new_ind[:] = new_ind + ind_2 * output.shape.y * output.shape.z

    output[new_ind] = input[ind]

def accumulate_per_pixel(
        accumulation_buffer: vd.Buffer,
        correlation_signal: vd.RFFTBuffer,
        index_var: vc.Var[vc.i32],):
    """
    This function accumulates the correlation signal per pixel in the accumulation buffer.

    Parameters
    ----------
    accumulation_buffer : vd.Buffer
        The buffer to accumulate the correlation signal in.
    correlation_signal : vd.RFFTBuffer
        The buffer containing the correlation signal.
    index_var : BoundVariable
        The variable to store the index of the correlation signal.
    """

    @vd.shader("accum_buff.size // 4")
    def accumulation_shader(
        accum_buff: vc.Buff[vc.f32],
        back_buffer: vc.Buff[vc.f32],
        index: vc.Var[vc.i32]):

    # with vc.builder_context(enable_exec_bounds=False) as builder:
    #     signature = vd.ShaderSignature.from_type_annotations(
    #         builder,
    #         [vc.Buff[vc.f32], vc.Buff[vc.f32], vc.Var[vc.i32]]
    #     )
        
    #     accum_buff = signature.get_variables()[0]
        
    #     back_buffer = signature.get_variables()[1]

    #     index: vc.ShaderVariable = signature.get_variables()[2]

        ind = vc.global_invocation().x.copy("ind")
        ind_padded = vc.new_int(ind + 2 * (ind / correlation_signal.shape[1]))

        curr_mip = back_buffer[ind_padded].copy("curr_mip")
        
        curr_index = (correlation_signal.shape[0] * index).copy("curr_index")
        sum_cross_register = accum_buff[4 * ind + 2].copy("sum_cross_register")
        sum2_cross_register = accum_buff[4 * ind + 3].copy("sum2_cross_register")

        best_mip = vc.new_float(curr_mip, var_name="best_mip")
        best_curr_index = vc.new_int(curr_index, var_name="best_curr_index")

        sum_cross_register[:] = sum_cross_register + curr_mip
        sum2_cross_register[:] = sum2_cross_register + curr_mip * curr_mip

        for i in range(1, correlation_signal.shape[0]):
            curr_mip[:] = back_buffer[ind_padded + i * (correlation_signal.shape[1] * correlation_signal.shape[2] * 2)]

            sum_cross_register[:] = sum_cross_register + curr_mip
            sum2_cross_register[:] = sum2_cross_register + curr_mip * curr_mip

            vc.if_statement(curr_mip > best_mip)
            best_mip[:] = curr_mip
            best_curr_index[:] = correlation_signal.shape[0] * index + i
            vc.end()

        accum_buff[4 * ind + 2] = sum_cross_register
        accum_buff[4 * ind + 3] = sum2_cross_register
        
        vc.if_statement(best_mip > accum_buff[4 * ind])
        accum_buff[4 * ind] = best_mip
        accum_buff[4 * ind + 1] = best_curr_index
        vc.end()

        #shader_object = vd.ShaderObject(builder.build("accumulation_shader"), signature)

    accumulation_shader(
        accumulation_buffer,
        correlation_signal,
        index_var,
        #exec_size=(correlation_signal.shape[1] * correlation_signal.shape[1], 1, 1),
    )
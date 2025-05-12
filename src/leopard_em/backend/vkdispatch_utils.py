import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

from typing import Tuple, List, Callable

import builtins
from contextlib import contextmanager

def extract_fft_slices(template_buffer: vd.Buffer, projection_filters: vd.Buffer, image: vd.Image3D, rotation: vc.Var[vc.m4]):

    # We generate the shader source inside this function because we want to hardcode
    # information about the buffer size into the source code of the shader.
    @vd.shader(exec_size=lambda args: args.buff.shape[1] * args.buff.shape[2])
    def extract_fft_slices_shader(
        buff: Buff[c64],
        projections: Buff[f32],
        img: Img3[f32], 
        img_shape: Const[iv4], 
        rotation: Var[m4]):

        ind = vc.global_invocation().x.cast_to(vc.i32).copy()

        vc.if_statement(ind == 0)
        buff[ind].x = 0
        buff[ind].y = 0
        vc.return_statement()
        vc.end()
        
        # calculate the planar position of the current buffer pixel
        my_pos = vc.new_vec4(0, 0, 0, 1)
        my_pos.x = ind % template_buffer.shape[2]
        my_pos.y = ind / template_buffer.shape[2]
        
        my_pos.y += img_shape.y / 2
        my_pos.y[:] = vc.mod(my_pos.y, img_shape.y)
        my_pos.y -= img_shape.y / 2
        
        # rotate the position to 3D template space
        my_pos[:] = rotation * my_pos
        my_pos.xyz += img_shape.xyz.cast_to(vc.v3) / 2

        my_pos.xy[:] = -1 * img.sample(my_pos.xyz).xy

        for i in range(projection_filters.shape[0]):
            index = ind + i * template_buffer.shape[1] * template_buffer.shape[2]
            buff[index] = my_pos.xy * projections[index]

    extract_fft_slices_shader(
        template_buffer,
        projection_filters,
        image.sample(),
        (*image.shape, 0),
        rotation
    )

@vd.shader(exec_size=lambda args: args.input.size * 2)
def fftshift(output: Buff[f32], input: Buff[f32]):
    ind = vc.global_invocation().x.cast_to(vd.int32).copy()

    image_ind = vc.new_int()
    image_ind[:] = ind % (input.shape.y * input.shape.y)

    out_x = (image_ind / output.shape.y).copy()
    out_y = (image_ind % output.shape.y).copy()

    image_ind[:] = ind / (input.shape.y * input.shape.y)

    in_x = ((out_x + input.shape.y / 2) % output.shape.y).copy()
    in_y = ((out_y + input.shape.y / 2) % output.shape.y).copy()

    image_ind += in_x * input.shape.y + in_y

    ind[:] = ind + 2 * (ind / input.shape.y)
    image_ind[:] = image_ind + 2 * (image_ind / input.shape.y)

    output[ind] = input[image_ind]


@contextmanager
def suppress_print():
    original_print = builtins.print
    builtins.print = lambda *args, **kwargs: None
    try:
        yield
    finally:
        builtins.print = original_print

def get_template_sums(templates: vd.Buffer):

    @vd.map_reduce(vd.SubgroupAdd, axes=[1, 2])
    def calculate_sums(buff: Buff[v2]) -> v4:
        """
        This function is a mapping function that preprocesses each "couplet" of real-space pixels
        in the projected template into a 4-vector which contains:
        1) The sum of the two real-space pixels
        2) The sum of the squares of the two real-space pixels
        3) The sum of only edge pixels
        4) Reserved as padding

        These vectors then get added together per template such that in the end we get the sum
        vector for each of out templates in this batch.
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
def normalize_templates_shader(buff: Buff[f32], sums: Buff[v4], relative_size: Const[v4]):
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

def normalize_templates(templates_buffer, sums_buffer, small_shape, large_shape):
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
def pad_templates(output: Buff[c64], input: Buff[c64]):
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
        index_var: Callable[[Tuple[str, str]], None],):

    with vc.builder_context(enable_exec_bounds=False) as builder:
        signature = vd.ShaderSignature.from_type_annotations(
            builder,
            [Buff[f32], Buff[f32], Var[i32]]
        )
        
        accum_buff = signature.get_variables()[0]
        
        back_buffer = signature.get_variables()[1]

        index: vc.ShaderVariable = signature.get_variables()[2]

        ind = vc.global_invocation().x.copy()
        ind_padded = vc.new_int(ind + 2 * (ind / correlation_signal.shape[1]))

        curr_mip = back_buffer[ind_padded].copy()
        curr_index = (correlation_signal.shape[0] * index).copy()
        sum_cross_register = accum_buff[4 * ind + 2].copy()
        sum2_cross_register = accum_buff[4 * ind + 3].copy()

        best_mip = vc.new_float(curr_mip)
        best_curr_index = vc.new_int(curr_index)

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
        
        vc.if_statement(curr_mip > accum_buff[4 * ind])
        accum_buff[4 * ind] = curr_mip
        accum_buff[4 * ind + 1] = best_curr_index
        vc.end()

        shader_object = vd.ShaderObject(builder.build("accumulation_shader"), signature)

    shader_object(
        accumulation_buffer,
        correlation_signal,
        index_var,
        exec_size=(correlation_signal.shape[1] * correlation_signal.shape[1], 1, 1),
    )
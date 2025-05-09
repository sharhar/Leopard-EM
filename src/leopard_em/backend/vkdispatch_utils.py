import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import builtins
from contextlib import contextmanager

@contextmanager
def suppress_print():
    original_print = builtins.print
    builtins.print = lambda *args, **kwargs: None
    try:
        yield
    finally:
        builtins.print = original_print

@vd.map_reduce(vd.SubgroupAdd)
def calc_sums(wave: Buff[v2]) -> v2:
    ind = vc.mapping_index()

    result = vc.new_vec2()

    result.x = wave[ind].x
    result.y = result.x * result.x

    wave[ind].x = result.x
    wave[ind].y = 0

    return result

@vd.shader(exec_size=lambda args: args.image.size)
def apply_normalization(image: Buff[v2], sum_buff: Buff[v2]):
    ind = vc.global_invocation().x.copy()

    sum_vec = (sum_buff[0] / (image.shape.x * image.shape.y)).copy()
    sum_vec.y = vc.sqrt(sum_vec.y - sum_vec.x * sum_vec.x)

    image[ind].x = (image[ind].x - sum_vec.x) / sum_vec.y

def normalize_signal(signal: vd.Buffer):

    # I accidentally left some print statements in the reduction code in vkdispatch
    # so for now we will suppress the print statements when calling this function,
    # and I will fix this in the next release.
    with suppress_print():
        sum_buff = calc_sums(signal) # The reduction returns a buffer with the result in the first value
        apply_normalization(signal, sum_buff)

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
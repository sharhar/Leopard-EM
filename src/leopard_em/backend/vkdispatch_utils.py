import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

@vd.shader(exec_size=lambda args: args.buff.size)
def extract_fft_slices(
    buff: vc.Buff[vc.c64], 
    img: vc.Img3[vc.f32], 
    img_shape: vc.Const[vc.iv4], 
    rotation: vc.Var[vc.m4]):

    ind = vc.global_invocation().x.cast_to(vc.i32).copy()
    
    # calculate the planar position of the current buffer pixel
    my_pos = vc.new_vec4(0, 0, 0, 1)
    my_pos.x = ind % buff.shape.y
    my_pos.y = ind / buff.shape.y
    
    my_pos.y += img_shape.y / 2
    my_pos.y[:] = vc.mod(my_pos.y, img_shape.y)
    my_pos.y -= img_shape.y / 2
    
    # rotate the position to 3D template space
    my_pos[:] = rotation * my_pos
    my_pos.xyz += img_shape.xyz.cast_to(vc.v3) / 2

    buff[ind] = img.sample(my_pos.xyz).xy
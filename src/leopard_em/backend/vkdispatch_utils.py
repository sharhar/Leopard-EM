import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

@vd.shader(exec_size=lambda args: args.buff.size)
def template_slice(buff: Buff[c64], img: Img3[f32], img_shape: Const[iv4], rotation: Var[m4]):
    ind = vc.global_invocation().x.cast_to(i32).copy()
    
    # calculate the planar position of the current buffer pixel
    my_pos = vc.new_vec4(0, 0, 0, 1)
    my_pos.xy[:] = vc.unravel_index(ind, buff.shape).xy
    my_pos.xy += buff.shape.xy / 2
    my_pos.xy[:] = vc.mod(my_pos.xy, buff.shape.xy)
    my_pos.xy -= buff.shape.xy / 2

    # rotate the position to 3D template space
    my_pos[:] = rotation * my_pos
    my_pos.xyz += img_shape.xyz.cast_to(v3) / 2
    
    # sample the 3D image at the current position
    buff[ind] = img.sample(my_pos.xyz).xy
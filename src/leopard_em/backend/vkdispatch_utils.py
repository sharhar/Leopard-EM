import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

def extract_fft_slices(template_buffer: vd.Buffer, projection_filters: vd.Buffer, image: vd.Image3D, rotation: vc.Var[vc.m4]):
    print(template_buffer.shape, projection_filters.shape, image.shape)

    @vd.shader(exec_size=lambda args: args.buff.shape[1] * args.buff.shape[2])
    def extract_fft_slices_shader(
        buff: vc.Buff[vc.c64],
        projections: vc.Buff[vc.f32],
        img: vc.Img3[vc.f32], 
        img_shape: vc.Const[vc.iv4], 
        rotation: vc.Var[vc.m4]):

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

            buff[index] = my_pos.xy * projections[index] # vc.mult_c64(my_pos.xy, projections[index])


        #buff[ind] = -1 * img.sample(my_pos.xyz).xy

    extract_fft_slices_shader(
        template_buffer,
        projection_filters,
        image.sample(),
        (*image.shape, 0),
        rotation
    )
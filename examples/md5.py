import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cyglfw3 as glfw
import time
import math

if not glfw.Init():
    exit()

version = (4,0)
glfw.WindowHint(glfw.CLIENT_API, glfw.OPENGL_API)
major, minor = version
glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, major)
glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, minor)
glfw.WindowHint(glfw.CONTEXT_ROBUSTNESS, glfw.NO_ROBUSTNESS)
glfw.WindowHint(glfw.OPENGL_FORWARD_COMPAT, 1)
glfw.WindowHint(glfw.OPENGL_DEBUG_CONTEXT, 1)
glfw.WindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

window_size = (640, 480)
window = glfw.CreateWindow(window_size[0], window_size[1], 'Bast')
if not window:
    glfw.Terminate()
    exit()

glfw.MakeContextCurrent(window)



from OpenGL import GL
import numpy as np

#from bast.common import debug
#debug.print_gl_calls()


vss = """
    #version 400
    layout (location = 0) in vec3 in_position;
    layout (location = 1) in vec2 in_uv;
    layout (location = 2) in uvec2 in_weight_indices;

    uniform mat4 in_projection;
    uniform mat4 in_model_view;
    uniform samplerBuffer in_weights;
    uniform samplerBuffer in_frame;

    out vec2 ex_uv;

    vec3 rotate_vector(vec4 quat, vec3 vec) {
        return vec + 2.0 * cross(cross(vec, quat.xyz) + quat.w * vec, quat.xyz);
    }

    void main() {
        uint weight_start = in_weight_indices.x;
        uint weight_count = in_weight_indices.y;

        vec4 position = vec4(0.,0.,0.,0.);

        for(uint index=0; index < weight_count; index++) {
            vec4 weight = texelFetch(in_weights, int(weight_start + index));
            vec4 weight_position = vec4(weight.rgb, 1.0);
            float f_bone_index;
            float weight_bias = modf(weight.a, f_bone_index);
            int bone_index = int(f_bone_index);

            if(weight_bias < 0.0001) {
                weight_bias = 1.0;
            }

            vec4 bone_position = texelFetch(in_frame, int(2 * bone_index));
            vec4 bone_orientation = texelFetch(in_frame, int(2 * bone_index + 1));

            vec4 weighted_position = (bone_position + vec4(rotate_vector(bone_orientation, weight_position.rgb),1.0)) * weight_bias;
            position += weighted_position;
        }

        gl_Position = in_projection * in_model_view * position;
        ex_uv = in_uv;
    }
    """

fss = """
    #version 400
    uniform sampler2D in_diffuse_texture;
    uniform samplerBuffer in_weights;
    in vec2 ex_uv;
    layout (location = 0) out vec4 out_color;

    void main(void) {
        out_color = texture(in_diffuse_texture, ex_uv);
    }
    """


from bast.graphics.shader.shader import FragmentShader, VertexShader
from bast.graphics.shader.program import Program
from bast.graphics.buffer.buffer import VertexBuffer, IndexBuffer, TextureBuffer
from bast.graphics.mesh.mesh import Mesh
from bast.graphics.mesh.sub_mesh import SubMesh
from bast.graphics.material.material import Material
from bast.graphics.texture.texture import Texture2D
from pyrr import Matrix44


fs = FragmentShader(fss)
vs = VertexShader(vss)
sp = Program([vs, fs], frag_locations='out_color')

with sp:
    sp.in_diffuse_texture = 0
    sp.in_weights = 1
    sp.in_frame = 2


meshes = []
textures = {}

from bast.graphics.buffer.buffer import UniformBuffer
ub = UniformBuffer(shape=(2,3), dtype=np.float32)
ub.bind()

from bast.graphics.mesh.md5.md5 import MD5_Mesh, MD5_Anim

mesh = MD5_Mesh.open('assets/md5/boblampclean.md5mesh')


#inverse_bind_pose = mesh.joints.prepare_inverse_bind_pose()
#inverse_bind_pose = TextureBuffer(inverse_bind_pose, internal_format=GL.GL_RGBA32F)

for submesh in mesh.meshes:
    filename = submesh.shader_filename
    if filename not in textures:
        textures[filename] = Texture2D.open('assets/md5/{}'.format(submesh.shader_filename))
    texture = textures[filename]

    weights = submesh.prepare_weights()
    weights_buffer = TextureBuffer(weights, internal_format=GL.GL_RGBA32F)

    mat = Material(sp,
        in_diffuse_texture=texture,
        in_weights=weights_buffer,
    )

    vertices = submesh.prepare_mesh(mesh.joints)

    vertex_buffer = VertexBuffer(vertices)
    index_buffer = IndexBuffer(submesh.tris)
    sm = SubMesh(mat, indices=index_buffer, **vertex_buffer.pointers)

    meshes.append(sm)

mesh = Mesh(meshes)


# animation
anim = MD5_Anim.open('assets/md5/boblampclean.md5anim')
frames = []

for frame in anim.frames:
    joints = frame.prepare_frame(anim.hierarchy, anim.base_frame)
    frame_buffer = TextureBuffer(joints, internal_format=GL.GL_RGBA32F)
    frames.append(frame_buffer)


aspect = float(window_size[0]) / float(window_size[1])
projection = Matrix44.perspective_projection(90., aspect, 1., 100., np.float32)
model_view = Matrix44.from_translation([0.,-30.,-50.], np.float32)
model_view = Matrix44.from_x_rotation(np.pi / 2.) * model_view

GL.glClearColor(0.2, 0.2, 0.2, 1.0)
GL.glEnable(GL.GL_DEPTH_TEST)
GL.glDisable(GL.GL_CULL_FACE)

start = time.clock()
last = start

glfw.MakeContextCurrent(window)
while not glfw.WindowShouldClose(window):
    current = time.clock()
    total = current - start
    delta = current - last

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    rotation = Matrix44.from_y_rotation(math.pi * delta, np.float32)
    #model_view = rotation * model_view

    frame = frames[int(total * 30) % len(frames)]
    mesh.render(in_projection=projection, in_model_view=model_view, in_frame=frame)

    glfw.SwapBuffers(window)
    glfw.PollEvents()

    last = current


glfw.DestroyWindow(window)

glfw.Terminate()
exit()


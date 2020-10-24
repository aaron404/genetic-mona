import ctypes
import os
import sys
import time

import numpy as np

import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
from OpenGL.arrays import vbo

from PIL import Image

import pdb

vert_shader = """
#version 450
layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;

out vec4 color_out;

void main()
{
   gl_Position = vec4(position.xyz *2 - 1, 1);
   color_out = color;
   color_out.w = 1.0;
}
"""

frag_shader = """
#version 450
layout(location = 0) uniform sampler2D image;
layout(location = 1) uniform vec2 resolution;
layout(location = 2) uniform int draw_diff;
smooth in vec4 color_out;

out vec4 outputColor;

void main()
{
    outputColor = color_out - texture(image, gl_FragCoord.xy / resolution) * draw_diff;
    outputColor.w = 1.0;
    outputColor *= outputColor;
}
"""


copy_vert_shader = """
#version 450
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 iTexCoords;

out vec2 oTexCoords;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    oTexCoords = iTexCoords;
}
"""
copy_frag_shader = """
#version 450

out vec4 fragColor;
in vec2 iTexCoords;
uniform sampler2D screenTexture;
uniform vec2 iResolution;

void main() {
    fragColor = texture(screenTexture, gl_FragCoord.xy / iResolution);
    //fragColor.x = gl_FragCoord.x / iResolution.x;
}
"""

class Controller():

    def __init__(self, image=None, scale=1):

        self.population_size = 30
        self.elite_pool_size = 1
        self.mutation_rate   = 0.0005
        self.mutation_amount = 0.05

        self.num_tris = 50
        self.tris_per_pop = self.num_tris * self.population_size
        self.verts_per_pop = self.tris_per_pop * 3
        self.floats_per_pop = self.verts_per_pop * 6    # (2 for position, 4 for RGBA)
        self.verts_per_gene = self.num_tris * 3
        self.floats_per_gene = self.verts_per_gene * 6

        self.vert_data = np.random.rand(self.floats_per_pop).astype(np.float32)
        print(self.vert_data.size)


        # State info
        self.img = image
        self.w = self.img.width
        self.h = self.img.height
        self.frame_count = 0
        self.best_gene = 0
        self.best_score = 1000000000000

        # GL stuff #
        # shader to draw triangles + image to framebuffer
        self.shader = shaders.compileProgram(
            shaders.compileShader(vert_shader, gl.GL_VERTEX_SHADER),
            shaders.compileShader(frag_shader, gl.GL_FRAGMENT_SHADER)
        )
        # shader to copy offscreen framebuffer to main framebuffer
        self.copy_shader = shaders.compileProgram(
            shaders.compileShader(copy_vert_shader, gl.GL_VERTEX_SHADER),
            shaders.compileShader(copy_frag_shader, gl.GL_FRAGMENT_SHADER)
        )

        # shader attribute indices
        self.pos_id = gl.glGetAttribLocation(self.shader, b'position')
        self.col_id = gl.glGetAttribLocation(self.shader, b'color')
        self.img_id = gl.glGetUniformLocation(self.shader, b'image')
        self.res_id = gl.glGetUniformLocation(self.shader, b"resolution")
        self.dif_id = gl.glGetUniformLocation(self.shader, b"draw_diff")

        # create texture
        self.texture_id = gl.glGenTextures(1)

        # create framebuffer for off-screen rendering
        self.frame_buffer = None
        self._init_texture()
        self._init_frame_buffer()
        self._init_vertex_buffers()
        #self._init_image()

        self._init_gl()
        self.t0 = time.time()

    def _init_texture(self):
        img = self.img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.convert("RGBA")
        img_data = np.array(list(img.getdata()), dtype=np.uint8)

        self.image_texture = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.image_texture)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)

        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.w, self.h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


    def _init_frame_buffer(self):
        """Initialize the framebuffer used for offscreen rendering"""

        self.frame_buffer = gl.glGenFramebuffers(1)
        color_buffer = gl.glGenRenderbuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frame_buffer)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, color_buffer)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA, self.w, self.h)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER,
                                    gl.GL_COLOR_ATTACHMENT0,
                                    gl.GL_RENDERBUFFER,
                                    color_buffer)

        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            print("Error initializing framebuffer")
            exit()

        gl.glViewport(0, 0, self.w, self.h)

    def _init_vertex_buffers(self):

        self.vert_data_buffer = gl.glGenBuffers(1)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vert_data_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vert_data.nbytes, (gl.GLfloat * (self.vert_data.size))(*self.vert_data), gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        self.quad_vao   = gl.glGenVertexArrays(1)
        self.quad_vbo   = gl.glGenBuffers(1)
        # array of (screen coord, uv coord) for two triangles
        quad_vertices = np.array([
            (-1.0, -1.0, 0.0, 0.0),
            (-1.0,  1.0, 0.0, 1.0),
            ( 1.0,  1.0, 1.0, 1.0),
            ( 1.0,  1.0, 1.0, 1.0),
            ( 1.0, -1.0, 1.0, 0.0),
            (-1.0, -1.0, 0.0, 0.0),
        ], dtype=np.float32)
        gl.glBindVertexArray(self.quad_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.quad_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, None) # 16 byte stride (4x float)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, 8) # 16 byte stride + 8 byte offset at start


    def _init_gl(self):
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_DST_ALPHA)
        gl.glViewport(0, 0, glut.glutGet(glut.GLUT_WINDOW_WIDTH), glut.glutGet(glut.GLUT_WINDOW_HEIGHT))

    def resize(self, width, height):
        self.w = width
        self.h = height

    def _get_fitness(self, index):
        gl.glViewport(0, 0, self.w, self.h)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frame_buffer)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glUseProgram(self.shader)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vert_data_buffer)
        gl.glEnableVertexAttribArray(self.pos_id)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(self.col_id)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, False, 0, ctypes.c_void_p(0))

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.image_texture)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glUniform1i(self.img_id, 0)
        gl.glUniform1i(self.dif_id, 1)
        gl.glUniform2f(self.res_id, self.w, self.h)

        gl.glDrawArrays(gl.GL_TRIANGLES, index * self.verts_per_gene, self.verts_per_gene)
        gl.glDisableVertexAttribArray(self.col_id)
        gl.glDisableVertexAttribArray(self.pos_id)
        gl.glUseProgram(0)

        # gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.frame_buffer)
        # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
        # gl.glClearColor(0, 0, 0, 1)
        # gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # gl.glBlitFramebuffer(0, 0, self.w, self.h, 0, 0, self.w, self.h, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)

        gl.glFlush()

        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        data = gl.glReadPixels (0, 0, self.w, self.h, gl.GL_RGB,  gl.GL_UNSIGNED_BYTE)
        fitness = np.frombuffer(data, dtype=np.uint8).sum()

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        return fitness

    def _mutate(self, index):
        """Apply mutations to the gene specified by index"""
        best = self.vert_data[self.best_gene * self.floats_per_gene:(self.best_gene + 1) * self.floats_per_gene]
        mutation = (np.random.rand(self.floats_per_gene).astype(np.float32) - 1.0) * 2 * self.mutation_amount * 0.01 # random between +/- mutation amount
        mutation *= np.random.rand(self.floats_per_gene) > self.mutation_rate # multiply by mutation mask
        self.vert_data[index*self.floats_per_gene:(index+1)*self.floats_per_gene] = best + mutation

    def step(self):
        # get fitness scores for population
        fitness_scores = sorted([(i, self._get_fitness(i)) for i in range(self.population_size)], key=lambda x: x[1])

        self.best_gene = fitness_scores[0][0]
        best_score = fitness_scores[0][1]
        if best_score < self.best_score:
            self.best_score = best_score
            print(self.best_gene, self.best_score)
            glut.glutPostRedisplay()

        # mutate the peasants
        for i in range(self.elite_pool_size, self.population_size):
            self._mutate(fitness_scores[i][0])
        self.vert_data = np.clip(self.vert_data, 0, 1)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vert_data_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vert_data.nbytes, (gl.GLfloat * self.vert_data.size)(*self.vert_data), gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)


    def draw(self):
        gl.glViewport(0, 0, self.w, self.h)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frame_buffer)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glUseProgram(self.shader)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vert_data_buffer)
        gl.glEnableVertexAttribArray(self.pos_id)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(self.col_id)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, False, 0, ctypes.c_void_p(0))

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.image_texture)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glUniform1i(self.img_id, 0)
        gl.glUniform1i(self.dif_id, 0)
        gl.glUniform2f(self.res_id, self.w, self.h)

        gl.glDrawArrays(gl.GL_TRIANGLES, self.best_gene * self.verts_per_gene, self.verts_per_gene)
        gl.glDisableVertexAttribArray(self.col_id)
        gl.glDisableVertexAttribArray(self.pos_id)
        gl.glUseProgram(0)

        #gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        #gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.frame_buffer)
        #gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
        #gl.glClearColor(0, 0, 0, 1)
        #gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBlitNamedFramebuffer(self.frame_buffer, 0, 0, 0, self.w, self.h, 0, 0, self.w, self.h, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)

        gl.glFlush()

        self.frame_count += 1

        # float colors python sum 83.90426205358702
        # float colors numpy sum  399.39343067326814


def idle_cb():
    controller.step()
    #glut.glutPostRedisplay()

def keyboard_cb(key, x, y):
    if key == b'q':
        exit()
    pass

def reshape_cb(w, h):
    gl.glViewport(0, 0, glut.glutGet(glut.GLUT_WINDOW_WIDTH), glut.glutGet(glut.GLUT_WINDOW_HEIGHT))
    controller.resize(w, h)

if __name__ == "__main__":
    w, h = (960, 540)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("Invalid image specified")
        exit()
    image = Image.open(image_path)

    try:
        scale = float(sys.argv[2])
        scale = abs(scale)
    except ValueError:
        print("Scale must be float")
        exit()
    image = image.resize((int(image.width * scale), int(image.height * scale)))
    w, h = image.size

    # Initialize GLUT library
    glut.glutInit()

    # Setup window
    screen_width = glut.glutGet(glut.GLUT_SCREEN_WIDTH)
    screen_height = glut.glutGet(glut.GLUT_SCREEN_HEIGHT)

    glut.glutInitWindowSize(w, h)
    glut.glutInitWindowPosition(0, 0)
    glut.glutInitDisplayMode(glut.GLUT_SINGLE | glut.GLUT_RGB)
    glut.glutCreateWindow(b'glMona')

    controller = Controller(image)

    # Register event callbacks
    glut.glutIdleFunc(idle_cb)
    glut.glutDisplayFunc(controller.draw)
    glut.glutKeyboardFunc(keyboard_cb)
    glut.glutReshapeFunc(reshape_cb)

    glut.glutMainLoop()

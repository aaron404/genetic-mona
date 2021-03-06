import ctypes
import os
import random
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

triangle_shader = {
    "vert": """
#version 450
layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;
out vec4 color_out;
void main() {
    gl_Position = vec4(position.xyz * 2 - 1, 1);
    color_out = color;
    color_out.w = 1.0;
}
""",
    "frag": """
#version 450
smooth in vec4 color_out;
out vec4 outputColor;
void main() {
    outputColor = color_out;
    outputColor.w = 0.25;
}
""",
}

ssd_shader = {
    "vert": """
#version 450
layout(location = 0) in vec4 position;
void main() {
    gl_Position = vec4(position.xyz * 2 - 1, 1);
}
""",
    "frag": """
#version 450
layout(location = 0) uniform vec2 resolution;
layout(location = 1) uniform sampler2D frame_texture;
layout(location = 2) uniform sampler2D image_texture;
out vec4 outputColor;
void main() {
    vec2 uv  = gl_FragCoord.xy / resolution;
    vec4 image = texture(image_texture, uv);
    vec4 frame = texture(frame_texture, uv);
    outputColor += 0.0 * pow(image - frame, vec4(2.0));
    outputColor = (image - frame);
    outputColor *= outputColor;
    outputColor.w = 1.0;
}
""",
}

class Controller():

    def __init__(self, image=None, scale=1):

        self.population_size = 30
        self.elite_pool_size = 1
        self.random_pool_size = 2
        self.crossover_rate  = 0.75
        self.mutation_rate   = 0.25
        self.mutation_amount = 0.025

        self.num_tris = 64

        # useful numbers to have
        self.tris_per_gene   = self.num_tris
        self.tris_per_pop    = self.num_tris * self.population_size
        self.verts_per_gene  = self.tris_per_gene * 3
        self.verts_per_pop   = self.tris_per_pop  * 3
        self.floats_per_gene = self.verts_per_gene * 6    # (2 for position, 4 for RGBA)
        self.floats_per_pop  = self.verts_per_pop  * 6
 
        self.vert_data = np.random.rand(self.floats_per_pop).astype(np.float32)
        for i in range(self.floats_per_pop):
            if (i % 6 == 5):
                self.vert_data[i] = 0.1
        print(self.vert_data.size)

        self.verts_by_gene = self.vert_data.reshape((self.population_size, self.num_tris, 3, 6))

        # State info
        self.img = image
        self.w = self.img.width
        self.h = self.img.height
        self.frame_count = 0
        self.gen_count = 0
        self.best_gene = 0
        self.best_score = 1000000000000

        # GL stuff #
        # shader to draw triangles to framebuffer
        self.triangle_shader = shaders.compileProgram(
            shaders.compileShader(triangle_shader["vert"], gl.GL_VERTEX_SHADER),
            shaders.compileShader(triangle_shader["frag"], gl.GL_FRAGMENT_SHADER)
        )
        # shader to draw triangles to framebuffer
        self.ssd_shader = shaders.compileProgram(
            shaders.compileShader(ssd_shader["vert"], gl.GL_VERTEX_SHADER),
            shaders.compileShader(ssd_shader["frag"], gl.GL_FRAGMENT_SHADER)
        )

        # shader attribute indices
        self.pos_id = gl.glGetAttribLocation(self.triangle_shader, b'position')
        self.col_id = gl.glGetAttribLocation(self.triangle_shader, b'color')
        self.image_id = gl.glGetUniformLocation(self.ssd_shader, b'image_texture')
        self.frame_id = gl.glGetUniformLocation(self.ssd_shader, b'frame_texture')
        self.res_id = gl.glGetUniformLocation(self.ssd_shader, b"resolution")

        # create texture
        self.texture_id = gl.glGenTextures(1)

        # create framebuffer for off-screen rendering
        self.frame_buffer = None
        self._init_buffers()
        self._init_vertex_buffers()
        #self._init_image()

        self._init_gl()
        self.t0 = time.time()

    def _get_image_data(self):
        # TODO: RGB instead of RGBA, maybe float datatype?
        img = self.img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.convert("RGBA")
        return np.array(list(img.getdata()), dtype=np.uint8)

    def _init_buffers(self):
        self.image_texture   = gl.glGenTextures(1)     # texture storing input image
        self.frame_buffer    = gl.glGenFramebuffers(1) # framebuffer for offscreen render
        self.frame_buffer2   = gl.glGenFramebuffers(1) # second offscreen buffer for ping-ponging
        self.render_texture  = gl.glGenTextures(1)     # texture for storing offscreen render
        self.render_texture2 = gl.glGenTextures(1)     # second offscreen texture for ping-ponging
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.image_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.w, self.h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, self._get_image_data())
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.render_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.w, self.h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.render_texture2)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.w, self.h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # attach texture to currently bound framebuffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frame_buffer)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.render_texture, 0)

        # attach texture to currently bound framebuffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frame_buffer2)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.render_texture2, 0)

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
        #gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_DST_ALPHA)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glViewport(0, 0, glut.glutGet(glut.GLUT_WINDOW_WIDTH), glut.glutGet(glut.GLUT_WINDOW_HEIGHT))

    def resize(self, width, height):
        self.w = width
        self.h = height

    def _get_fitness(self, index):
        gl.glViewport(0, 0, self.w, self.h)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frame_buffer)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # draw triangles to framebuffer
        gl.glUseProgram(self.triangle_shader)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vert_data_buffer)
        gl.glEnableVertexAttribArray(self.pos_id)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(self.col_id)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, False, 24, ctypes.c_void_p(8))

        gl.glDrawArrays(gl.GL_TRIANGLES, index * self.verts_per_gene, self.verts_per_gene)
        gl.glDisableVertexAttribArray(self.col_id)
        gl.glDisableVertexAttribArray(self.pos_id)
        gl.glUseProgram(0) # TODO: remove

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frame_buffer2)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glUseProgram(self.ssd_shader)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.render_texture)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.image_texture)
        gl.glUniform1i(self.frame_id, 0)
        gl.glUniform1i(self.image_id, 1)
        gl.glUniform2f(self.res_id, self.w, self.h)

        # TODO: use vertex array
        gl.glBindVertexArray(self.quad_vao)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex4f(0, 0, 0, 1)
        gl.glVertex4f(0, 1, 0, 1)
        gl.glVertex4f(1, 1, 0, 1)
        gl.glVertex4f(1, 0, 0, 1)
        gl.glEnd()
        #gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glUseProgram(0)

        gl.glFlush()

        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        # TODO: try floats?
        data = gl.glReadPixels (0, 0, self.w, self.h, gl.GL_RGB,  gl.GL_FLOAT)
        fitness = np.frombuffer(data, dtype=np.uint8).sum()

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        return fitness

    def _mutate_perturb(self, index):
        """Apply subtle mutations to the gene specified by index"""
        best = self.vert_data[self.best_gene * self.floats_per_gene:(self.best_gene + 1) * self.floats_per_gene]
        mutation = (np.random.rand(self.floats_per_gene).astype(np.float32) - 0.5) * 2 * self.mutation_amount# random between +/- mutation amount
        mutation *= np.random.rand(self.floats_per_gene) < self.mutation_rate # multiply by mutation mask
        self.vert_data[index*self.floats_per_gene:(index+1)*self.floats_per_gene] = best + mutation

    def _mutate_random(self, gene):
        """Fully randomize the gene"""
        #self.vert_data[gene * self.floats_per_gene:(gene + 1) * self.floats_per_gene] = np.random.rand(self.floats_per_gene)
        self.verts_by_gene[gene] = np.random.rand(self.floats_per_gene).reshape((self.tris_per_gene, 3, 6))

    def _mutate_crossover_1p(self, index):
        """A crossover point is randomly selected. Use parent1 gene before this point and parent2 gene after"""
       #crossover_point = random.randint(0, self.floats_per_gene)
        crossover_point = random.randint(0, self.tris_per_gene)
        parent1 = random.randint(0, self.population_size - 1)
        parent2 = random.randint(0, self.population_size - 1)
        child = index
        self.verts_by_gene[child][:crossover_point] = self.verts_by_gene[index][:crossover_point]
        self.verts_by_gene[child][crossover_point:] = self.verts_by_gene[index][crossover_point:]

    def _mutate_crossover_random(self, index):
        """Randomly choose from parent1 and parent2 traits"""
        parent1 = random.randint(0, self.population_size - 1)
        parent2 = random.randint(0, self.population_size - 1)
        mask    = np.random.rand(self.tris_per_gene)

        self.verts_by_gene[index][mask > 0.5] = self.verts_by_gene[parent1][mask > 0.5]
        self.verts_by_gene[index][mask < 0.5] = self.verts_by_gene[parent2][mask < 0.5]

    def step(self):
        # get fitness scores for population
        fitness_scores = sorted([(i, self._get_fitness(i)) for i in range(self.population_size)], key=lambda x: x[1])
        self.best_gene = fitness_scores[0][0]
        best_score = fitness_scores[0][1]
        if best_score < self.best_score:
            self.best_score = best_score

        # perform mutations on all the genes except for the elites        
        for i in range(self.elite_pool_size, self.population_size):
            index = fitness_scores[i][0]
            if random.random() < self.crossover_rate:
                if random.random() < 0.5:
                    self._mutate_crossover_1p(index)
                else:
                    self._mutate_crossover_random(index)
            else:
                if random.random() < 0.95:
                    self._mutate_perturb(index)
                else:
                    self._mutate_random(index)
        
        # clamp values to [0,1]
        self.vert_data = np.clip(self.vert_data, 0, 1)

        # update data buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vert_data_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vert_data.nbytes, (gl.GLfloat * self.vert_data.size)(*self.vert_data), gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        self.gen_count += 1
        if self.gen_count % 100 == 0:
            glut.glutSetWindowTitle(u"glMona - {}".format(self.gen_count / (time.time() - self.t0)))

        glut.glutPostRedisplay()


    def draw(self):
        gl.glViewport(0, 0, self.w, self.h)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # draw triangles to framebuffer
        gl.glUseProgram(self.triangle_shader)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vert_data_buffer)
        gl.glEnableVertexAttribArray(self.pos_id)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(self.col_id)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, False, 24, ctypes.c_void_p(8))

        gl.glDrawArrays(gl.GL_TRIANGLES, self.best_gene * self.verts_per_gene, self.verts_per_gene)
        gl.glDisableVertexAttribArray(self.col_id)
        gl.glDisableVertexAttribArray(self.pos_id)
        gl.glUseProgram(0)

        gl.glFlush()

        self.frame_count += 1


def idle_cb():
    controller.step()

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

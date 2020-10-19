import ctypes
import os
import sys

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
}
"""

frag_shader = """
#version 450
smooth in vec4 color_out;

out vec4 outputColor;

void main()
{
   outputColor = color_out;
}
"""

class Controller():
    
    def __init__(self, image_path="mona.jpg", scale=1):

        self.num_tris = 50
        self.vert_data = np.random.rand(self.num_tris * 6 * 3).astype(np.float32)
        self.frame_count = 0

        self._init_vertex_buffers()
        self.shader = shaders.compileProgram(
            shaders.compileShader(vert_shader, gl.GL_VERTEX_SHADER),
            shaders.compileShader(frag_shader, gl.GL_FRAGMENT_SHADER)
        )

        self.pos_id = gl.glGetAttribLocation(self.shader, b'position')
        self.col_id = gl.glGetAttribLocation(self.shader, b'color')

        # create texture
        self.texture_id = gl.glGenTextures(1)
        self.img = Image.open("mona.jpg")
        self.img_data = self.img.tobytes()
        self.w = self.img.width * scale
        self.h = self.img.height * scale

        # create framebuffer for off-screen rendering
        self.frame_buffer = None
        #self._init_frame_buffer()

        self._init_gl()

    def _init_frame_buffer(self):
        
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
        

    def _init_vertex_buffers(self):

        self.vert_data_buffer = gl.glGenBuffers(1)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vert_data_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vert_data.nbytes, (gl.GLfloat * self.vert_data.size)(*self.vert_data), gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
       
    def _init_gl(self):
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_DST_ALPHA)
        gl.glViewport(0, 0, glut.glutGet(glut.GLUT_WINDOW_WIDTH), glut.glutGet(glut.GLUT_WINDOW_HEIGHT))
        
    def draw(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glUseProgram(self.shader)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vert_data_buffer)
        gl.glEnableVertexAttribArray(self.pos_id)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(self.col_id)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, False, 0, ctypes.c_void_p(0))
  
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.num_tris * 3)
        gl.glDisableVertexAttribArray(self.col_id)
        gl.glDisableVertexAttribArray(self.pos_id)
        gl.glUseProgram(0)

        gl.glFlush()

        self.frame_count += 1

def idle_cb():
    glut.glutPostRedisplay()

def keyboard_cb(key, x, y):
    if key == b'q':
        exit()
    pass

def reshape_cb(w, h):
    gl.glViewport(0, 0, glut.glutGet(glut.GLUT_WINDOW_WIDTH), glut.glutGet(glut.GLUT_WINDOW_HEIGHT))

if __name__ == "__main__":
    w, h = (960, 540)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("Invalid image specified")
        exit()
    
    try:
        scale = int(sys.argv[2])
    except ValueError:
        print("Scale must be positive integer")
        exit()

    # Initialize GLUT library
    glut.glutInit()

    # Setup window
    screen_width = glut.glutGet(glut.GLUT_SCREEN_WIDTH)
    screen_height = glut.glutGet(glut.GLUT_SCREEN_HEIGHT)
    window_width = w
    window_height = h

    glut.glutInitWindowSize(w, h)
    glut.glutInitWindowPosition(0, 0)
    glut.glutInitDisplayMode(glut.GLUT_SINGLE | glut.GLUT_RGB)
    glut.glutCreateWindow(b'glMona')

    controller = Controller(image_path, scale)

    # Register event callbacks
    glut.glutIdleFunc(idle_cb)
    glut.glutDisplayFunc(controller.draw)
    glut.glutKeyboardFunc(keyboard_cb)
    glut.glutReshapeFunc(reshape_cb)
    
    glut.glutMainLoop()

"""DIAG-GT7-905. Render each comparable capture via one shared shader on Mesa EGL.

Run python render_check.py <shaders-directory> on Linux with desktop EGL/OpenGL.
A 1x1 float attachment keeps output values independent of display color conversion.
"""
import argparse
import ctypes as ct
from pathlib import Path
import struct

from probe import cases


def bind(lib, name, result, *args):
    fn = getattr(lib, name)
    fn.restype = result
    fn.argtypes = list(args)
    return fn


INT = ct.c_int
UINT = ct.c_uint
PTR = ct.c_void_p
INTS = ct.POINTER(INT)
UINTS = ct.POINTER(UINT)


def surfaceless_context():
    egl = ct.CDLL('libEGL.so.1')
    display = bind(egl, 'eglGetPlatformDisplay', PTR, UINT, PTR, PTR)(0x31dd, None, None)
    if not bind(egl, 'eglInitialize', UINT, PTR, INTS, INTS)(display, None, None):
        raise RuntimeError('Surfaceless EGL display unavailable')
    if not bind(egl, 'eglBindAPI', UINT, UINT)(0x30a2):
        raise RuntimeError('Desktop OpenGL unavailable')
    config, count = PTR(), INT()
    attrs = (INT * 5)(0x3033, 1, 0x3040, 8, 0x3038)  # pbuffer, desktop GL
    choose = bind(egl, 'eglChooseConfig', UINT, PTR, INTS, ct.POINTER(PTR), INT, INTS)
    if not choose(display, attrs, ct.byref(config), 1, ct.byref(count)) or not count.value:
        raise RuntimeError('No matching EGL config')
    context = bind(egl, 'eglCreateContext', PTR, PTR, PTR, PTR, INTS)(
        display, config, None, (INT * 5)(0x3098, 4, 0x30fb, 5, 0x3038))
    if not context or not bind(egl, 'eglMakeCurrent', UINT, PTR, PTR, PTR, PTR)(
            display, None, None, context):
        raise RuntimeError('OpenGL 4.5 context unavailable')
    return egl, display


def check(root):
    stats, inputs = cases(root)
    if not inputs:
        raise ValueError('No comparable captured translations')
    egl, display = surfaceless_context()
    try:
        draw_cases(inputs)
    finally:
        bind(egl, 'eglTerminate', UINT, PTR)(display)
    print(dict(stats))


def draw_cases(inputs):
    gl = ct.CDLL('libGL.so.1')
    def call(name, result, *args):
        return bind(gl, 'gl' + name, result, *args)

    print(call('GetString', ct.c_char_p, UINT)(0x1f01).decode())
    program = call('CreateProgram', UINT)()
    vertex = b"""#version 450 core
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(2.0 * p - 1.0, 0.0, 1.0);
    }"""
    fragment = Path(__file__).with_name('constant_export.frag').read_bytes()
    status = INT()
    for stage, source in [(0x8b31, vertex), (0x8b30, fragment)]:
        shader = call('CreateShader', UINT, UINT)(stage)
        src = ct.c_char_p(source)
        call('ShaderSource', None, UINT, INT, ct.POINTER(ct.c_char_p), INTS)(shader, 1, ct.byref(src), None)
        call('CompileShader', None, UINT)(shader)
        call('GetShaderiv', None, UINT, UINT, INTS)(shader, 0x8b81, ct.byref(status))
        if not status.value:
            log = ct.create_string_buffer(8192)
            call('GetShaderInfoLog', None, UINT, INT, INTS, PTR)(shader, len(log), None, log)
            raise RuntimeError(log.value.decode())
        call('AttachShader', None, UINT, UINT)(program, shader)
    call('LinkProgram', None, UINT)(program)
    call('GetProgramiv', None, UINT, UINT, INTS)(program, 0x8b82, ct.byref(status))
    if not status.value:
        log = ct.create_string_buffer(8192)
        call('GetProgramInfoLog', None, UINT, INT, INTS, PTR)(program, len(log), None, log)
        raise RuntimeError(log.value.decode())
    call('UseProgram', None, UINT)(program)
    location = call('GetUniformLocation', INT, UINT, ct.c_char_p)(program, b'words[0]')
    if location < 0:
        raise RuntimeError('Program input uniform unavailable')

    vao, texture, fbo = UINT(), UINT(), UINT()
    call('GenVertexArrays', None, INT, UINTS)(1, ct.byref(vao))
    call('BindVertexArray', None, UINT)(vao)
    call('CreateTextures', None, UINT, INT, UINTS)(0x0de1, 1, ct.byref(texture))
    call('TextureStorage2D', None, UINT, INT, UINT, INT, INT)(texture, 1, 0x8814, 1, 1)  # RGBA32F
    call('CreateFramebuffers', None, INT, UINTS)(1, ct.byref(fbo))
    call('NamedFramebufferTexture', None, UINT, UINT, UINT, INT)(fbo, 0x8ce0, texture, 0)
    call('BindFramebuffer', None, UINT, UINT)(0x8d40, fbo)
    if call('CheckFramebufferStatus', UINT, UINT)(0x8d40) != 0x8cd5:
        raise RuntimeError('Incomplete float framebuffer')
    call('Viewport', None, INT, INT, INT, INT)(0, 0, 1, 1)

    clear = call('ClearBufferfv', None, UINT, INT, ct.POINTER(ct.c_float))
    upload = call('Uniform4uiv', None, INT, INT, UINTS)
    draw = call('DrawArrays', None, UINT, INT, INT)
    read = call('ReadPixels', None, INT, INT, INT, INT, UINT, UINT, PTR)
    pixel = (ct.c_float * 4)()
    for pgm, words, reference in inputs:
        clear(0x1800, 0, (ct.c_float * 4)(-12345.0, -12345.0, -12345.0, -12345.0))
        upload(location, 4, (UINT * 16)(*words))
        draw(4, 0, 3)
        read(0, 0, 1, 1, 0x1908, 0x1406, pixel)
        if bytes(pixel) != struct.pack('=4f', *reference):
            raise AssertionError(f'{pgm:016x}: {list(pixel)} != {reference}')
    err = call('GetError', UINT)()
    if err:
        raise RuntimeError(f'OpenGL error 0x{err:x}')
    print(f'{len(inputs)} reference pixels matched using one fragment shader')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    check(parser.parse_args().root)

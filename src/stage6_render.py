"""Minimal calibrated SMPL-X mesh rasterization using off-screen EGL."""
from contextlib import contextmanager
import subprocess

import moderngl
import numpy as np

from .virtual_pose import apply_affine, pinhole

VERTEX = """
#version 330
in vec3 camera_point;
in vec2 image_point;
in float fabric;
out vec3 v_camera;
out float v_fabric;
uniform vec2 z_range;
void main() {
    float z = camera_point.z;
    float n = z_range.x;
    float f = z_range.y;
    vec2 ndc = image_point * vec2(2., -2.) + vec2(-1., 1.);
    gl_Position = vec4(ndc * z, (f + n) / (f - n) * z - 2. * f * n / (f - n), z);
    v_camera = camera_point;
    v_fabric = fabric;
}
"""
FRAGMENT = """
#version 330
in vec3 v_camera;
in float v_fabric;
out vec4 color;
uniform vec3 skin;
uniform vec3 shirt;
void main() {
    vec3 n = normalize(cross(dFdx(v_camera), dFdy(v_camera)));
    if (n.z > 0.) n = -n;
    float light = 0.51 + 0.49 * max(0., dot(n, normalize(vec3(-.35,-.45,-1.))));
    color = vec4(mix(skin, shirt, smoothstep(.25,.75,v_fabric)) * light, 1.);
}
"""
PALETTES = {
    "shirt": ((.85, .64, .48), (.22, .35, .49)),
    "clay": ((.72, .72, .72), (.72, .72, .72)),
}


class AvatarRenderer:
    def __init__(self, wh, faces, rest):
        self.wh = tuple(wh)
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.renderer = self.ctx.info["GL_RENDERER"]
        self.fbo = self.ctx.simple_framebuffer(self.wh, components=3)
        self.fbo.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.program = self.ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.index = self.ctx.buffer(np.ascontiguousarray(faces, dtype=np.int32))
        self.update = np.empty((len(rest), 6), dtype=np.float32)
        x, y, _ = rest.T
        self.update[:, 5] = (np.abs(x) < .37) & (y > -.31) & (y < .165)
        self.buffer = self.ctx.buffer(reserve=self.update.nbytes, dynamic=True)
        columns = [(self.buffer, "3f 2f 1f", "camera_point", "image_point", "fabric")]
        self.array = self.ctx.vertex_array(self.program, columns, self.index)

    def frames(self, vertices, translations, camera, palette="shirt", pixel_shift=0):
        xy = apply_affine(pinhole(vertices, translations), camera)
        xy[..., 0] += pixel_shift / self.wh[0]
        depth = vertices + translations[:, None]
        minimum, maximum = float(depth[..., 2].min()), float(depth[...,2].max())
        self.program["z_range"].value = (max(.001, minimum*.95), maximum*1.05)
        self.program["skin"].value, self.program["shirt"].value = PALETTES[palette]
        for camera_xyz, projected in zip(depth, xy):
            self.update[:, :3] = camera_xyz
            self.update[:, 3:5] = projected
            self.buffer.write(self.update)
            self.fbo.clear(.69, .71, .73, 1., depth=1.)
            self.array.render()
            rgb = np.frombuffer(self.fbo.read(components=3, alignment=1), dtype=np.uint8)
            yield np.ascontiguousarray(rgb.reshape(self.wh[1], self.wh[0],3)[::-1,:,::-1])

    def close(self):
        for resource in (self.array, self.buffer, self.index, self.program, self.fbo, self.ctx):
            resource.release()


@contextmanager
def video_writer(path, wh, fps):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{wh[0]}x{wh[1]}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-threads", "2", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        try:
            yield proc.stdin
        finally:
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError(f"Could not write {path}")

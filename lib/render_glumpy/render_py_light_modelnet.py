# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
from glumpy import app, gl, gloo, data, log
import logging

log.setLevel(logging.WARNING)  # ERROR, WARNING, DEBUG, INFO
from lib.pair_matching.RT_transform import quat2mat

vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 position;
attribute vec3 normal;
attribute vec2 texcoord;
varying vec3   v_normal;
varying vec3   v_position;
varying vec2   v_texcoord;
void main()
{
    // Assign varying variables
    v_normal   = normal;
    v_position = position;
    v_texcoord = texcoord;
    // Final position
    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
}
"""


def get_fragment(brightness_ratio=0.4):
    fragment = """
    uniform mat4      u_model;           // Model matrix
    uniform mat4      u_view;            // View matrix
    uniform mat4      u_normal;          // Normal matrix
    uniform mat4      u_projection;      // Projection matrix
    uniform sampler2D u_texture;         // Texture
    uniform vec3      u_light_position;  // Light position
    uniform vec3      u_light_intensity; // Light intensity

    varying vec3      v_normal;          // Interpolated normal (in)
    varying vec3      v_position;        // Interpolated position (in)
    varying vec2      v_texcoord;        // Interpolated fragment texture coordinates (in)
    void main()
    {{
        // Calculate normal in world coordinates
        vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;

        // Calculate the location of this fragment (pixel) in world coordinates
        vec3 position = vec3(u_view*u_model * vec4(v_position, 1));
        //vec3 light_position = vec3(u_view*u_model * vec4(u_light_position, 1));
        // Calculate the vector from this pixels surface to the light source
        vec3 surfaceToLight = u_light_position - position;

        // Calculate the cosine of the angle of incidence (brightness)
        float brightness = dot(normal, surfaceToLight) /
                          (length(surfaceToLight) * length(normal));
        brightness = max(min(brightness,1.0),0.0);

        // Calculate final color of the pixel, based on:
        // 1. The angle of incidence: brightness
        // 2. The color/intensities of the light: light.intensities
        // 3. The texture and texture coord: texture(tex, fragTexCoord)

        // Get texture color
        vec4 t_color = vec4(texture2D(u_texture, v_texcoord).rgb, 1.0);

        // Final color
        gl_FragColor = t_color * (({} + {}*brightness) * vec4(u_light_intensity, 1));
    }}
    """.format(
        1 - brightness_ratio, brightness_ratio
    )
    return fragment


class Render_Py_Light_ModelNet:
    def __init__(
        self,
        model_path,
        texture_path,
        K,
        width=640,
        height=480,
        zNear=0.25,
        zFar=6.0,
        brightness_ratios=[0.7],
    ):
        self.width = width
        self.height = height
        self.zNear = zNear
        self.zFar = zFar
        self.K = K
        self.model_path = model_path

        log.info("Loading mesh")
        vertices, indices = data.objload("{}".format(model_path), rescale=True)
        vertices["position"] = vertices["position"] / 10.0

        self.render_kernels = []
        for brightness_ratio in brightness_ratios:
            fragment = get_fragment(brightness_ratio=brightness_ratio)
            render_kernel = gloo.Program(vertex, fragment)
            render_kernel.bind(vertices)

            log.info("Loading texture")
            render_kernel["u_texture"] = np.copy(
                data.load("{}".format(texture_path))[::-1, :, :]
            )

            render_kernel["u_model"] = np.eye(4, dtype=np.float32)
            u_projection = self.my_compute_calib_proj(K, width, height, zNear, zFar)
            render_kernel["u_projection"] = np.copy(u_projection)

            render_kernel["u_light_intensity"] = 1, 1, 1
            self.render_kernels.append(render_kernel)
        self.brightness_k = 0  # init

        self.window = app.Window(width=width, height=height, visible=False)

        @self.window.event
        def on_draw(dt):
            self.window.clear()
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            # print('brightness_k', self.brightness_k) # this function runs when running app.run()
            self.render_kernels[self.brightness_k].draw(gl.GL_TRIANGLES)

        @self.window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)

    def render(
        self, r, t, light_position, light_intensity, brightness_k=0, r_type="quat"
    ):
        """
        :param r:
        :param t:
        :param light_position:
        :param light_intensity:
        :param brightness_k: choose which brightness in __init__
        :param r_type:
        :return:
        """
        if r_type == "quat":
            R = quat2mat(r)
        elif r_type == "mat":
            R = r
        self.brightness_k = brightness_k
        self.render_kernels[brightness_k]["u_view"] = self._get_view_mtx(R, t)
        self.render_kernels[brightness_k]["u_light_position"] = light_position
        self.render_kernels[brightness_k]["u_normal"] = np.array(
            np.matrix(
                np.dot(
                    self.render_kernels[brightness_k]["u_view"].reshape(4, 4),
                    self.render_kernels[brightness_k]["u_model"].reshape(4, 4),
                )
            ).I.T
        )
        self.render_kernels[brightness_k]["u_light_intensity"] = light_intensity

        app.run(framecount=0)
        rgb_buffer = np.zeros((self.height, self.width, 4), dtype=np.float32)
        gl.glReadPixels(
            0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_FLOAT, rgb_buffer
        )

        rgb_gl = np.copy(rgb_buffer)
        rgb_gl.shape = 480, 640, 4
        rgb_gl = rgb_gl[::-1, :]
        rgb_gl = np.round(rgb_gl[:, :, :3] * 255).astype(
            np.uint8
        )  # Convert to [0, 255]
        bgr_gl = rgb_gl[:, :, [2, 1, 0]]

        depth_buffer = np.zeros((self.height, self.width), dtype=np.float32)
        gl.glReadPixels(
            0,
            0,
            self.width,
            self.height,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            depth_buffer,
        )
        depth_gl = np.copy(depth_buffer)
        depth_gl.shape = 480, 640
        depth_gl = depth_gl[::-1, :]
        depth_bg = depth_gl == 1
        depth_gl = (
            2
            * self.zFar
            * self.zNear
            / (self.zFar + self.zNear - (self.zFar - self.zNear) * (2 * depth_gl - 1))
        )
        depth_gl[depth_bg] = 0
        return bgr_gl, depth_gl

    def __del__(self):
        self.window.close()

    def my_compute_calib_proj(self, K, w, h, zNear, zFar):
        u0 = K[0, 2] + 0.5
        v0 = K[1, 2] + 0.5
        fu = K[0, 0]
        fv = K[1, 1]
        L = +(u0) * zNear / -fu
        T = +(v0) * zNear / fv
        R = -(w - u0) * zNear / -fu
        B = -(h - v0) * zNear / fv
        proj = np.zeros((4, 4))
        proj[0, 0] = 2 * zNear / (R - L)
        proj[1, 1] = 2 * zNear / (T - B)
        proj[2, 2] = -(zFar + zNear) / (zFar - zNear)
        proj[2, 0] = (R + L) / (R - L)
        proj[2, 1] = (T + B) / (T - B)
        proj[2, 3] = -1.0
        proj[3, 2] = -(2 * zFar * zNear) / (zFar - zNear)
        return proj

    def _get_view_mtx(self, R, t):
        u_view = np.eye(4, dtype=np.float32)
        u_view[:3, :3], u_view[:3, 3] = R, t.squeeze()
        yz_flip = np.eye(4, dtype=np.float32)
        yz_flip[1, 1], yz_flip[2, 2] = -1, -1
        u_view = yz_flip.dot(u_view)  # OpenCV to OpenGL camera system
        u_view = u_view.T  # OpenGL expects column-wise matrix format
        return u_view

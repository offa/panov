# panov - A viewer for 360° equirectangular panoramas
#
# Copyright (C) 2026  offa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Image helpers, GLSL sources and the NumPy panorama renderer.

Shader strings and raw GL enums are plain text and ints, so this module is
importable without an OpenGL context.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtGui import QImage

if TYPE_CHECKING:
    from .camera import Camera

# Raw GL enums, avoids adding a PyOpenGL dependency just for constants.
GL_TRIANGLE_STRIP = 0x0005
GL_COLOR_BUFFER_BIT = 0x4000
GL_TEXTURE0 = 0x84C0
GL_RENDERER = 0x1F01

# Renderers that mean "OpenGL, but on the CPU" -- usually slower than NumPy.
SOFTWARE_RENDERERS = (
    "llvmpipe",
    "softpipe",
    "swrast",
    "gdi generic",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def qimage_to_array(image: QImage) -> np.ndarray:
    """Return an RGB888 (H, W, 3) uint8 array that OWNS its memory.

    ``QImage.constBits()`` only exposes a pointer that stays valid while the
    QImage lives, and ``np.frombuffer`` does not take ownership.  Since
    ``convertToFormat`` usually allocates a fresh buffer that dies with this
    function's locals, the result must be an explicit copy -- otherwise
    pixel reads would touch freed heap.

    The returned array is contiguous in memory and safe to use after the
    QImage is garbage collected.
    """
    if image.isNull():
        raise ValueError("Cannot convert a null QImage to array")

    converted = image.convertToFormat(QImage.Format.Format_RGB888)
    if converted.isNull():
        raise ValueError(
            f"Could not convert QImage to RGB888 (original format: {image.format()})"
        )

    width, height = converted.width(), converted.height()
    stride = converted.bytesPerLine()  # may be padded to a 4-byte boundary

    view = np.frombuffer(converted.constBits(), dtype=np.uint8)
    expected = stride * height
    if view.size < expected:
        # A QImage whose pixel buffer is shorter than bytesPerLine() * height()
        # is internally inconsistent; there is no safe way to read it, so fail
        # loudly instead of letting the reshape below raise a cryptic error.
        raise ValueError(
            f"QImage pixel buffer too short: {view.size} bytes, "
            f"expected {expected} (stride {stride} x {height} rows)"
        )
    view = view[:expected]

    rows = view.reshape(height, stride)[:, : width * 3]

    owned = np.empty((height, width * 3), dtype=np.uint8)
    owned[:] = rows  # genuine memcpy, no aliasing
    return owned.reshape((height, width, 3))


def _set_uniform_f(program, name: str, value: float) -> None:
    try:
        program.setUniformValue1f(name, float(value))
    except RuntimeError:
        program.setUniformValue(name, float(value))


def _set_uniform_i(program, name: str, value: int) -> None:
    try:
        program.setUniformValue1i(name, int(value))
    except RuntimeError:
        program.setUniformValue(name, int(value))


# --------------------------------------------------------------------------- #
# Software renderer
# --------------------------------------------------------------------------- #


class EquirectRenderer:
    """Renders perspective views of an equirectangular panorama on the CPU.

    Longitude spans the image width ([-180, +180] -> [0, W]) and latitude spans
    the image height ([+90, -90] -> [0, H]).
    """

    def __init__(self, image: QImage) -> None:
        # Keep the QImage alive as well as the array; cheap and removes any
        # doubt about the lifetime of the buffer the array was copied from.
        self._source_image = image

        self._src = qimage_to_array(image)  # (H, W, 3) uint8, owned
        self.height, self.width = self._src.shape[:2]

        # Cached grid of camera-space rays; only depends on size + fov, so it
        # survives yaw/pitch changes (the common case while dragging).
        self._ray_key: tuple[int, int, float] | None = None
        self._rays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        # Persistent output buffer + QImage, allocated once per output size.
        # RGBA8888 is 4 bytes per pixel, so scanlines are always 32-bit
        # aligned (Qt's SIMD blitters require it) and it is Qt's fast path.
        self._target_key: tuple[int, int] | None = None
        self._rgba: np.ndarray | None = None
        self._frame: QImage | None = None

    # -- output surface ---------------------------------------------------- #

    def _target(self, out_w: int, out_h: int) -> tuple[np.ndarray, QImage]:
        if self._target_key != (out_w, out_h) or self._frame is None:
            self._rgba = np.empty((out_h, out_w, 4), dtype=np.uint8)
            self._rgba[..., 3] = 255
            self._frame = QImage(
                self._rgba.data,
                out_w,
                out_h,
                out_w * 4,
                QImage.Format.Format_RGBA8888,
            )
            self._target_key = (out_w, out_h)
        return self._rgba, self._frame

    # -- ray grid ---------------------------------------------------------- #

    def _camera_rays(
        self, out_w: int, out_h: int, fov_deg: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unit direction vectors for every output pixel, in camera space."""
        key = (out_w, out_h, fov_deg)
        if self._ray_key == key and self._rays is not None:
            return self._rays

        aspect = out_w / out_h
        tan_half = math.tan(math.radians(fov_deg * 0.5))

        # Normalised device coordinates of pixel centres, then the pinhole ray.
        xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (2.0 / out_w) - 1.0
        ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (2.0 / out_h) - 1.0

        dx = np.multiply.outer(np.ones(out_h, np.float32), xs * (tan_half * aspect))
        dy = np.multiply.outer(-ys * tan_half, np.ones(out_w, np.float32))
        dz = np.ones((out_h, out_w), np.float32)

        inv = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
        dx *= inv
        dy *= inv
        dz *= inv

        self._ray_key = key
        self._rays = (dx, dy, dz)
        return self._rays

    # -- main entry point -------------------------------------------------- #

    def render(
        self, camera: Camera, out_w: int, out_h: int, *, bilinear: bool = True
    ) -> QImage:
        """Render the panorama from *camera* into an out_w x out_h QImage."""
        out_w = max(2, int(out_w))
        out_h = max(2, int(out_h))

        rays = self._camera_rays(out_w, out_h, round(camera.fov, 4))
        x, y, z = self._rotate_rays(*rays, camera.pitch, camera.yaw)
        u, v = self._equirectangular_coords(x, y, z)

        sampler = self._sample_bilinear if bilinear else self._sample_nearest
        pixels = sampler(self._src, u, v)

        rgba, frame = self._target(out_w, out_h)
        rgba[..., :3] = pixels
        return frame

    @staticmethod
    def _rotate_rays(
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray,
        pitch: float,
        yaw: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotate camera-space rays into world directions."""
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        y1 = dy * cp + dz * sp
        z1 = dz * cp - dy * sp
        x2 = dx * cy + z1 * sy
        z2 = z1 * cy - dx * sy

        return x2, y1, z2

    def _equirectangular_coords(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert world directions to equirectangular texel coordinates."""
        lon = np.arctan2(x, z)
        lat = np.arcsin(np.clip(y, -1.0, 1.0))

        u = (lon * (0.5 / math.pi) + 0.5) * self.width
        v = (0.5 - lat / math.pi) * self.height
        return u, v

    # -- sampling ---------------------------------------------------------- #

    def _sample_nearest(
        self, src: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Fastest path: used while the user is actively moving the camera."""
        np.nan_to_num(u, copy=False)
        np.nan_to_num(v, copy=False)
        np.clip(v, 0.0, self.height - 1.0, out=v)

        ui = np.floor(u).astype(np.intp) % self.width  # horizontal wrap-around
        vi = v.astype(np.intp)
        return src[vi, ui]

    def _sample_bilinear(
        self, src: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Quality path: used for the settled, full-resolution frame."""
        np.nan_to_num(u, copy=False)
        np.nan_to_num(v, copy=False)

        u0, u1, fu = self._bilinear_axis(u, self.width, wrap=True)
        v0, v1, fv = self._bilinear_axis(v, self.height, wrap=False)

        return self._interpolate_bilinear(src, (u0, u1), (v0, v1), (fu, fv))

    @staticmethod
    def _bilinear_axis(
        coords: np.ndarray, size: int, *, wrap: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return neighbouring indices and interpolation fractions for one axis."""
        if wrap:
            i0f = np.floor(coords)
            i0 = i0f.astype(np.intp) % size
            i1 = (i0 + 1) % size  # wraps at the seam
        else:
            np.clip(coords, 0.0, size - 1.0, out=coords)
            i0f = np.floor(coords)
            i0 = i0f.astype(np.intp)
            i1 = np.minimum(i0 + 1, size - 1)

        fraction = (coords - i0f).astype(np.float32)
        return i0, i1, fraction

    @staticmethod
    def _interpolate_bilinear(
        src: np.ndarray,
        u_bounds: tuple[np.ndarray, np.ndarray],
        v_bounds: tuple[np.ndarray, np.ndarray],
        f_weights: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Interpolate the four neighbouring texels."""
        u0, u1 = u_bounds
        v0, v1 = v_bounds
        fu, fv = f_weights

        # Expand weights for broadcasting
        fu, fv = fu[..., None], fv[..., None]

        # top = c00 * (1-fu) + c01 * fu
        top = src[v0, u0] * (1 - fu) + src[v0, u1] * fu
        # bottom = c10 * (1-fu) + c11 * fu
        bottom = src[v1, u0] * (1 - fu) + src[v1, u1] * fu

        # out = top * (1-fv) + bottom * fv
        out = (top * (1 - fv) + bottom * fv).astype(np.float32)

        out += 0.5
        np.clip(out, 0.0, 255.0, out=out)
        return out.astype(np.uint8)


# --------------------------------------------------------------------------- #
# GPU shaders
# --------------------------------------------------------------------------- #

VERT_SRC = """#version 330 core
out vec2 v_ndc;

void main() {
    // Fullscreen triangle strip, corners chosen by gl_VertexID. No VBO needed.
    const vec2 corners[4] = vec2[4](
        vec2(-1.0, -1.0), vec2(1.0, -1.0),
        vec2(-1.0,  1.0), vec2(1.0,  1.0));
    v_ndc = corners[gl_VertexID];
    gl_Position = vec4(v_ndc, 0.0, 1.0);
}
"""

FRAG_SRC = """#version 330 core
in vec2 v_ndc;
out vec4 frag_color;

uniform sampler2D pano;
uniform float tan_half_fov;
uniform float aspect;
uniform float cy, sy, cp, sp;
uniform float u_scale, u_bias;

const float PI  = 3.141592653589793;
const float TAU = 6.283185307179586;

void main() {
    vec3 d = normalize(vec3(v_ndc.x * tan_half_fov * aspect,
                            v_ndc.y * tan_half_fov,
                            1.0));

    float y1 = d.y * cp + d.z * sp;
    float z1 = d.z * cp - d.y * sp;
    float x2 = d.x * cy + z1 * sy;
    float z2 = z1 * cy - d.x * sy;

    float lon = atan(x2, z2);
    float lat = asin(clamp(y1, -1.0, 1.0));

    vec2 uv = vec2((lon / TAU + 0.5) * u_scale + u_bias, 0.5 - lat / PI);

    // uv.x is discontinuous at the +/-180deg meridian: neighbouring pixels
    // differ by u_scale instead of by the true sub-pixel step.  The implicit
    // derivatives the sampler uses would therefore collapse to the coarsest
    // mip level and draw a hard line down the seam.  Undo the jump before
    // handing explicit gradients to textureGrad.
    vec2 ddx = dFdx(uv);
    vec2 ddy = dFdy(uv);
    if (abs(ddx.x) > 0.5 * u_scale) ddx.x -= sign(ddx.x) * u_scale;
    if (abs(ddy.x) > 0.5 * u_scale) ddy.x -= sign(ddy.x) * u_scale;

    frag_color = vec4(textureGrad(pano, uv, ddx, ddy).rgb, 1.0);
}
"""

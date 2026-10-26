#!/usr/bin/env python3

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

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback

import numpy as np
from PySide6.QtCore import QElapsedTimer, QRect, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QImage,
    QOffscreenSurface,
    QOpenGLContext,
    QPainter,
    QSurfaceFormat,
)
from PySide6.QtOpenGL import (
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

NAV_KEYS = frozenset(
    {
        Qt.Key.Key_Left,
        Qt.Key.Key_Right,
        Qt.Key.Key_Up,
        Qt.Key.Key_Down,
        Qt.Key.Key_A,
        Qt.Key.Key_D,
        Qt.Key.Key_W,
        Qt.Key.Key_S,
        Qt.Key.Key_Plus,
        Qt.Key.Key_Equal,
        Qt.Key.Key_Minus,
        Qt.Key.Key_Underscore,
    }
)

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
    "microsoft basic render",
    "mesa offscreen",
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
    converted = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height = converted.width(), converted.height()
    stride = converted.bytesPerLine()  # may be padded to a 4-byte boundary

    view = np.frombuffer(converted.constBits(), dtype=np.uint8)
    expected = stride * height
    if view.size < expected:  # defensive: short buffer from some backends
        view = np.frombuffer(bytes(converted.constBits()), dtype=np.uint8)
    view = view[:expected]

    rows = view.reshape(height, stride)[:, : width * 3]

    owned = np.empty((height, width * 3), dtype=np.uint8)
    owned[:] = rows  # genuine memcpy, no aliasing
    return owned.reshape(height, width, 3)


def wrap180(angle: float) -> float:
    """Wrap an angle in degrees into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


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
# Camera
# --------------------------------------------------------------------------- #


class Camera:
    """Pinhole camera described by yaw, pitch and vertical field of view."""

    MIN_FOV = 15.0
    MAX_FOV = 120.0
    DEFAULT_FOV = 75.0

    def __init__(self, fov: float = DEFAULT_FOV) -> None:
        self.yaw = 0.0  # degrees, positive = looking right
        self.pitch = 0.0  # degrees, positive = looking up
        self.fov = self._clamp_fov(fov)

    # -- mutation ---------------------------------------------------------- #

    def rotate(self, d_yaw: float, d_pitch: float) -> None:
        self.yaw = wrap180(self.yaw + d_yaw)
        self.pitch = max(-90.0, min(90.0, self.pitch + d_pitch))

    def zoom(self, factor: float) -> None:
        """Multiply the field of view; factor < 1 zooms in."""
        self.fov = self._clamp_fov(self.fov * factor)

    def reset(self) -> None:
        self.yaw = 0.0
        self.pitch = 0.0
        self.fov = self.DEFAULT_FOV

    # -- read-only helpers ------------------------------------------------- #

    @property
    def zoom_level(self) -> float:
        """Magnification relative to the default field of view."""
        return self.DEFAULT_FOV / self.fov

    @classmethod
    def _clamp_fov(cls, fov: float) -> float:
        return max(cls.MIN_FOV, min(cls.MAX_FOV, fov))


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

        dx, dy, dz = self._camera_rays(out_w, out_h, round(camera.fov, 4))

        pitch = math.radians(camera.pitch)
        yaw = math.radians(camera.yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # World direction = R_y(yaw) @ R_x(pitch) @ camera_ray
        y1 = dy * cp + dz * sp
        z1 = dz * cp - dy * sp
        x2 = dx * cy + z1 * sy
        z2 = z1 * cy - dx * sy

        # Spherical coordinates -> equirectangular texel coordinates.
        lon = np.arctan2(x2, z2)  # [-pi, pi]
        lat = np.arcsin(np.clip(y1, -1.0, 1.0))  # [-pi/2, pi/2]

        u = (lon * (0.5 / math.pi) + 0.5) * self.width
        v = (0.5 - lat / math.pi) * self.height

        src = self._src
        if bilinear:
            pixels = self._sample_bilinear(src, u, v)
        else:
            pixels = self._sample_nearest(src, u, v)

        rgba, frame = self._target(out_w, out_h)
        rgba[..., :3] = pixels
        return frame

    # -- sampling ---------------------------------------------------------- #

    def _sample_nearest(
        self, src: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Fastest path: used while the user is actively moving the camera."""
        np.nan_to_num(u, copy=False)
        np.nan_to_num(v, copy=False)
        np.clip(v, 0.0, self.height - 1.0, out=v)

        ui = u.astype(np.intp) % self.width  # horizontal wrap-around
        vi = v.astype(np.intp)
        return src[vi, ui]

    def _sample_bilinear(
        self, src: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Quality path: used for the settled, full-resolution frame."""
        np.nan_to_num(u, copy=False)
        np.nan_to_num(v, copy=False)

        u0f = np.floor(u)
        fu = (u - u0f).astype(np.float32)
        u0 = u0f.astype(np.intp) % self.width
        u1 = (u0 + 1) % self.width  # wraps at the seam

        np.clip(v, 0.0, self.height - 1.0, out=v)
        v0f = np.floor(v)
        fv = (v - v0f).astype(np.float32)
        v0 = v0f.astype(np.intp)
        v1 = np.minimum(v0 + 1, self.height - 1)

        fu3 = fu[..., None]
        fv3 = fv[..., None]

        c00 = src[v0, u0].astype(np.float32)
        c01 = src[v0, u1].astype(np.float32)
        c10 = src[v1, u0].astype(np.float32)
        c11 = src[v1, u1].astype(np.float32)

        top = c00 + (c01 - c00) * fu3
        bottom = c10 + (c11 - c10) * fu3
        out = top + (bottom - top) * fv3

        out += 0.5
        np.clip(out, 0.0, 255.0, out=out)
        return out.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Shared viewer behaviour
# --------------------------------------------------------------------------- #


class ViewerMixin:
    """Camera navigation, input handling and overlay drawing.

    Mixed into a QWidget subclass.  Concrete viewers must call
    ``self._init_common(...)`` from their ``__init__`` and provide
    ``self._source`` (a QImage or None).
    """

    NAV_INTERVAL_MS = 16  # ~60 Hz keyboard navigation
    SETTLE_MS = 250  # CPU only: delay before switching to a quality render

    # -- construction ------------------------------------------------------ #

    def _init_common(
        self,
        name: str,
        error: str | None,
        show_info: bool,
        source: QImage | None,
    ) -> None:
        self.cam = Camera()
        self._name = name
        self._error = error
        self._show_info = show_info
        self._source = source
        self._aspect_warning = bool(
            source
            and source.height() > 0
            and abs(source.width() / source.height() - 2.0) > 0.05
        )

        self._dragging = False
        self._keys: set[int] = set()
        self._last_pos = None

        self._nav_timer = QTimer(self)
        self._nav_timer.setInterval(self.NAV_INTERVAL_MS)
        self._nav_timer.timeout.connect(self._on_nav_tick)
        self._nav_clock = QElapsedTimer()
        self._nav_clock.start()

        self._render_ms = 0.0
        self._fps = 0.0
        self._frame_count = 0
        self._fps_clock = QElapsedTimer()
        self._fps_clock.start()

    def _apply_window_common(self, name: str) -> None:
        self.setWindowTitle(f"Panov - {name}")
        self.setMinimumSize(480, 270)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    # -- hooks overridden by the CPU viewer -------------------------------- #

    def _touch(self) -> None:
        """CPU viewer: mark the camera as moving so we render cheaply."""
        timer = getattr(self, "_settle_timer", None)
        if timer is None:
            return
        self._interactive = True
        timer.start()

    @property
    def _backend_label(self) -> str:
        return "cpu"

    # -- painting helpers -------------------------------------------------- #

    def _draw_error(self, painter: QPainter, width: int, height: int) -> None:
        painter.fillRect(QRect(0, 0, width, height), QColor(18, 20, 26))
        painter.setPen(QColor(232, 130, 130))
        font = QFont()
        font.setPointSizeF(13.0)
        painter.setFont(font)
        painter.drawText(
            QRect(0, 0, width, height),
            int(Qt.AlignmentFlag.AlignCenter),
            f"Could not open panorama\n\n{self._error or 'unknown error'}",
        )

    def _tick_fps(self) -> None:
        self._frame_count += 1
        elapsed = self._fps_clock.elapsed()
        if elapsed >= 500:
            self._fps = self._frame_count * 1000.0 / elapsed
            self._frame_count = 0
            self._fps_clock.restart()

    def _draw_status(self, painter: QPainter, width: int, height: int) -> None:
        """Semi-transparent status panel in the top-left corner."""
        if self._source is None:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        font.setPointSizeF(9.5)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        lines = [
            f"file        {self._name}",
            f"source      {self._source.width()} x {self._source.height()} px",
            f"yaw/pitch   {self.cam.yaw:+7.1f}deg  {self.cam.pitch:+6.1f}deg",
            f"fov/zoom    {self.cam.fov:7.1f}deg  {self.cam.zoom_level:5.2f}x",
            f"render      {self._render_ms:6.1f} ms  {self._fps:4.0f} fps",
            f"backend     {self._backend_label}",
        ]
        if self._aspect_warning:
            lines.append("note        source aspect is not 2:1 (may look stretched)")

        pad, left, top = 10, 12, 12
        box_w = max(metrics.horizontalAdvance(line) for line in lines) + 2 * pad
        box_h = metrics.lineSpacing() * len(lines) + 2 * pad

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 12, 16, 175))
        painter.drawRoundedRect(QRect(left, top, box_w, box_h), 8, 8)

        painter.setPen(QColor(235, 239, 248))
        y = top + pad + metrics.ascent()
        for line in lines:
            painter.drawText(left + pad, y, line)
            y += metrics.lineSpacing()

    def _draw_hints(self, painter: QPainter, width: int, height: int) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        font.setPointSizeF(9.5)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        hints = (
            "left-drag: look   wheel: zoom   arrows/WASD: look   +/-: zoom   "
            "R: reset   I: info   F: fullscreen   Esc: quit"
        )
        pad = 8
        box_w = metrics.horizontalAdvance(hints) + 2 * pad
        box_h = metrics.lineSpacing() + 2 * pad
        left = max(12, (width - box_w) // 2)
        top = height - box_h - 12

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 12, 16, 150))
        painter.drawRoundedRect(QRect(left, top, box_w, box_h), 8, 8)

        painter.setPen(QColor(208, 216, 230, 225))
        painter.drawText(left + pad, top + pad + metrics.ascent(), hints)

    # ------------------------------------------------------------------ #
    # Mouse
    # ------------------------------------------------------------------ #

    def mousePressEvent(self, event) -> None:  # pylint: disable=invalid-name
        if event.button() == Qt.MouseButton.LeftButton and self._source is not None:
            self._dragging = True
            self._last_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._touch()
            self.update()

    def mouseMoveEvent(self, event) -> None:  # pylint: disable=invalid-name
        if not self._dragging or self._last_pos is None:
            return
        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos

        # 1:1 "grab" feel: one pixel of drag == one pixel of image motion.
        deg_per_px = self.cam.fov / max(1, self.height())
        self.cam.rotate(-dx * deg_per_px, dy * deg_per_px)

        self._touch()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # pylint: disable=invalid-name
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._last_pos = None
            self.unsetCursor()
            self._touch()

    def wheelEvent(self, event) -> None:  # pylint: disable=invalid-name
        steps = event.angleDelta().y() / 120.0
        if not steps or self._source is None:
            return

        factor = 0.9**steps
        pos = event.position()

        # Zoom towards the cursor: keep the ray under the pointer fixed.
        lon_c, lat_c = self._ray_lonlat(pos.x(), pos.y())
        old_yaw, old_pitch = self.cam.yaw, self.cam.pitch

        self.cam.zoom(factor)

        lon_new, lat_new = self._ray_lonlat(pos.x(), pos.y())
        self.cam.yaw = wrap180(old_yaw + (lon_c - old_yaw) - (lon_new - old_yaw))
        self.cam.pitch = max(
            -90.0, min(90.0, old_pitch + (lat_c - old_pitch) - (lat_new - old_pitch))
        )

        self._touch()
        self.update()

    def _ray_lonlat(self, px: float, py: float) -> tuple[float, float]:
        """Longitude/latitude (radians) of the ray through a widget pixel."""
        width, height = max(1, self.width()), max(1, self.height())
        tan_half = math.tan(math.radians(self.cam.fov * 0.5))

        ndc_x = (px + 0.5) / width * 2.0 - 1.0
        ndc_y = (py + 0.5) / height * 2.0 - 1.0

        dx = ndc_x * tan_half * (width / height)
        dy = -ndc_y * tan_half
        dz = 1.0
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / norm, dy / norm, dz / norm

        pitch = math.radians(self.cam.pitch)
        yaw = math.radians(self.cam.yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        y1 = dy * cp + dz * sp
        z1 = dz * cp - dy * sp
        x2 = dx * cy + z1 * sy
        z2 = z1 * cy - dx * sy

        return math.atan2(x2, z2), math.asin(max(-1.0, min(1.0, y1)))

    # ------------------------------------------------------------------ #
    # Keyboard
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event) -> None:  # pylint: disable=invalid-name
        if event.isAutoRepeat():
            event.accept()
            return

        key = event.key()

        if key == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
            return

        if key in (Qt.Key.Key_F, Qt.Key.Key_F11):
            self._toggle_fullscreen()
            return

        if key == Qt.Key.Key_I:
            self._show_info = not self._show_info
            self.update()
            return

        if key == Qt.Key.Key_R:
            self.cam.reset()
            self._touch()
            self.update()
            return

        if key in NAV_KEYS:
            self._keys.add(key)
            if not self._nav_timer.isActive():
                self._nav_clock.restart()
                self._nav_timer.start()
            self._touch()
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # pylint: disable=invalid-name
        if event.isAutoRepeat():
            event.accept()
            return
        if event.key() in NAV_KEYS:
            self._keys.discard(event.key())
            if not self._keys:
                self._nav_timer.stop()
            self._touch()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _on_nav_tick(self) -> None:
        dt = max(0.001, min(0.1, self._nav_clock.restart() / 1000.0))
        speed = 70.0 * (self.cam.fov / Camera.DEFAULT_FOV)  # deg/s, zoom-aware

        d_yaw = d_pitch = 0.0
        if self._keys & {Qt.Key.Key_Left, Qt.Key.Key_A}:
            d_yaw -= speed * dt
        if self._keys & {Qt.Key.Key_Right, Qt.Key.Key_D}:
            d_yaw += speed * dt
        if self._keys & {Qt.Key.Key_Up, Qt.Key.Key_W}:
            d_pitch += speed * dt
        if self._keys & {Qt.Key.Key_Down, Qt.Key.Key_S}:
            d_pitch -= speed * dt
        if d_yaw or d_pitch:
            self.cam.rotate(d_yaw, d_pitch)

        zoom_keys = {Qt.Key.Key_Plus, Qt.Key.Key_Equal}
        shrink_keys = {Qt.Key.Key_Minus, Qt.Key.Key_Underscore}
        if self._keys & zoom_keys:
            self.cam.zoom(0.9 ** (dt * 10.0))
        if self._keys & shrink_keys:
            self.cam.zoom((1.0 / 0.9) ** (dt * 10.0))

        self.update()

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()


# --------------------------------------------------------------------------- #
# CPU viewer
# --------------------------------------------------------------------------- #


class PanoramaSoftwareViewer(ViewerMixin, QWidget):
    """Software (NumPy) renderer.  Used when OpenGL is not a good option."""

    def __init__(
        self,
        image: QImage | None,
        name: str,
        error: str | None = None,
        show_info: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._apply_window_common(name)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        source = image if image is not None and not image.isNull() else None
        self._init_common(name, error, show_info, source)

        self.renderer = EquirectRenderer(source) if source is not None else None

        # Interaction state specific to the two-tier CPU render.
        self._interactive = False
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(self.SETTLE_MS)
        self._settle_timer.timeout.connect(self._on_settle)

    # ------------------------------------------------------------------ #

    def _on_settle(self) -> None:
        if self._dragging or self._keys:
            self._settle_timer.start()
            return
        if self._interactive:
            self._interactive = False
            self.update()

    # ------------------------------------------------------------------ #

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            self._paint(painter)
        except (ValueError, RuntimeError, AttributeError):
            # An unhandled exception inside a Qt virtual can take the whole
            # process down; never let that happen during a repaint.
            traceback.print_exc()
            painter.fillRect(self.rect(), QColor(48, 0, 0))
        finally:
            painter.end()

    def _paint(self, painter: QPainter) -> None:
        width, height = max(1, self.width()), max(1, self.height())

        if self.renderer is None:
            self._draw_error(painter, width, height)
        else:
            dpr = self.devicePixelRatioF() or 1.0
            if self._interactive:
                render_w = int(min(900.0, max(360.0, width * dpr * 0.5)))
                bilinear = False
            else:
                render_w = int(min(3000.0, max(360.0, width * dpr)))
                bilinear = True
            render_h = max(2, round(render_w * height / width))

            started = time.perf_counter()
            frame = self.renderer.render(
                self.cam, render_w, render_h, bilinear=bilinear
            )
            self._render_ms = (time.perf_counter() - started) * 1000.0

            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(QRect(0, 0, width, height), frame)
            self._tick_fps()

        if self._show_info:
            self._draw_status(painter, width, height)
        self._draw_hints(painter, width, height)


# --------------------------------------------------------------------------- #
# GPU viewer
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


class PanoramaOpenGLViewer(ViewerMixin, QOpenGLWidget):
    """GPU renderer: one textured fullscreen quad, inverse projection in GLSL."""

    def __init__(
        self,
        image: QImage | None,
        name: str,
        error: str | None = None,
        show_info: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._apply_window_common(name)

        source = image if image is not None and not image.isNull() else None
        self._init_common(name, error, show_info, source)

        self._gl = None
        self._program: QOpenGLShaderProgram | None = None
        self._vao: QOpenGLVertexArrayObject | None = None
        self._texture: QOpenGLTexture | None = None
        self._u_scale = 1.0
        self._u_bias = 0.0
        self._context_hooked = False
        self._gl_failed = False

    @property
    def _backend_label(self) -> str:
        return "gpu"

    # ------------------------------------------------------------------ #
    # GL lifecycle
    # ------------------------------------------------------------------ #

    def initializeGL(self) -> None:
        self._gl = self.context().functions()

        if not self._context_hooked:
            self.context().aboutToBeDestroyed.connect(self._release_gl)
            self._context_hooked = True

        program = QOpenGLShaderProgram(self)
        ok = (
            program.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Vertex, VERT_SRC
            )
            and program.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Fragment, FRAG_SRC
            )
            and program.link()
        )
        if not ok:
            self._gl_failed = True
            self._error = f"shader error:\n{program.log()}"
            return
        self._program = program

        # Core profile requires a bound VAO even with no vertex attributes.
        self._vao = QOpenGLVertexArrayObject(self)
        self._vao.create()

        if self._source is not None:
            self._texture = self._build_texture()

    def _build_texture(self) -> QOpenGLTexture | None:
        """Upload the panorama once, wrapped-padded so mipmaps stay seamless."""
        src = self._source

        try:
            max_tex = int(QOpenGLTexture.maximumSize())
        except (AttributeError, ValueError):
            max_tex = 16384
        if max_tex > 0 and src.width() > max_tex:
            src = src.scaledToWidth(max_tex, Qt.TransformationMode.SmoothTransformation)

        src = src.convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = src.width(), src.height()
        if w < 2 or h < 2:
            return None

        pad = max(1, min(256, w // 8))
        padded = QImage(w + 2 * pad, h, QImage.Format.Format_RGBA8888)
        painter = QPainter(padded)
        try:
            painter.drawImage(0, 0, src, w - pad, 0, pad, h)  # right edge -> left
            painter.drawImage(pad, 0, src)
            painter.drawImage(pad + w, 0, src, 0, 0, pad, h)  # left edge -> right
        finally:
            painter.end()

        self._u_scale = w / (w + 2 * pad)
        self._u_bias = pad / (w + 2 * pad)

        texture = QOpenGLTexture(
            padded, QOpenGLTexture.MipMapGeneration.GenerateMipMaps
        )
        texture.setAutoMipMapGenerationEnabled(False)
        texture.setWrapMode(QOpenGLTexture.WrapMode.Repeat)
        texture.setMinificationFilter(QOpenGLTexture.Filter.LinearMipMapLinear)
        texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
        try:
            texture.setMaximumAnisotropy(texture.maximumAnisotropy())
        except (AttributeError, TypeError) as e:
            print(f"Warning: Could not set maximum anisotropy: {e}", file=sys.stderr)
        return texture

    def _release_gl(self) -> None:
        if not self.isValid():
            return
        self.makeCurrent()
        self._texture = None
        self._program = None
        self._vao = None
        self.doneCurrent()

    def closeEvent(self, event) -> None:
        # Free GL objects while the context is still current.
        self.makeCurrent()
        self._texture = None
        self._program = None
        self._vao = None
        self.doneCurrent()
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #

    def paintGL(self) -> None:
        width, height = max(1, self.width()), max(1, self.height())
        dpr = self.devicePixelRatioF() or 1.0

        painter = QPainter(self)
        painter.beginNativePainting()
        try:
            f = self._gl
            f.glViewport(0, 0, round(width * dpr), round(height * dpr))
            f.glClearColor(0.07, 0.08, 0.10, 1.0)
            f.glClear(GL_COLOR_BUFFER_BIT)

            if self._program is not None and self._texture is not None:
                started = time.perf_counter()
                self._draw_panorama(f, width, height)
                self._render_ms = (time.perf_counter() - started) * 1000.0
                self._tick_fps()
        except (ValueError, RuntimeError, AttributeError):
            traceback.print_exc()
        finally:
            painter.endNativePainting()
            if self._source is None:
                self._draw_error(painter, width, height)
            elif self._show_info:
                self._draw_status(painter, width, height)
            self._draw_hints(painter, width, height)
            painter.end()

    def _draw_panorama(self, f, width: int, height: int) -> None:
        program = self._program
        yaw, pitch = math.radians(self.cam.yaw), math.radians(self.cam.pitch)

        program.bind()
        _set_uniform_f(
            program, "tan_half_fov", math.tan(math.radians(self.cam.fov * 0.5))
        )
        _set_uniform_f(program, "aspect", width / height)
        _set_uniform_f(program, "cy", math.cos(yaw))
        _set_uniform_f(program, "sy", math.sin(yaw))
        _set_uniform_f(program, "cp", math.cos(pitch))
        _set_uniform_f(program, "sp", math.sin(pitch))
        _set_uniform_f(program, "u_scale", self._u_scale)
        _set_uniform_f(program, "u_bias", self._u_bias)
        _set_uniform_i(program, "pano", 0)

        f.glActiveTexture(GL_TEXTURE0)
        self._texture.bind()
        self._vao.bind()
        f.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        self._vao.release()
        self._texture.release()
        program.release()


# --------------------------------------------------------------------------- #
# Environment checks
# --------------------------------------------------------------------------- #


def configure_surface_format(vsync: bool = True) -> None:
    """Request a 3.3 core context.  Must run before QApplication exists."""
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    fmt.setSamples(0)  # we sample a texture, not draw edges
    fmt.setSwapInterval(1 if vsync else 0)
    QSurfaceFormat.setDefaultFormat(fmt)


def gl_is_usable() -> tuple[bool, str]:
    """Offscreen probe.  Returns (usable, reason-if-not).  Needs a QApplication."""
    fmt = QSurfaceFormat.defaultFormat()
    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    if not surface.isValid():
        return False, "no offscreen surface"

    ctx = QOpenGLContext()
    ctx.setFormat(fmt)
    if not ctx.create() or not ctx.makeCurrent(surface):
        return False, "no OpenGL 3.3 core context"

    try:
        if ctx.format().version() < (3, 3):
            return False, f"OpenGL {ctx.format().version()} < 3.3"
        renderer = ""
        try:
            raw = ctx.functions().glGetString(GL_RENDERER)
            renderer = (
                raw.decode(errors="replace")
                if isinstance(raw, (bytes, bytearray))
                else str(raw)
            )
        except (AttributeError, ValueError, RuntimeError) as e:
            print(f"Warning: Could not get OpenGL renderer: {e}", file=sys.stderr)
        if any(name in renderer.lower() for name in SOFTWARE_RENDERERS):
            return False, f"software OpenGL ({renderer})"
        return True, renderer or "unknown adapter"
    finally:
        ctx.doneCurrent()
        surface.destroy()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = text.lower().split("x")
        return max(320, int(width)), max(240, int(height))
    except (ValueError, AttributeError):
        return 1280, 720


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="panov",
        description="Interactive 360° equirectangular panorama viewer.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to an equirectangular panorama (.jpg / .png). If omitted, a file dialog is shown.",
    )
    parser.add_argument(
        "-f",
        "--fullscreen",
        action="store_true",
        help="start in fullscreen mode",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=Camera.DEFAULT_FOV,
        help="Initial vertical field of view in degrees (default: 75)",
    )
    parser.add_argument(
        "--size",
        default="1280x720",
        help="Initial window size, e.g. 1280x720",
    )
    parser.add_argument(
        "--no-info",
        action="store_true",
        help="Start with the status overlay hidden (toggle at runtime with I)",
    )
    parser.add_argument(
        "--software",
        action="store_true",
        help="Force the NumPy renderer instead of OpenGL",
    )
    parser.add_argument(
        "--no-vsync",
        action="store_true",
        help="Uncap the frame rate (renders as fast as the GPU allows)",
    )
    args = parser.parse_args(argv)

    configure_surface_format(vsync=not args.no_vsync)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Panov")

    if args.software:
        use_gl, reason = False, "--software"
    else:
        use_gl, reason = gl_is_usable()
    if not use_gl:
        print(f"Using CPU renderer ({reason})", file=sys.stderr)

    path = args.image
    if path is None:
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Open panorama",
            "",
            "Panoramas (*.jpg *.JPG *.jpeg *.JPEG *.png *.PNG);;All files (*)",
        )
        if not path:
            return 0

    image = QImage()
    error: str | None = None

    if not os.path.isfile(path):
        error = f"file not found: {path}"
    else:
        image = QImage(path)
        if image.isNull():
            error = f"could not decode image: {path}"

    name = os.path.basename(path)
    viewer_cls = PanoramaOpenGLViewer if use_gl else PanoramaSoftwareViewer
    viewer = viewer_cls(
        image if not image.isNull() else None,
        name,
        error,
        show_info=not args.no_info,
    )
    viewer.cam.fov = Camera._clamp_fov(args.fov)
    viewer.resize(*parse_size(args.size))

    if args.fullscreen:
        viewer.showFullScreen()
    else:
        viewer.show()

    if error:
        QMessageBox.warning(None, "Panov", error)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

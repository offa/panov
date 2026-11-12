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

"""Camera navigation, overlays, and both concrete viewers."""

from __future__ import annotations

import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QElapsedTimer, QPointF, QRect, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QImage,
    QPainter,
)
from PySide6.QtOpenGL import (
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from .camera import Camera, wrap180
from .render import (
    FRAG_SRC,
    GL_COLOR_BUFFER_BIT,
    GL_TEXTURE0,
    GL_TRIANGLE_STRIP,
    VERT_SRC,
    EquirectRenderer,
    _set_uniform_f,
    _set_uniform_i,
)

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


# --------------------------------------------------------------------------- #
# Shared viewer behaviour
# --------------------------------------------------------------------------- #


@dataclass
class _Input:
    """Mouse/keyboard navigation state and its key-repeat timer."""

    timer: QTimer
    clock: QElapsedTimer
    keys: set[int] = field(default_factory=set)
    dragging: bool = False
    last_pos: QPointF | None = None
    interactive: bool = False


@dataclass
class _Stats:
    """Render timing counters for the status overlay."""

    clock: QElapsedTimer = field(default_factory=QElapsedTimer)
    render_ms: float = 0.0
    fps: float = 0.0
    frames: int = 0


class ViewerMixin:
    """Camera navigation, input handling and overlay drawing.

    Mixed into a QWidget subclass.

    Concrete viewers must also provide:

    ``cam``            Camera instance (set up by ``__init__``)
    ``_source``        QImage or None
    ``_error``         str or None
    ``_show_info``     bool
    ``_input``         _Input (set up by ``__init__``)
    ``_stats``         _Stats (set up by ``__init__``)
    ``_settle_timer``  QTimer, or absent/None for viewers with no settle delay
    ``_backend_label`` str property, used by the status overlay
    """

    NAV_INTERVAL_MS = 16  # ~60 Hz keyboard navigation
    SETTLE_MS = 250  # CPU only: delay before switching to a quality render

    # -- construction ------------------------------------------------------ #

    def __init__(
        self,
        name: str,
        error: str | None,
        show_info: bool,
        source: QImage | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

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

        nav_timer = QTimer(self)
        nav_timer.setInterval(self.NAV_INTERVAL_MS)
        nav_timer.timeout.connect(self._on_nav_tick)  # pylint: disable=no-member

        nav_clock = QElapsedTimer()
        nav_clock.start()

        self._input = _Input(timer=nav_timer, clock=nav_clock)
        self._stats = _Stats()
        self._stats.clock.start()

    @property
    def error(self) -> str | None:
        return self._error

    def load_image(self, path: str) -> None:
        """Load a new panorama from disk, replacing the current source."""
        path = os.fspath(path)
        name = os.path.basename(path)
        self._name = name or "no image"

        if not os.path.isfile(path):
            self._source = None
            self._error = f"file not found: {path}"
            self._aspect_warning = False
        else:
            image = QImage(path)
            if image.isNull():
                self._source = None
                self._error = f"could not decode image: {path}"
                self._aspect_warning = False
            else:
                self._source = image
                self._error = None
                self._aspect_warning = bool(
                    image.height() > 0
                    and abs(image.width() / image.height() - 2.0) > 0.05
                )

        self.cam.reset()
        self._on_source_changed()
        self._apply_window_common()
        self._touch()
        self.update()

    def _on_source_changed(self) -> None:
        """Hook for subclasses to rebuild backend state after a source change."""

    def _apply_window_common(self) -> None:
        self.setMinimumSize(480, 270)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    # -- hooks overridden by the CPU viewer -------------------------------- #

    def _touch(self) -> None:
        """Mark the camera as moving so we render cheaply."""
        timer = getattr(self, "_settle_timer", None)
        if timer is None:
            return
        self._input.interactive = True
        timer.start()

    def release_resources(self) -> None:
        """Release backend-specific resources. No-op by default."""

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
        stats = self._stats
        stats.frames += 1
        elapsed = stats.clock.elapsed()
        if elapsed >= 500:
            stats.fps = stats.frames * 1000.0 / elapsed
            stats.frames = 0
            stats.clock.restart()

    @staticmethod
    def _status_layout(
        lines: list[str], metrics: QFontMetrics
    ) -> tuple[int, int, int, int]:
        """Dimensions and position for the status panel."""
        pad = 10
        box_w = max(metrics.horizontalAdvance(line) for line in lines) + 2 * pad
        box_h = metrics.lineSpacing() * len(lines) + 2 * pad
        return 12, 12, box_w, box_h

    def _draw_status(self, painter: QPainter) -> None:
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
            f"render      {self._stats.render_ms:6.1f} ms  {self._stats.fps:4.0f} fps",
            f"backend     {self._backend_label}",
        ]
        if self._aspect_warning:
            lines.append("note        source aspect is not 2:1 (may look stretched)")

        left, top, box_w, box_h = self._status_layout(lines, metrics)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 12, 16, 175))
        painter.drawRoundedRect(QRect(left, top, box_w, box_h), 8, 8)

        painter.setPen(QColor(235, 239, 248))
        y = top + 10 + metrics.ascent()
        for line in lines:
            painter.drawText(left + 10, y, line)
            y += metrics.lineSpacing()

    def _draw_hints(self, painter: QPainter, width: int, height: int) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        font.setPointSizeF(9.5)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        hints = (
            "left-drag: look   wheel: zoom   arrows/WASD: look   +/-: zoom   "
            "R: reset   I: info   B: browser   F: fullscreen   Esc: quit"
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
            self._input.dragging = True
            self._input.last_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._touch()
            self.update()

    def mouseMoveEvent(self, event) -> None:  # pylint: disable=invalid-name
        nav = self._input
        if not nav.dragging or nav.last_pos is None:
            return
        pos = event.position()
        dx = pos.x() - nav.last_pos.x()
        dy = pos.y() - nav.last_pos.y()
        nav.last_pos = pos

        # 1:1 "grab" feel: one pixel of drag == one pixel of image motion.
        deg_per_px = self.cam.fov / max(1, self.height())
        self.cam.rotate(-dx * deg_per_px, dy * deg_per_px)

        self._touch()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # pylint: disable=invalid-name
        if event.button() == Qt.MouseButton.LeftButton:
            self._input.dragging = False
            self._input.last_pos = None
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
        self.cam.yaw = wrap180(old_yaw + lon_c - lon_new)
        self.cam.pitch = max(-90.0, min(90.0, old_pitch + lat_c - lat_new))

        self._touch()
        self.update()

    def _ray_lonlat(self, px: float, py: float) -> tuple[float, float]:
        """Longitude/latitude in degrees of the ray through a widget pixel."""
        width, height = max(1, self.width()), max(1, self.height())
        tan_half = math.tan(math.radians(self.cam.fov * 0.5))

        ndc = np.array(
            [(px + 0.5) / width * 2.0 - 1.0, (py + 0.5) / height * 2.0 - 1.0]
        )

        ray = np.array([ndc[0] * tan_half * (width / height), -ndc[1] * tan_half, 1.0])
        ray /= np.linalg.norm(ray)

        pitch, yaw = map(math.radians, [self.cam.pitch, self.cam.yaw])

        y1 = ray[1] * math.cos(pitch) + ray[2] * math.sin(pitch)
        z1 = ray[2] * math.cos(pitch) - ray[1] * math.sin(pitch)

        x2 = ray[0] * math.cos(yaw) + z1 * math.sin(yaw)
        z2 = z1 * math.cos(yaw) - ray[0] * math.sin(yaw)

        return math.degrees(math.atan2(x2, z2)), math.degrees(
            math.asin(np.clip(y1, -1.0, 1.0))
        )

    # ------------------------------------------------------------------ #
    # Keyboard
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event) -> None:  # pylint: disable=invalid-name,too-many-return-statements
        if event.isAutoRepeat():
            event.accept()
            return

        key = event.key()

        if key == Qt.Key.Key_Escape:
            win = self.window()
            if win.isFullScreen():
                win.showNormal()
            else:
                win.close()
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

        if key == Qt.Key.Key_B:
            win = self.window()
            dock = getattr(win, "dock", None)
            if dock is not None:
                dock.setVisible(not dock.isVisible())
            return

        if key in NAV_KEYS:
            nav = self._input
            nav.keys.add(key)
            if not nav.timer.isActive():
                nav.clock.restart()
                nav.timer.start()
            self._touch()
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # pylint: disable=invalid-name
        if event.isAutoRepeat():
            event.accept()
            return
        if event.key() in NAV_KEYS:
            nav = self._input
            nav.keys.discard(event.key())
            if not nav.keys:
                nav.timer.stop()
            self._touch()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:  # pylint: disable=invalid-name
        nav = self._input
        nav.keys.clear()
        nav.timer.stop()
        nav.dragging = False
        nav.last_pos = None
        self._touch()
        super().focusOutEvent(event)

    def _on_nav_tick(self) -> None:
        nav = self._input
        dt = max(0.001, min(0.1, nav.clock.restart() / 1000.0))
        speed = 70.0 * (self.cam.fov / Camera.DEFAULT_FOV)  # deg/s, zoom-aware

        d_yaw = d_pitch = 0.0
        if nav.keys & {Qt.Key.Key_Left, Qt.Key.Key_A}:
            d_yaw -= speed * dt
        if nav.keys & {Qt.Key.Key_Right, Qt.Key.Key_D}:
            d_yaw += speed * dt
        if nav.keys & {Qt.Key.Key_Up, Qt.Key.Key_W}:
            d_pitch += speed * dt
        if nav.keys & {Qt.Key.Key_Down, Qt.Key.Key_S}:
            d_pitch -= speed * dt
        if d_yaw or d_pitch:
            self.cam.rotate(d_yaw, d_pitch)

        zoom_keys = {Qt.Key.Key_Plus, Qt.Key.Key_Equal}
        shrink_keys = {Qt.Key.Key_Minus, Qt.Key.Key_Underscore}
        if nav.keys & zoom_keys:
            self.cam.zoom(0.9 ** (dt * 10.0))
        if nav.keys & shrink_keys:
            self.cam.zoom((1.0 / 0.9) ** (dt * 10.0))

        self.update()

    def _toggle_fullscreen(self) -> None:
        win = self.window()
        if win.isFullScreen():
            win.showNormal()
        else:
            win.showFullScreen()


# --------------------------------------------------------------------------- #
# CPU viewer
# --------------------------------------------------------------------------- #


class PanoramaSoftwareViewer(ViewerMixin, QWidget):
    """Software (NumPy) renderer. Used when OpenGL is not a good option."""

    def __init__(
        self,
        image: QImage | None,
        name: str,
        error: str | None = None,
        show_info: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        source = image if image is not None and not image.isNull() else None
        super().__init__(name, error, show_info, source, parent)

        self._apply_window_common()
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self.renderer = EquirectRenderer(source) if source is not None else None

        self._input.interactive = False
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(self.SETTLE_MS)
        self._settle_timer.timeout.connect(self._on_settle)  # pylint: disable=no-member

    # ------------------------------------------------------------------ #

    def _on_source_changed(self) -> None:
        self.renderer = (
            EquirectRenderer(self._source) if self._source is not None else None
        )

    def _on_settle(self) -> None:
        nav = self._input
        if nav.dragging or nav.keys:
            self._settle_timer.start()
            return
        if nav.interactive:
            nav.interactive = False
            self.update()

    # ------------------------------------------------------------------ #

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        try:
            self._paint(painter)
        except Exception:  # pylint: disable=broad-exception-caught  # noqa: BLE001
            # Last-resort safety net: an unhandled exception inside a Qt
            # event handler can terminate the process. Expected errors are
            # handled upstream (load_image, etc.); this catches anything
            # unexpected (MemoryError, IndexError, TypeError, …) so there's
            # a degraded frame instead of a crash.
            traceback.print_exc()
            painter.fillRect(self.rect(), QColor(48, 0, 0))
        finally:
            painter.end()

    def _paint(self, painter: QPainter) -> None:
        width, height = max(1, self.width()), max(1, self.height())

        if self._source is None or self.renderer is None:
            self._draw_error(painter, width, height)
        else:
            dpr = self.devicePixelRatioF() or 1.0
            if self._input.interactive:
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
            self._stats.render_ms = (time.perf_counter() - started) * 1000.0

            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(QRect(0, 0, width, height), frame)
            self._tick_fps()

        if self._show_info and self._source is not None and self.renderer is not None:
            self._draw_status(painter)
        self._draw_hints(painter, width, height)


# --------------------------------------------------------------------------- #
# GPU viewer
# --------------------------------------------------------------------------- #


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
        source = image if image is not None and not image.isNull() else None
        super().__init__(name, error, show_info, source, parent)

        self._apply_window_common()

        self._gl = None
        self._program: QOpenGLShaderProgram | None = None
        self._vao: QOpenGLVertexArrayObject | None = None
        self._texture: QOpenGLTexture | None = None
        self._u_scale = 1.0
        self._u_bias = 0.0
        self._context_hooked = False
        self._gl_failed = False
        self._gl_error: str | None = None
        self._needs_texture_rebuild = False

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
            self._gl_error = f"shader error:\n{program.log()}"
            self._error = self._gl_error
            return
        self._program = program

        # Core profile requires a bound VAO even with no vertex attributes.
        self._vao = QOpenGLVertexArrayObject(self)
        self._vao.create()

        self._rebuild_texture()

    def _rebuild_texture(self) -> None:
        """Rebuild the GL texture for the current source.

        Runs from ``initializeGL`` and from ``paintGL`` after a new image
        was loaded.  Sets ``_error`` if the source decoded but is too
        small to upload, so the error overlay explains the blank frame
        instead of leaving a silent dark gray one.
        """
        if self._source is None:
            self._texture = None
        else:
            self._texture = self._build_texture()
            if self._texture is None:
                self._error = (
                    f"image is too small for the GPU renderer "
                    f"({self._source.width()} x {self._source.height()} px, "
                    f"minimum is 2x2)"
                )
        self._needs_texture_rebuild = False

    def _build_texture(self) -> QOpenGLTexture | None:
        """Upload the panorama once, wrapped-padded so mipmaps stay seamless."""
        src = self._source

        try:
            max_tex = int(QOpenGLTexture.maximumSize())
        except (AttributeError, ValueError):
            max_tex = 16384
        if 0 < max_tex < src.width():
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

    def _on_source_changed(self) -> None:
        self._needs_texture_rebuild = True
        if self._gl_failed:
            self._error = self._gl_error

    def _release_gl(self) -> None:
        if not self.isValid():
            return
        self.makeCurrent()
        self._texture = None
        self._program = None
        self._vao = None
        self._context_hooked = False
        self.doneCurrent()

    def release_resources(self) -> None:
        self._release_gl()

    def closeEvent(self, event) -> None:
        # Free GL objects while the context is still current.
        if self._gl is not None and self.isValid():
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
            if f is None:
                return

            f.glViewport(0, 0, round(width * dpr), round(height * dpr))
            f.glClearColor(0.07, 0.08, 0.10, 1.0)
            f.glClear(GL_COLOR_BUFFER_BIT)

            if self._needs_texture_rebuild:
                self._rebuild_texture()

            if self._program is not None and self._texture is not None:
                started = time.perf_counter()
                self._draw_panorama(f, width, height)
                self._stats.render_ms = (time.perf_counter() - started) * 1000.0
                self._tick_fps()
        except Exception:  # pylint: disable=broad-exception-caught  # noqa: BLE001
            # Last-resort safety net: an unhandled exception inside a Qt
            # event handler can terminate the process. Expected errors are
            # handled upstream; this catches anything unexpected so there's
            # a degraded frame instead of a crash.
            traceback.print_exc()
        finally:
            painter.endNativePainting()
            if (
                self._source is None
                or self._gl_failed
                or self._program is None
                or self._texture is None
            ):
                self._draw_error(painter, width, height)
            elif self._show_info:
                self._draw_status(painter)
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

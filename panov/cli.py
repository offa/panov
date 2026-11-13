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

"""Command line entry point and environment probing."""

from __future__ import annotations

import argparse
import sys

from PySide6.QtCore import QThreadPool
from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from .camera import Camera
from .main_window import PanovMainWindow
from .render import GL_RENDERER, SOFTWARE_RENDERERS
from .viewer import PanoramaOpenGLViewer, PanoramaSoftwareViewer

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
    if not ctx.create():
        return False, "no OpenGL 3.3 core context"

    try:
        if ctx.format().version() < (3, 3):
            return False, f"OpenGL {ctx.format().version()} < 3.3"
        if not ctx.makeCurrent(surface):
            return False, "could not make context current"
        renderer = ""
        try:
            raw = ctx.functions().glGetString(GL_RENDERER)
            renderer = (
                raw.decode(errors="replace")
                if isinstance(raw, (bytes, bytearray)) and raw is not None
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
    """Parse a ``WxH`` window size, falling back to 1280x720 with a warning.

    No lower bound is applied here: the viewer widget's minimum size is
    the single source of truth, and Qt clamps the window to it.
    """
    try:
        width, height = text.lower().split("x")
        size = (int(width), int(height))
    except (ValueError, AttributeError):
        print(
            f"Warning: malformed --size {text!r}, using default 1280x720",
            file=sys.stderr,
        )
        return 1280, 720
    if min(size) < 1:
        print(
            f"Warning: --size {text!r} is not positive, using default 1280x720",
            file=sys.stderr,
        )
        return 1280, 720
    return size


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
        help="Initial window size, e.g. 1280x720 (minimum 480x270)",
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
    size = parse_size(args.size)

    configure_surface_format(vsync=not args.no_vsync)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Panov")

    if args.software:
        use_gl, reason = False, "--software"
    else:
        use_gl, reason = gl_is_usable()
    if not use_gl:
        print(f"Using CPU renderer ({reason})", file=sys.stderr)

    viewer_cls = PanoramaOpenGLViewer if use_gl else PanoramaSoftwareViewer
    viewer = viewer_cls(None, "no image", None, show_info=not args.no_info)
    main_window = PanovMainWindow(viewer)

    path = args.image or None
    if path is None:
        path, _ = QFileDialog.getOpenFileName(
            main_window,
            "Open panorama",
            "",
            "Panoramas (*.jpg *.JPG *.jpeg *.JPEG *.png *.PNG *.bmp *.BMP *.tiff *.TIFF *.tif *.TIF);;All files (*)",
        )
        if not path:
            return 0

    main_window.load_image(path)
    viewer.cam.set_fov(args.fov)
    main_window.resize(*size)

    if args.fullscreen:
        main_window.showFullScreen()
    else:
        main_window.show()

    if viewer.error:
        QMessageBox.warning(main_window, "Panov", viewer.error)

    status = app.exec()
    # Let in-flight thumbnail decodes finish before the interpreter tears
    # down the global thread pool (see image_browser._ThumbnailJob).
    QThreadPool.globalInstance().waitForDone()
    return status

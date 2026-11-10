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

"""Main window hosting the panorama viewer and image browser dock."""

from __future__ import annotations

import os

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import QMainWindow, QMenu, QWidget

from .image_browser import ImageBrowserDock
from .viewer import ViewerMixin


class PanovMainWindow(QMainWindow):
    """Top-level window containing the viewer and an optional image dock."""

    def __init__(
        self,
        viewer: ViewerMixin,
        initial_directory: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._viewer = viewer
        self.setCentralWidget(viewer)
        self.setWindowTitle("Panov")

        self._dock = ImageBrowserDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock)
        self._dock.hide()

        self._browser_action = QAction("Show Image Browser", self)
        self._browser_action.setCheckable(True)
        self._browser_action.setChecked(self._dock.isVisible())
        self._browser_action.toggled.connect(self._dock.setVisible)  # pylint: disable=no-member
        self._dock.visibilityChanged.connect(self._browser_action.setChecked)  # pylint: disable=no-member

        view_menu = QMenu("View", self)
        view_menu.addAction(self._browser_action)
        self.menuBar().addMenu(view_menu)

        self._dock.image_selected.connect(self._on_image_selected)

        if initial_directory is not None:
            self._dock.set_directory(initial_directory)

    def load_image(self, path: str) -> None:
        self._viewer.load_image(path)
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            self._dock.set_directory(directory)
        self.setWindowTitle(f"Panov - {os.path.basename(path) or 'no image'}")

    def _on_image_selected(self, path: str) -> None:
        self.load_image(path)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.WindowStateChange:
            if self.isFullScreen():
                self.menuBar().hide()
            else:
                self.menuBar().show()
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._viewer._release_gl()
        super().closeEvent(event)

    @property
    def viewer(self) -> ViewerMixin:
        return self._viewer

    @property
    def dock(self) -> ImageBrowserDock:
        return self._dock

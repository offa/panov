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

"""Dock widget that lists image files in a directory with thumbnails."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, QSize, Signal
from PySide6.QtGui import QFontMetrics, QIcon, QImageReader, QPixmap
from PySide6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QWidget

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"})
THUMBNAIL_SIZE = 64


class ImageBrowserDock(QDockWidget):
    """A dockable list of image files in a single directory."""

    image_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Image Browser")

        self._directory: str | None = None
        self._thumbnail_cache: dict[str, QPixmap] = {}
        self._pending: list[tuple[QListWidgetItem, str]] = []

        self._list = QListWidget(self)
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setMovement(QListWidget.Movement.Static)
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.itemClicked.connect(self._on_item_clicked)
        self.setWidget(self._list)

        self._thumbnail_timer = QTimer(self)
        self._thumbnail_timer.setSingleShot(True)
        self._thumbnail_timer.setInterval(16)
        self._thumbnail_timer.timeout.connect(self._process_thumbnails)

    @property
    def current_directory(self) -> str | None:
        return self._directory

    def set_directory(self, path: str | None) -> None:
        if path is not None:
            path = os.fspath(path)
            if not os.path.isdir(path):
                path = None

        if path == self._directory:
            return

        self._directory = path
        self._thumbnail_cache.clear()
        self._pending.clear()
        self._thumbnail_timer.stop()
        self._list.clear()

        if path is None:
            self._add_placeholder("No directory selected")
            return

        paths = self._scan_directory(path)
        if not paths:
            self._add_placeholder("No images found")
            return

        for file_path in paths:
            item = self._make_item(file_path)
            self._list.addItem(item)
            self._pending.append((item, file_path))

        if self._pending:
            self._thumbnail_timer.start()

    def _add_placeholder(self, text: str) -> None:
        item = QListWidgetItem(text)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        self._list.addItem(item)

    def _make_item(self, file_path: str) -> QListWidgetItem:
        name = os.path.basename(file_path)
        metrics = QFontMetrics(self._list.font())
        elided = metrics.elidedText(name, Qt.TextElideMode.ElideRight, THUMBNAIL_SIZE)
        item = QListWidgetItem(elided)
        item.setIcon(QIcon())
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        item.setToolTip(file_path)
        return item

    @staticmethod
    def _scan_directory(path: str) -> list[str]:
        result: list[str] = []
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    try:
                        if not entry.is_file():
                            continue
                    except OSError:
                        continue
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        result.append(entry.path)
        except OSError:
            return []
        return sorted(result, key=lambda p: os.path.basename(p).lower())

    def _process_thumbnails(self) -> None:
        if not self._pending:
            self._thumbnail_timer.stop()
            return

        item, path = self._pending.pop(0)
        pixmap = self._thumbnail_cache.get(path)
        if pixmap is None:
            pixmap = self._load_thumbnail(path)
            if pixmap is not None:
                self._thumbnail_cache[path] = pixmap

        if pixmap is not None:
            item.setIcon(QIcon(pixmap))

        if self._pending:
            self._thumbnail_timer.start()

    @staticmethod
    def _load_thumbnail(path: str) -> QPixmap | None:
        reader = QImageReader(path)
        reader.setScaledSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        image = reader.read()
        if image.isNull():
            return None
        return QPixmap.fromImage(image)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(path, str) and path:
            self.image_selected.emit(path)

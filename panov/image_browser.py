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
import sys
from collections import deque
from typing import ClassVar

from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import QFontMetrics, QIcon, QImage, QImageReader, QKeyEvent, QPixmap
from PySide6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QMenu, QWidget

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"})
THUMBNAIL_SIZE = 64
THUMBNAIL_WORKERS = 4  # concurrent decodes; keeps CPU use and memory bounded


class _ThumbnailBridge(QObject):
    """Carries thumbnail results from pool worker threads to the GUI thread.

    ``ready`` is emitted from a worker thread; Qt delivers it queued to the
    GUI thread, where the dock consumes it.  A plain QObject is used because
    PySide6 binds QRunnable without QObject support, so Python-defined
    signals on a QRunnable subclass are not connected.
    """

    ready = Signal(int, int, str, QImage)  # job id, generation, path, image

    def publish(self, job_id: int, generation: int, path: str, image: QImage) -> None:
        """Publish a finished thumbnail (may be called from any thread)."""
        self.ready.emit(job_id, generation, path, image)


class _ThumbnailJob(QRunnable):
    """Decode one thumbnail on a global thread pool worker thread.

    The image decode (the expensive part) runs off the GUI thread; the
    result is reported through the dock's ``_ThumbnailBridge``, which Qt
    delivers queued to the GUI thread.  The pool auto-deletes the C++ side
    after ``run()`` returns, so ``active`` must keep the Python wrapper (and
    with it the C++ object) alive until the job id is released -- even if
    the dock that started the job is destroyed first (e.g. at application
    exit).
    """

    active: ClassVar[dict[int, _ThumbnailJob]] = {}
    _next_id: ClassVar[int] = 0

    @classmethod
    def create(
        cls, generation: int, path: str, bridge: _ThumbnailBridge
    ) -> _ThumbnailJob:
        """Register a new job in ``active`` and return it."""
        job = _ThumbnailJob(cls._next_id, generation, path, bridge)
        cls._next_id += 1
        cls.active[job._job_id] = job
        return job

    @classmethod
    def release(cls, job_id: int) -> None:
        """Drop the wrapper of a finished job (no-op if unknown)."""
        cls.active.pop(job_id, None)

    def __init__(
        self, job_id: int, generation: int, path: str, bridge: _ThumbnailBridge
    ) -> None:
        super().__init__()
        self._job_id = job_id
        self._generation = generation
        self._path = path
        self._bridge = bridge

    def run(self) -> None:
        image = QImage()
        try:
            reader = QImageReader(self._path)
            reader.setScaledSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
            image = reader.read()
        except Exception as e:  # pylint: disable=broad-exception-caught  # noqa: BLE001
            # Decode failed (missing/unreadable file): the null image is
            # still published below so the dock frees the worker slot.
            print(f"Warning: Could not decode thumbnail: {e}", file=sys.stderr)
        try:
            self._bridge.publish(self._job_id, self._generation, self._path, image)
        except RuntimeError as e:
            # The bridge was destroyed while we were running (application
            # exit): the result is no longer needed.
            print(f"Warning: Could not publish thumbnail: {e}", file=sys.stderr)


class ImageBrowserDock(QDockWidget):
    """A dockable list of image files in a single directory."""

    image_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Image Browser")

        self._directory: str | None = None
        self._generation: int = 0  # invalidates in-flight thumbnails
        self._thumbnail_cache: dict[str, QPixmap] = {}
        self._items: dict[str, QListWidgetItem] = {}
        self._pending: deque[str] = deque()
        self._inflight: int = 0

        # Results hop from worker threads to the GUI thread via this bridge.
        self._bridge = _ThumbnailBridge(self)
        self._bridge.ready.connect(self._on_thumbnail_ready)

        self._list = QListWidget(self)
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setMovement(QListWidget.Movement.Static)
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        # itemActivated (not itemClicked) so Enter/Return on the
        # highlighted item loads it too; a single click still activates
        # in the default SingleClick activation mode.
        # pylint: disable=no-member
        self._list.itemActivated.connect(self._on_item_activated)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        # pylint: disable=no-member
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        self._list.installEventFilter(self)
        self.setWidget(self._list)

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

        self._rebuild(path)

    def refresh(self) -> None:
        """Re-scan the current directory (e.g. after files were added)."""
        if self._directory is not None:
            self._rebuild(self._directory)

    def _rebuild(self, path: str | None) -> None:
        """Replace the list contents with a fresh scan of ``path``.

        Bumps the generation to invalidate in-flight decodes: their
        results carry the old generation and are dropped on arrival.
        Restores the previous selection when that file is still present.
        """
        current: str | None = None
        current_item = self._list.currentItem()
        if current_item is not None:
            data = current_item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, str):
                current = data

        self._generation += 1
        self._directory = path
        self._thumbnail_cache.clear()
        self._items.clear()
        self._pending.clear()
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
            self._items[file_path] = item
            self._list.addItem(item)
            self._pending.append(file_path)

        if current is not None:
            item = self._items.get(current)
            if item is not None:
                self._list.setCurrentItem(item)
                self._list.scrollToItem(item)

        self._dispatch_thumbnails()

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

    def _dispatch_thumbnails(self) -> None:
        """Start decoding pending thumbnails on the global thread pool."""
        pool = QThreadPool.globalInstance()
        while self._inflight < THUMBNAIL_WORKERS and self._pending:
            job = _ThumbnailJob.create(
                self._generation, self._pending.popleft(), self._bridge
            )
            self._inflight += 1
            pool.start(job)

    def _on_thumbnail_ready(
        self, job_id: int, generation: int, path: str, image: QImage
    ) -> None:
        _ThumbnailJob.release(job_id)
        self._inflight -= 1

        if generation == self._generation and not image.isNull():
            item = self._items.get(path)
            if item is not None:
                pixmap = QPixmap.fromImage(image)
                self._thumbnail_cache[path] = pixmap
                item.setIcon(QIcon(pixmap))

        # The freed slot may pick up the next pending thumbnail.
        self._dispatch_thumbnails()

    def _on_item_activated(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(path, str) and path:
            self.image_selected.emit(path)

    # pylint: disable=invalid-name
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        # F5 re-scans the current directory (key events only reach the
        # list, so the filter is active whenever it has the focus).
        if (
            obj is self._list
            and isinstance(event, QKeyEvent)
            and not event.isAutoRepeat()
            and event.key() == Qt.Key.Key_F5
        ):
            self.refresh()
            return True
        return super().eventFilter(obj, event)

    def _on_context_menu(self, position: QPoint) -> None:
        menu = QMenu(self._list)
        refresh_action = menu.addAction("Refresh")
        refresh_action.setEnabled(self._directory is not None)
        refresh_action.triggered.connect(self.refresh)  # pylint: disable=no-member
        menu.exec(self._list.mapToGlobal(position))

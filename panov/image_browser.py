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
        """Update the list to show a fresh scan of ``path``.

        The scan is diffed against the displayed items, so unchanged
        files keep their item and thumbnail and only vanished or new
        files are touched.  Besides the obvious UX wins (no flicker, no
        re-decoding) this avoids the clear-and-rebuild churn that can
        leave the icon-mode grid stale and make items overlap.

        In-flight decodes are invalidated by bumping the generation:
        their results carry the old generation and are dropped on
        arrival.  The previous selection is restored when that file is
        still present.
        """
        current: str | None = None
        current_item = self._list.currentItem()
        if current_item is not None:
            data = current_item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, str):
                current = data

        self._directory = path
        paths = self._scan_directory(path) if path is not None else []

        if not paths:
            self._show_placeholder(
                "No directory selected" if path is None else "No images found"
            )
            return

        if paths == list(self._items):
            # The displayed list is already up to date: keep the items,
            # icons and in-flight decodes as they are.
            return

        self._generation += 1
        self._repopulate(paths)

        if current is not None:
            item = self._items.get(current)
            if item is not None:
                self._list.setCurrentItem(item)
                self._list.scrollToItem(item)

        self._dispatch_thumbnails()

    def _show_placeholder(self, text: str) -> None:
        """Replace the list contents with a single placeholder row."""
        # Invalidate in-flight decodes: their files are no longer shown,
        # so their results would be dropped on arrival anyway.
        self._generation += 1
        self._pending.clear()
        self._items.clear()
        self._thumbnail_cache.clear()
        self._list.clear()

        item = QListWidgetItem(text)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        self._list.addItem(item)
        self._list.doItemsLayout()

    def _repopulate(self, paths: list[str]) -> None:
        """Update the displayed items in place to match ``paths``.

        Removes items for files that vanished, inserts items for new
        files (keeping the scan order) and re-queues kept files that
        have no thumbnail yet (their old-generation jobs, if any, were
        invalidated).  ``_generation`` must already be bumped.
        """
        self._pending.clear()
        new_paths = set(paths)

        for path in list(self._items):
            if path not in new_paths:
                item = self._items.pop(path)
                self._list.takeItem(self._list.row(item))
                self._thumbnail_cache.pop(path, None)

        # A leftover placeholder (only present while no real items
        # exist) always sits at row 0.
        if self._list.count() != len(self._items):
            self._list.takeItem(0)

        cursor = 0
        ordered: dict[str, QListWidgetItem] = {}
        for path in paths:
            item = self._items.get(path)
            if item is None:
                item = self._make_item(path)
                self._list.insertItem(cursor, item)
                self._pending.append(path)
                cursor += 1
            else:
                # Kept file: pass it, and re-decode if its thumbnail is
                # missing.
                cursor = self._list.row(item) + 1
                if path not in self._thumbnail_cache:
                    self._pending.append(path)
            ordered[path] = item

        # Keep _items in display order: an unchanged scan is detected by
        # comparing it against the scan result.
        self._items = ordered

        # The icon-mode grid can go stale when items are inserted or
        # removed around existing ones; force a clean relayout.
        self._list.doItemsLayout()

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
        # list, so the filter is active whenever it has the focus).  Only
        # the press triggers the refresh; the release falls through to
        # the list.
        if (
            obj is self._list
            and isinstance(event, QKeyEvent)
            and event.type() == QEvent.Type.KeyPress
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

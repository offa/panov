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

"""Regression tests for the image browser dock's thumbnail layout.

Pressing F5 used to clear and rebuild the whole list (twice per key
press, even), which left the icon-mode grid stale and made the
thumbnails overlap each other.  These tests refresh the dock -- via F5,
added/removed files and directory switches -- and assert that the
items always sit in a clean, non-overlapping grid.
"""

# pylint: disable=protected-access
# The tests poke at the dock's internals on purpose; the layout can only
# be observed through them.

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from panov.image_browser import _ThumbnailJob, ImageBrowserDock

NUM_FILES = 8
FILE_NAMES = [f"img_{i:02d}.png" for i in range(NUM_FILES)]


@pytest.fixture(scope="module")
def app() -> Iterator[QApplication]:
    """Shared application; QT_QPA_PLATFORM is set in conftest.py."""
    application = QApplication.instance() or QApplication(sys.argv[:1])
    yield application


@pytest.fixture()
def dock_with_images(
    tmp_path: Path, app: QApplication
) -> Iterator[tuple[ImageBrowserDock, Path]]:
    """A visible dock whose NUM_FILES thumbnails have fully settled."""
    _make_images(tmp_path)
    dock = ImageBrowserDock()
    dock.set_directory(str(tmp_path))
    dock.resize(400, 800)
    dock.show()
    app.processEvents()
    _settle(dock, app)
    yield dock, tmp_path
    _settle(dock, app)
    dock.close()
    app.processEvents()


def _make_image(directory: Path, name: str, color: QColor) -> None:
    # 2:1 like a real equirectangular panorama, so the scaled
    # thumbnail is non-square (64x32) as in production.
    image = QImage(48, 24, QImage.Format.Format_RGB888)
    image.fill(color)
    assert image.save(str(directory / name))


def _make_images(directory: Path) -> None:
    for i, name in enumerate(FILE_NAMES):
        _make_image(directory, name, QColor(40 * (i + 1), 20, 200))


def _settle(dock: ImageBrowserDock, app: QApplication) -> None:
    """Block until every in-flight decode finished and was delivered."""
    pool = QThreadPool.globalInstance()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        pool.waitForDone(100)
        app.processEvents()
        if (
            dock._inflight == 0
            and not dock._pending
            and not _ThumbnailJob.active
        ):
            return
    pytest.fail("timed out waiting for thumbnail jobs to settle")


def _assert_grid_is_clean(dock: ImageBrowserDock, app: QApplication) -> None:
    """Every item must occupy its own cell: real area, no overlaps.

    PySide6 does not expose ``itemRect`` on ``QListWidget``, so each
    cell is measured with a temporary marker widget that the view
    positions inside the item's icon area.  A marker smaller than the
    icon cell cannot spill into a neighbouring cell, and the icon-mode
    grid pitch derives from the icon size and font metrics, so the
    markers do not disturb the layout.
    """
    list_ = dock._list
    markers: list[QWidget] = []
    for i in range(list_.count()):
        item = list_.item(i)
        assert item is not None
        marker = QWidget(list_.viewport())
        marker.resize(32, 32)
        list_.setItemWidget(item, marker)
        markers.append(marker)

    list_.updateGeometries()
    list_.viewport().repaint()
    app.processEvents()
    rects = [marker.geometry() for marker in markers]

    # Restore the plain icon items.
    for i in range(list_.count()):
        item = list_.item(i)
        assert item is not None
        list_.setItemWidget(item, None)

    assert rects, "expected at least one item"
    for index, rect in enumerate(rects):
        assert rect.isValid() and rect.width() > 0 and rect.height() > 0, (
            f"item {index} has no area: {rect}"
        )
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not rects[i].intersects(rects[j]), (
                f"items {i} and {j} overlap: {rects[i]} / {rects[j]}"
            )


def test_f5_refresh_keeps_clean_grid(dock_with_images, app: QApplication) -> None:
    """F5 must leave a clean grid (regression: overlapping thumbnails)."""
    dock, _ = dock_with_images
    _assert_grid_is_clean(dock, app)

    # A real F5 keypress is a press plus a release on the focused list.
    dock._list.setFocus()
    QTest.keyClick(dock._list, Qt.Key.Key_F5)
    app.processEvents()
    _settle(dock, app)

    assert dock._list.count() == NUM_FILES
    _assert_grid_is_clean(dock, app)


def test_f5_press_rebuilds_exactly_once(dock_with_images, app: QApplication) -> None:
    """The refresh must run on the key press only, not on the release."""
    dock, _ = dock_with_images
    _settle(dock, app)

    builds = 0
    original = dock._rebuild

    def counting_rebuild(path: str | None) -> None:
        nonlocal builds
        builds += 1
        original(path)

    dock._rebuild = counting_rebuild
    try:
        dock._list.setFocus()
        QTest.keyClick(dock._list, Qt.Key.Key_F5)
        app.processEvents()
    finally:
        dock._rebuild = original

    assert builds == 1


def test_f5_after_adding_file(dock_with_images, app: QApplication) -> None:
    """New files appear in scan order, with their thumbnail, cleanly."""
    dock, directory = dock_with_images
    _settle(dock, app)

    added = "img_04b.png"  # sorts between img_04 and img_05
    _make_image(directory, added, QColor(250, 10, 10))

    dock._list.setFocus()
    QTest.keyClick(dock._list, Qt.Key.Key_F5)
    app.processEvents()
    _settle(dock, app)

    paths: list[str] = []
    for i in range(dock._list.count()):
        item = dock._list.item(i)
        assert item is not None
        data = item.data(Qt.ItemDataRole.UserRole)
        assert isinstance(data, str)
        paths.append(data)
    expected = sorted(
        [str(directory / name) for name in FILE_NAMES]
        + [str(directory / added)],
        key=os.path.basename,
    )
    assert paths == expected
    assert str(directory / added) in dock._thumbnail_cache
    _assert_grid_is_clean(dock, app)


def test_f5_after_removing_file(dock_with_images, app: QApplication) -> None:
    """Vanished files disappear from the list and the thumbnail cache."""
    dock, directory = dock_with_images
    _settle(dock, app)

    removed = directory / "img_02.png"
    removed.unlink()

    dock._list.setFocus()
    QTest.keyClick(dock._list, Qt.Key.Key_F5)
    app.processEvents()
    _settle(dock, app)

    assert dock._list.count() == NUM_FILES - 1
    assert str(removed) not in dock._items
    assert str(removed) not in dock._thumbnail_cache
    _assert_grid_is_clean(dock, app)


def test_directory_switch_keeps_clean_grid(
    tmp_path: Path, app: QApplication
) -> None:
    """Switching directories is a full repopulate and must stay clean."""
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _make_images(dir_a)
    _make_images(dir_b)

    dock = ImageBrowserDock()
    dock.set_directory(str(dir_a))
    dock.resize(400, 800)
    dock.show()
    app.processEvents()
    _settle(dock, app)
    _assert_grid_is_clean(dock, app)

    dock.set_directory(str(dir_b))
    app.processEvents()
    _settle(dock, app)

    assert dock._list.count() == NUM_FILES
    _assert_grid_is_clean(dock, app)

    dock.close()
    app.processEvents()


def test_empty_directory_shows_placeholder(
    tmp_path: Path, app: QApplication
) -> None:
    """An imageless directory shows the placeholder, not a broken grid."""
    empty = tmp_path / "empty"
    empty.mkdir()

    dock = ImageBrowserDock()
    dock.set_directory(str(empty))
    dock.resize(400, 800)
    dock.show()
    app.processEvents()

    assert dock._list.count() == 1
    item = dock._list.item(0)
    assert item is not None
    assert item.text() == "No images found"

    dock.close()
    app.processEvents()

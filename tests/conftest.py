"""Test-suite configuration.

Loaded by pytest before the test modules in this directory are imported,
so it runs ahead of the first PySide6 import / QApplication construction.

Runs the Qt widget tests on the offscreen platform so they work in
headless environments (CI, containers) without a display server.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

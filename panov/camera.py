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

"""Pinhole camera model. Deliberately free of Qt imports."""

from __future__ import annotations


def wrap180(angle: float) -> float:
    """Wrap an angle in degrees into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


class Camera:
    """Pinhole camera described by yaw, pitch and vertical field of view."""

    MIN_FOV = 15.0
    MAX_FOV = 120.0
    DEFAULT_FOV = 75.0

    def __init__(self, fov: float = DEFAULT_FOV) -> None:
        if fov <= 0:
            raise ValueError("FOV must be positive")
        self.yaw = 0.0  # degrees, positive = looking right
        self.pitch = 0.0  # degrees, positive = looking up
        self.fov = self._clamp_fov(fov)

    # -- mutation ---------------------------------------------------------- #

    def set_fov(self, fov: float) -> None:
        self.fov = self._clamp_fov(fov)

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

"""Data models for the layout optimiser (dataclass-based, no Pydantic needed here)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import Polygon as SPolygon


@dataclass
class Cube:
    """A rectangular machine / workstation to be placed on the floor."""
    id: str
    xdim: float          # x-dimension in original orientation (mm)
    ydim: float          # y-dimension in original orientation (mm)
    color: Optional[str] = None
    needs_water: bool = False
    needs_electricity: bool = False

    def dims(self, rot: int) -> tuple[float, float]:
        """Return (width, height) for rotation 0 (as-is) or 1 (90° swap)."""
        return (self.xdim, self.ydim) if rot == 0 else (self.ydim, self.xdim)

    @property
    def area(self) -> float:
        return self.xdim * self.ydim


@dataclass
class Placement:
    """Bottom-left anchor (x, y) and rotation for one cube."""
    cube_id: str
    x: float    # bottom-left x (mm, in normalised polygon coordinate frame)
    y: float    # bottom-left y (mm)
    rot: int    # 0 = original orientation, 1 = 90° rotated


@dataclass
class Solution:
    """A complete assignment of all (or some) cubes to placements."""
    placements: dict[str, Placement] = field(default_factory=dict)

    def copy(self) -> "Solution":
        return Solution(placements=dict(self.placements))


@dataclass
class FlowEdge:
    """Directed material-flow edge between two cubes."""
    src: str
    dst: str
    intensity: float   # arbitrary units (higher = stronger pull)


@dataclass
class ProblemData:
    """All input data needed by the optimiser."""
    outline_polygon: SPolygon
    cubes: list[Cube]
    flows: list[FlowEdge]
    # Pre-extracted polygon vertices for Shapely-free boundary checks.
    # Populated automatically in __post_init__; override only for testing.
    outline_verts: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.outline_verts:
            self.outline_verts = [
                (float(x), float(y))
                for x, y in list(self.outline_polygon.exterior.coords)[:-1]
            ]

    def cube_by_id(self, cube_id: str) -> Cube:
        for c in self.cubes:
            if c.id == cube_id:
                return c
        raise KeyError(f"Cube '{cube_id}' not found")

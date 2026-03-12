"""Load building outline polygon from a DXF file."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

from shapely.geometry import Polygon as SPolygon
from shapely.affinity import translate, rotate
from shapely.validation import make_valid

logger = logging.getLogger(__name__)


def load_outline(dxf_path: str | Path) -> SPolygon:
    """
    Extract the first closed polyline from a DXF file and return it as a
    normalised Shapely Polygon (min-x = 0, min-y = 0).

    Supports LWPOLYLINE and 2-D POLYLINE entity types.
    Repairs self-intersections via shapely.validation.make_valid if needed.
    """
    try:
        import ezdxf
    except ImportError:
        raise ImportError("ezdxf is required: pip install ezdxf")

    path = Path(dxf_path)
    if not path.exists():
        raise FileNotFoundError(f"DXF file not found: {path}")

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    vertices: Optional[list[tuple[float, float]]] = None

    for entity in msp:
        etype = entity.dxftype()
        if etype == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in entity.get_points()]
            if len(pts) >= 3:
                vertices = pts
                logger.info(f"Found LWPOLYLINE with {len(pts)} vertices")
                break
        elif etype == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
            if len(pts) >= 3:
                vertices = pts
                logger.info(f"Found POLYLINE with {len(pts)} vertices")
                break

    if vertices is None:
        raise ValueError(
            "No usable polyline found in DXF. "
            "Expected a LWPOLYLINE or POLYLINE entity with ≥ 3 vertices."
        )

    poly = SPolygon(vertices)

    if not poly.is_valid:
        logger.warning("Outline polygon self-intersects; attempting make_valid() repair.")
        poly = make_valid(poly)
        # make_valid may return a GeometryCollection; keep the largest polygon
        if not isinstance(poly, SPolygon):
            from shapely.geometry import MultiPolygon, GeometryCollection
            if isinstance(poly, (MultiPolygon, GeometryCollection)):
                poly = max(
                    (g for g in poly.geoms if isinstance(g, SPolygon)),
                    key=lambda g: g.area,
                )

    if not isinstance(poly, SPolygon):
        raise ValueError("Could not extract a valid Polygon from the DXF outline.")

    # ── Align principal axis with the x-axis ─────────────────────────────────
    # The minimum-area bounding rectangle gives us the polygon's dominant
    # orientation.  Rotating by -angle so that dominant edge becomes horizontal
    # lets SA place axis-aligned cubes far more efficiently.
    obb = poly.minimum_rotated_rectangle
    obb_coords = list(obb.exterior.coords)
    # Pick the longer edge of the OBB to find the dominant direction
    ex = obb_coords[1][0] - obb_coords[0][0]
    ey = obb_coords[1][1] - obb_coords[0][1]
    edge_len = math.hypot(ex, ey)
    ex2 = obb_coords[2][0] - obb_coords[1][0]
    ey2 = obb_coords[2][1] - obb_coords[1][1]
    edge_len2 = math.hypot(ex2, ey2)
    if edge_len2 > edge_len:
        ex, ey = ex2, ey2
    angle_deg = math.degrees(math.atan2(ey, ex))
    logger.info(f"OBB dominant angle: {angle_deg:.1f}°  → rotating by {-angle_deg:.1f}°")

    cx, cy = poly.centroid.x, poly.centroid.y
    poly = rotate(poly, -angle_deg, origin=(cx, cy), use_radians=False)

    # Normalise: shift so bottom-left corner is at (0, 0)
    minx, miny, maxx, maxy = poly.bounds
    poly = translate(poly, -minx, -miny)
    logger.info(
        f"Outline loaded: {len(vertices)} vertices, "
        f"bounding box {maxx - minx:.0f} × {maxy - miny:.0f} mm, "
        f"area {poly.area / 1e6:.2f} m²"
    )
    return poly

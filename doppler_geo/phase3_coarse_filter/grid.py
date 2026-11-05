"""Candidate Grid Initialization.

Generates a regular metric resolution grid (e.g. 200m) of (lat, lon) coordinates
across a geographic search bounding box using vectorized NumPy operations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from doppler_geo.phase1_ingestion.schemas import BoundingBox
from doppler_geo.coordinates import geodetic_to_enu


@dataclass
class CandidateGrid:
    """Represents a discrete spatial search grid."""
    lats: np.ndarray       # 1D array of latitude coordinates for all points
    lons: np.ndarray       # 1D array of longitude coordinates for all points
    resolution_m: float    # Grid spacing in meters
    ref_lat: float         # Reference origin latitude
    ref_lon: float         # Reference origin longitude

    @property
    def num_points(self) -> int:
        return len(self.lats)

    def to_enu(self, ground_elevations: np.ndarray) -> np.ndarray:
        """Convert all grid points with their ground elevations to ENU coords (N, 3)."""
        e, n, u = geodetic_to_enu(
            self.lats,
            self.lons,
            ground_elevations,
            self.ref_lat,
            self.ref_lon,
            ref_alt_m=0.0,
        )
        return np.column_stack((e, n, u))


def generate_candidate_grid(
    bbox: BoundingBox,
    resolution_m: float = 200.0,
) -> CandidateGrid:
    """Generate a metric-spaced grid across a bounding box.

    Args:
        bbox: BoundingBox defining the search region.
        resolution_m: Grid spacing in meters (default 200m).

    Returns:
        CandidateGrid with vectorized 1D arrays of lat and lon points.
    """
    center_lat = (bbox.min_lat + bbox.max_lat) / 2.0
    center_lon = (bbox.min_lon + bbox.max_lon) / 2.0

    # Metric step conversions
    meters_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * np.radians(center_lat))
    meters_per_deg_lon = 111412.84 * np.cos(np.radians(center_lat))

    delta_lat = resolution_m / meters_per_deg_lat
    delta_lon = resolution_m / max(meters_per_deg_lon, 1e-6)

    lat_steps = np.arange(bbox.min_lat, bbox.max_lat + delta_lat * 0.5, delta_lat)
    lon_steps = np.arange(bbox.min_lon, bbox.max_lon + delta_lon * 0.5, delta_lon)

    lon_grid, lat_grid = np.meshgrid(lon_steps, lat_steps)

    return CandidateGrid(
        lats=lat_grid.ravel(),
        lons=lon_grid.ravel(),
        resolution_m=resolution_m,
        ref_lat=center_lat,
        ref_lon=center_lon,
    )


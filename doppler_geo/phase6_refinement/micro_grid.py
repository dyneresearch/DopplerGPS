"""High-Resolution Micro-Grid Expansion.

Generates high-resolution spatial micro-grids (e.g. 10m to 25m resolution)
around top-scoring coarse candidate clusters to pinpoint exact ground coordinates.
"""

from __future__ import annotations
import numpy as np
from doppler_geo.phase1_ingestion.schemas import BoundingBox
from doppler_geo.phase3_coarse_filter.grid import CandidateGrid, generate_candidate_grid


def generate_micro_grid(
    center_lat: float,
    center_lon: float,
    radius_m: float = 400.0,
    resolution_m: float = 20.0,
) -> CandidateGrid:
    """Generate a high-resolution micro-grid centered at a coarse candidate location.

    Args:
        center_lat: Latitude of candidate peak.
        center_lon: Longitude of candidate peak.
        radius_m: Search radius in meters around the center (default 400m).
        resolution_m: Micro-grid resolution in meters (default 20m, or 10m/25m).

    Returns:
        CandidateGrid with micro-spacing.
    """
    meters_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * np.radians(center_lat))
    meters_per_deg_lon = 111412.84 * np.cos(np.radians(center_lat))

    delta_lat = radius_m / meters_per_deg_lat
    delta_lon = radius_m / max(meters_per_deg_lon, 1e-6)

    bbox = BoundingBox(
        min_lat=center_lat - delta_lat,
        max_lat=center_lat + delta_lat,
        min_lon=center_lon - delta_lon,
        max_lon=center_lon + delta_lon,
    )

    return generate_candidate_grid(bbox, resolution_m=resolution_m)


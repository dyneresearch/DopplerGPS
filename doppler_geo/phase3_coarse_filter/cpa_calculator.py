"""Delayed CPA (Closest Point of Arrival) Calculator.

Calculates the exact moment a flight was closest to each grid point, taking
into account 3D terrain elevation and acoustic propagation delay (c = 343.0 m/s).
Vectorized across thousands of grid points with chunked memory management.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from doppler_geo.phase1_ingestion.dem_handler import DEMHandler
from doppler_geo.coordinates import geodetic_to_enu

SPEED_OF_SOUND = 343.0  # m/s


@dataclass
class FlightCPAResults:
    """Vectorized CPA metrics for a single flight across all grid points."""
    flight_id: str
    min_slant_distances: np.ndarray  # Shape (N,) in meters
    flight_cpa_times: np.ndarray     # Shape (N,) in UTC epoch seconds
    delayed_cpa_times: np.ndarray    # Shape (N,) in UTC epoch seconds
    cpa_velocities: np.ndarray       # Shape (N,) in m/s


class CPACalculator:
    """Vectorized calculation of delayed Closest Point of Arrival (CPA)."""

    def __init__(self, speed_of_sound: float = SPEED_OF_SOUND):
        self.c = speed_of_sound

    def calculate_cpa_for_flight(
        self,
        flight_df: pd.DataFrame,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        grid_elevations: np.ndarray,
        ref_lat: float,
        ref_lon: float,
        chunk_size: int = 4096,
    ) -> FlightCPAResults:
        """Compute delayed CPA metrics for a flight across an array of grid points.

        Args:
            flight_df: DataFrame of a single flight with columns:
                ['flight_id', 'timestamp_utc', 'lat', 'lon', 'altitude_asl_meters', 'velocity_m_s']
            grid_lats: 1D array of grid latitudes.
            grid_lons: 1D array of grid longitudes.
            grid_elevations: 1D array of DEM ground elevations in meters.
            ref_lat: Reference origin latitude for local ENU.
            ref_lon: Reference origin longitude for local ENU.
            chunk_size: Number of grid points to process per vectorized block.

        Returns:
            FlightCPAResults containing min_slant_distances, cpa_times, delayed_cpa_times, and cpa_velocities.
        """
        flight_id = str(flight_df["flight_id"].iloc[0])
        flight_times = flight_df["timestamp_utc"].to_numpy(dtype=np.float64)
        flight_lats = flight_df["lat"].to_numpy(dtype=np.float64)
        flight_lons = flight_df["lon"].to_numpy(dtype=np.float64)
        flight_alts = flight_df["altitude_asl_meters"].to_numpy(dtype=np.float64)
        flight_vels = flight_df["velocity_m_s"].to_numpy(dtype=np.float64)

        # Convert flight path to local ENU
        f_e, f_n, f_u = geodetic_to_enu(
            flight_lats, flight_lons, flight_alts, ref_lat, ref_lon, ref_alt_m=0.0
        )
        flight_enu = np.column_stack((f_e, f_n, f_u))  # Shape (K, 3)

        num_grid = len(grid_lats)
        # Convert all grid points to ENU
        g_e, g_n, g_u = geodetic_to_enu(
            grid_lats, grid_lons, grid_elevations, ref_lat, ref_lon, ref_alt_m=0.0
        )
        grid_enu = np.column_stack((g_e, g_n, g_u))  # Shape (N, 3)

        min_slant = np.empty(num_grid, dtype=np.float64)
        flight_cpa_t = np.empty(num_grid, dtype=np.float64)
        cpa_v = np.empty(num_grid, dtype=np.float64)

        # Process in chunks to maintain optimal L3 cache locality
        for start_idx in range(0, num_grid, chunk_size):
            end_idx = min(start_idx + chunk_size, num_grid)
            sub_grid = grid_enu[start_idx:end_idx]  # Shape (M, 3)

            # diff shape: (M, K, 3)
            # diff = sub_grid[:, np.newaxis, :] - flight_enu[np.newaxis, :, :]
            diff = sub_grid[:, np.newaxis, :] - flight_enu[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=-1)  # Shape (M, K)

            min_idx = np.argmin(dist_sq, axis=1)  # Shape (M,)
            min_dist = np.sqrt(dist_sq[np.arange(len(sub_grid)), min_idx])

            min_slant[start_idx:end_idx] = min_dist
            flight_cpa_t[start_idx:end_idx] = flight_times[min_idx]
            cpa_v[start_idx:end_idx] = flight_vels[min_idx]

        delayed_cpa_t = flight_cpa_t + (min_slant / self.c)

        return FlightCPAResults(
            flight_id=flight_id,
            min_slant_distances=min_slant,
            flight_cpa_times=flight_cpa_t,
            delayed_cpa_times=delayed_cpa_t,
            cpa_velocities=cpa_v,
        )


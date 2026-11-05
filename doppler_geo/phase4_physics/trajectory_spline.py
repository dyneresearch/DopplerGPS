"""Trajectory Interpolation with Cubic Splines.

Interpolates sporadic ADS-B telemetry into smooth, high-precision 3D ENU
trajectories and analytical velocity vectors using scipy.interpolate.CubicSpline.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from doppler_geo.coordinates import geodetic_to_enu


@dataclass
class InterpolatedTrajectory:
    """Sampled trajectory in local ENU coordinates over time."""
    times: np.ndarray       # 1D array of emission timestamps (epoch seconds)
    enu_positions: np.ndarray  # Shape (N, 3) in meters (East, North, Up)
    enu_velocities: np.ndarray # Shape (N, 3) in m/s (d/dt East, d/dt North, d/dt Up)
    ref_lat: float
    ref_lon: float
    ref_alt: float


class TrajectorySpline:
    """Fits cubic splines to ADS-B telemetry for smooth position and velocity evaluation."""

    def __init__(
        self,
        flight_df: pd.DataFrame,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float = 0.0,
    ):
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt

        # Ensure sorted and unique timestamps
        df_sorted = flight_df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])
        if len(df_sorted) < 2:
            raise ValueError("Flight trajectory requires at least 2 distinct timestamp points for spline interpolation.")

        raw_times = df_sorted["timestamp_utc"].to_numpy(dtype=np.float64)
        raw_lats = df_sorted["lat"].to_numpy(dtype=np.float64)
        raw_lons = df_sorted["lon"].to_numpy(dtype=np.float64)
        raw_alts = df_sorted["altitude_asl_meters"].to_numpy(dtype=np.float64)

        e, n, u = geodetic_to_enu(
            raw_lats, raw_lons, raw_alts, self.ref_lat, self.ref_lon, self.ref_alt
        )
        self.raw_times = raw_times
        self.raw_enu = np.column_stack((e, n, u))

        # Fit cubic spline for (East, North, Up) coordinates vs time
        # bc_type='natural' ensures stable extrapolation if needed
        self.pos_spline = CubicSpline(self.raw_times, self.raw_enu, axis=0, bc_type="natural")
        self.vel_spline = self.pos_spline.derivative(nu=1)

    @property
    def time_range(self) -> Tuple[float, float]:
        return float(self.raw_times[0]), float(self.raw_times[-1])

    def sample_window(
        self,
        center_time_utc: float,
        window_half_sec: float = 60.0,
        step_sec: float = 1.0,
    ) -> InterpolatedTrajectory:
        """Sample the flight trajectory at regular intervals around a center time.

        Args:
            center_time_utc: Center timestamp (e.g. CPA time in UTC).
            window_half_sec: Half-window width in seconds (default 60s for 120s window).
            step_sec: Sampling step in seconds (e.g. 1.0s or 0.5s).

        Returns:
            InterpolatedTrajectory with positions and exact analytical velocities.
        """
        t_start = center_time_utc - window_half_sec
        t_end = center_time_utc + window_half_sec
        t_samples = np.arange(t_start, t_end + step_sec * 0.5, step_sec, dtype=np.float64)

        positions = self.pos_spline(t_samples)
        velocities = self.vel_spline(t_samples)

        return InterpolatedTrajectory(
            times=t_samples,
            enu_positions=positions,
            enu_velocities=velocities,
            ref_lat=self.ref_lat,
            ref_lon=self.ref_lon,
            ref_alt=self.ref_alt,
        )


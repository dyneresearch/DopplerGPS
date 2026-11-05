"""The Core Physics Engine: Doppler Modeler.

Implements the exact 3D vector math and emission-delay physics to generate
theoretical Doppler frequency curves over observer reception time.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from doppler_geo.phase3_coarse_filter.cpa_calculator import SPEED_OF_SOUND
from doppler_geo.phase4_physics.trajectory_spline import (
    InterpolatedTrajectory,
    TrajectorySpline,
)
from doppler_geo.coordinates import geodetic_to_enu


@dataclass
class DopplerCurve:
    """The computed physical Doppler frequency curve as observed on the ground."""
    t_obs: np.ndarray             # Arrival timestamps at observer in epoch seconds
    f_obs: np.ndarray             # Observed frequencies in Hz
    t_emit: np.ndarray            # Emission timestamps from aircraft in epoch seconds
    slant_distances: np.ndarray   # 3D distance R(t) in meters
    radial_velocities: np.ndarray # Radial velocity V_r(t) in m/s
    f0_baseline_hz: float         # Fundamental baseline frequency F0 in Hz

    @property
    def cpa_obs_time(self) -> float:
        """Observation time corresponding to closest point of approach (min slant distance)."""
        min_idx = np.argmin(self.slant_distances)
        return float(self.t_obs[min_idx])

    @property
    def min_slant_distance(self) -> float:
        return float(np.min(self.slant_distances))


class DopplerModeler:
    """Physics-based Doppler curve modeler with line-of-sight unit vector math."""

    def __init__(self, speed_of_sound: float = SPEED_OF_SOUND):
        self.c = speed_of_sound

    def model_curve(
        self,
        trajectory: InterpolatedTrajectory,
        obs_lat: float,
        obs_lon: float,
        obs_elevation_m: float,
        f0_baseline_hz: float,
    ) -> DopplerCurve:
        """Model the Doppler shift curve for a stationary observer on the ground.

        Args:
            trajectory: InterpolatedTrajectory with ENU positions and velocities.
            obs_lat: Observer ground latitude in WGS84.
            obs_lon: Observer ground longitude in WGS84.
            obs_elevation_m: Observer ground elevation ASL in meters.
            f0_baseline_hz: Aircraft engine/propeller base frequency F0 in Hz.

        Returns:
            DopplerCurve containing (t_obs, f_obs, t_emit, slant_distances, radial_velocities).
        """
        # Convert observer location to local ENU coordinates using trajectory's origin
        obs_e, obs_n, obs_u = geodetic_to_enu(
            obs_lat,
            obs_lon,
            obs_elevation_m,
            ref_lat=trajectory.ref_lat,
            ref_lon=trajectory.ref_lon,
            ref_alt=trajectory.ref_alt,
        )
        observer_enu = np.array([obs_e, obs_n, obs_u], dtype=np.float64)  # Shape (3,)

        # 1. Distance Vector: D(t) = P(t) - O
        # P(t) has shape (N, 3), O has shape (3,)
        dist_vec = trajectory.enu_positions - observer_enu  # Shape (N, 3)

        # 2. Slant Distance: R(t) = ||D(t)||
        slant_dist = np.linalg.norm(dist_vec, axis=1)  # Shape (N,)
        # Avoid division by zero if aircraft flies through microphone
        slant_dist_safe = np.maximum(slant_dist, 1e-6)

        # 3. Line-of-Sight Unit Vector: u_hat(t) = D(t) / R(t)
        u_hat = dist_vec / slant_dist_safe[:, np.newaxis]  # Shape (N, 3)

        # 4. Plane Velocity Vector: V(t) = d/dt P(t)
        v_vec = trajectory.enu_velocities  # Shape (N, 3)

        # 5. Radial Velocity: V_r(t) = V(t) . u_hat(t) (Dot product)
        # Positive = plane moving away from observer; negative = moving toward observer
        v_r = np.sum(v_vec * u_hat, axis=1)  # Shape (N,)

        # 6. Doppler Shift: F_obs = F0 * (c / (c + V_r))
        f_obs = f0_baseline_hz * (self.c / (self.c + v_r))

        # 7. Arrival Time at Observer: t_obs = t_emit + R / c
        t_obs = trajectory.times + (slant_dist / self.c)

        # Sort strictly by arrival time t_obs (in case of slight ordering anomalies)
        sort_order = np.argsort(t_obs)

        return DopplerCurve(
            t_obs=t_obs[sort_order],
            f_obs=f_obs[sort_order],
            t_emit=trajectory.times[sort_order],
            slant_distances=slant_dist[sort_order],
            radial_velocities=v_r[sort_order],
            f0_baseline_hz=f0_baseline_hz,
        )


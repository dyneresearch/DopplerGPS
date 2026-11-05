"""The Multi-Event Offset & Silence Filter.

Filters candidate grid points by testing whether two flight passes match the
macroscopic speed, slant distance, and temporal delay constraints of the audio
events, and enforces the 10km acoustic silence filter between events.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from doppler_geo.phase2_audio.measurement_interface import (
    AudioEventMeasurement,
    MeasurementConfig,
)
from doppler_geo.phase3_coarse_filter.grid import CandidateGrid
from doppler_geo.phase3_coarse_filter.cpa_calculator import (
    CPACalculator,
    FlightCPAResults,
)


@dataclass
class CoarseCandidateMatch:
    """A surviving candidate grid point matching multi-event geometry."""
    lat: float
    lon: float
    elevation_m: float
    flight_id_1: str
    flight_id_2: str
    flight_cpa_1: float
    flight_cpa_2: float
    delayed_cpa_1: float
    delayed_cpa_2: float
    slant_dist_1: float
    slant_dist_2: float
    velocity_1: float
    velocity_2: float
    dt_observed: float

    def to_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "elevation_m": self.elevation_m,
            "flight_id_1": self.flight_id_1,
            "flight_id_2": self.flight_id_2,
            "flight_cpa_1": self.flight_cpa_1,
            "flight_cpa_2": self.flight_cpa_2,
            "delayed_cpa_1": self.delayed_cpa_1,
            "delayed_cpa_2": self.delayed_cpa_2,
            "slant_dist_1": self.slant_dist_1,
            "slant_dist_2": self.slant_dist_2,
            "velocity_1": self.velocity_1,
            "velocity_2": self.velocity_2,
            "dt_observed": self.dt_observed,
        }


class MultiEventFilter:
    """Evaluates candidate grid points against dual-event timing and the silence constraint."""

    def __init__(
        self,
        time_tolerance_s: float = 5.0,
        silence_distance_threshold_m: float = 10000.0,
    ):
        self.time_tolerance_s = time_tolerance_s
        self.silence_threshold_m = silence_distance_threshold_m
        self.cpa_calculator = CPACalculator()

    def filter_grid(
        self,
        grid: CandidateGrid,
        grid_elevations: np.ndarray,
        flight_dfs: Dict[str, pd.DataFrame],
        measurement_config: MeasurementConfig,
    ) -> List[CoarseCandidateMatch]:
        """Run the coarse spatial filter across all grid points and flight combinations.

        Args:
            grid: CandidateGrid object with lats and lons.
            grid_elevations: 1D array of DEM ground elevations in meters.
            flight_dfs: Dictionary mapping flight_id -> flight trajectory DataFrame.
            measurement_config: MeasurementConfig containing Event 1 and Event 2 specifications.

        Returns:
            List of surviving CoarseCandidateMatch objects.
        """
        if len(measurement_config.events) < 2:
            raise ValueError("MultiEventFilter requires at least two events in MeasurementConfig.")

        event1 = measurement_config.events[0]
        event2 = measurement_config.events[1]
        expected_dt = event2.cpa_time_in_audio - event1.cpa_time_in_audio

        # Step 1: Precompute CPA results for all flights across the grid
        cpa_results: Dict[str, FlightCPAResults] = {}
        for f_id, f_df in flight_dfs.items():
            cpa_results[f_id] = self.cpa_calculator.calculate_cpa_for_flight(
                flight_df=f_df,
                grid_lats=grid.lats,
                grid_lons=grid.lons,
                grid_elevations=grid_elevations,
                ref_lat=grid.ref_lat,
                ref_lon=grid.ref_lon,
            )

        surviving_candidates: List[CoarseCandidateMatch] = []
        flight_ids = list(flight_dfs.keys())

        # Step 2: Iterate over potential pairs of flights for Event 1 and Event 2
        for f1_id in flight_ids:
            res1 = cpa_results[f1_id]

            # Vectorized check for Event 1 criteria
            mask_e1 = (
                (res1.min_slant_distances >= event1.est_slant_distance_range_m[0])
                & (res1.min_slant_distances <= event1.est_slant_distance_range_m[1])
                & (res1.cpa_velocities >= event1.est_velocity_range_m_s[0])
                & (res1.cpa_velocities <= event1.est_velocity_range_m_s[1])
            )

            if not np.any(mask_e1):
                continue

            for f2_id in flight_ids:
                if f1_id == f2_id:
                    continue

                res2 = cpa_results[f2_id]

                # Vectorized check for Event 2 criteria
                mask_e2 = (
                    (res2.min_slant_distances >= event2.est_slant_distance_range_m[0])
                    & (res2.min_slant_distances <= event2.est_slant_distance_range_m[1])
                    & (res2.cpa_velocities >= event2.est_velocity_range_m_s[0])
                    & (res2.cpa_velocities <= event2.est_velocity_range_m_s[1])
                )

                # Combined mask for both events
                both_mask = mask_e1 & mask_e2
                if not np.any(both_mask):
                    continue

                # Check temporal offset: delayed_cpa_2 - delayed_cpa_1 ~ expected_dt
                dt = res2.delayed_cpa_times - res1.delayed_cpa_times
                dt_mask = both_mask & (np.abs(dt - expected_dt) <= self.time_tolerance_s)

                surviving_indices = np.where(dt_mask)[0]
                if len(surviving_indices) == 0:
                    continue

                # Step 3: Enforce Silence Filter on surviving candidate points
                # Verify that NO OTHER propeller flight came within 10 km between Event 1 & Event 2
                other_flights = [fid for fid in flight_ids if fid not in (f1_id, f2_id)]

                for idx in surviving_indices:
                    t1_delay = res1.delayed_cpa_times[idx]
                    t2_delay = res2.delayed_cpa_times[idx]
                    t_min = min(t1_delay, t2_delay)
                    t_max = max(t1_delay, t2_delay)

                    is_silent = True
                    for other_id in other_flights:
                        other_res = cpa_results[other_id]
                        # Check if this other flight passed during the window
                        other_cpa_t = other_res.delayed_cpa_times[idx]
                        if t_min <= other_cpa_t <= t_max:
                            if other_res.min_slant_distances[idx] < self.silence_threshold_m:
                                is_silent = False
                                break

                    if is_silent:
                        surviving_candidates.append(
                            CoarseCandidateMatch(
                                lat=float(grid.lats[idx]),
                                lon=float(grid.lons[idx]),
                                elevation_m=float(grid_elevations[idx]),
                                flight_id_1=f1_id,
                                flight_id_2=f2_id,
                                flight_cpa_1=float(res1.flight_cpa_times[idx]),
                                flight_cpa_2=float(res2.flight_cpa_times[idx]),
                                delayed_cpa_1=float(t1_delay),
                                delayed_cpa_2=float(t2_delay),
                                slant_dist_1=float(res1.min_slant_distances[idx]),
                                slant_dist_2=float(res2.min_slant_distances[idx]),
                                velocity_1=float(res1.cpa_velocities[idx]),
                                velocity_2=float(res2.cpa_velocities[idx]),
                                dt_observed=float(dt[idx]),
                            )
                        )

        return surviving_candidates


"""Unit tests for Phase 3: The Coarse Spatial Filter (Grid Generation)."""

import pytest
import numpy as np
import pandas as pd

from doppler_geo.phase1_ingestion.schemas import BoundingBox
from doppler_geo.phase2_audio.measurement_interface import (
    AudioEventMeasurement,
    MeasurementConfig,
)
from doppler_geo.phase3_coarse_filter import (
    CandidateGrid,
    generate_candidate_grid,
    CPACalculator,
    MultiEventFilter,
    SPEED_OF_SOUND,
)


def test_candidate_grid_generation():
    """Verify that candidate grid creates regular metric points within bounding box."""
    bbox = BoundingBox(min_lat=40.0, max_lat=40.05, min_lon=-75.05, max_lon=-75.0)
    grid = generate_candidate_grid(bbox, resolution_m=200.0)

    assert grid.num_points > 100
    assert np.all(grid.lats >= bbox.min_lat)
    assert np.all(grid.lats <= bbox.max_lat + 0.005)
    assert np.all(grid.lons >= bbox.min_lon)
    assert np.all(grid.lons <= bbox.max_lon + 0.005)

    elevations = np.zeros(grid.num_points)
    enu = grid.to_enu(elevations)
    assert enu.shape == (grid.num_points, 3)


def test_delayed_cpa_calculator_math():
    """Verify CPA distance, time, and propagation delay calculations against known geometry."""
    # Target observer at (40.0, -75.0, elevation 100m)
    # Flight passes directly overhead at altitude 1100m ASL (relative height = 1000m) at t = 1000.0s
    calc = CPACalculator(speed_of_sound=343.0)

    flight_times = np.array([990.0, 1000.0, 1010.0])
    flight_lats = np.array([39.99, 40.0, 40.01])
    flight_lons = np.array([-75.0, -75.0, -75.0])
    flight_alts = np.array([1100.0, 1100.0, 1100.0])
    flight_vels = np.array([100.0, 100.0, 100.0])

    flight_df = pd.DataFrame({
        "flight_id": ["TEST_FLIGHT"] * 3,
        "timestamp_utc": flight_times,
        "lat": flight_lats,
        "lon": flight_lons,
        "altitude_asl_meters": flight_alts,
        "velocity_m_s": flight_vels,
        "heading": [0.0, 0.0, 0.0],
    })

    grid_lats = np.array([40.0])
    grid_lons = np.array([-75.0])
    grid_elevations = np.array([100.0])

    res = calc.calculate_cpa_for_flight(
        flight_df=flight_df,
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        grid_elevations=grid_elevations,
        ref_lat=40.0,
        ref_lon=-75.0,
    )

    # Directly overhead: slant distance should be exactly 1000m
    assert np.isclose(res.min_slant_distances[0], 1000.0, atol=1.0)
    assert np.isclose(res.flight_cpa_times[0], 1000.0)
    expected_delayed_time = 1000.0 + (1000.0 / 343.0)
    assert np.isclose(res.delayed_cpa_times[0], expected_delayed_time, atol=0.01)
    assert np.isclose(res.cpa_velocities[0], 100.0)


def test_multi_event_filter_and_silence_filter():
    """Verify multi-event filtering with locked timing and 10km silence check."""
    # Setup observer location
    obs_lat, obs_lon = 40.0, -75.0
    grid = CandidateGrid(
        lats=np.array([obs_lat, 40.1]),
        lons=np.array([obs_lon, -75.1]),
        resolution_m=200.0,
        ref_lat=obs_lat,
        ref_lon=obs_lon,
    )
    elevations = np.array([100.0, 100.0])

    # Flight 1 passes obs at t = 1000s, slant distance ~ 3000m (alt 3100m), speed 100 m/s
    f1_df = pd.DataFrame({
        "flight_id": ["F1"] * 3,
        "timestamp_utc": [990.0, 1000.0, 1010.0],
        "lat": [39.99, obs_lat, 40.01],
        "lon": [obs_lon, obs_lon, obs_lon],
        "altitude_asl_meters": [3100.0, 3100.0, 3100.0],
        "velocity_m_s": [100.0, 100.0, 100.0],
        "heading": [0.0, 0.0, 0.0],
    })

    # Flight 2 passes obs at t = 1920s (920s later), slant distance ~ 2000m (alt 2100m), speed 95 m/s
    f2_df = pd.DataFrame({
        "flight_id": ["F2"] * 3,
        "timestamp_utc": [1910.0, 1920.0, 1930.0],
        "lat": [obs_lat, obs_lat, obs_lat],
        "lon": [-75.01, obs_lon, -74.99],
        "altitude_asl_meters": [2100.0, 2100.0, 2100.0],
        "velocity_m_s": [95.0, 95.0, 95.0],
        "heading": [90.0, 90.0, 90.0],
    })

    # Noise intruder flight that violates silence filter: passes within 5km at t = 1500s
    intruder_df = pd.DataFrame({
        "flight_id": ["INTRUDER"] * 3,
        "timestamp_utc": [1490.0, 1500.0, 1510.0],
        "lat": [obs_lat, obs_lat, obs_lat],
        "lon": [obs_lon, obs_lon, obs_lon],
        "altitude_asl_meters": [4000.0, 4000.0, 4000.0],  # 3900m slant dist (< 10km)
        "velocity_m_s": [100.0, 100.0, 100.0],
        "heading": [90.0, 90.0, 90.0],
    })

    # Measurement config expecting Event 1 and Event 2 with ~920s gap
    config = MeasurementConfig(events=[
        AudioEventMeasurement(
            event_id=1,
            cpa_time_in_audio=15.0,
            f0_baseline_hz=130.0,
            est_velocity_range_m_s=(90.0, 110.0),
            est_slant_distance_range_m=(2500.0, 3500.0),
        ),
        AudioEventMeasurement(
            event_id=2,
            cpa_time_in_audio=935.0,
            f0_baseline_hz=110.0,
            est_velocity_range_m_s=(85.0, 105.0),
            est_slant_distance_range_m=(1500.0, 2500.0),
        ),
    ])

    filter_engine = MultiEventFilter(time_tolerance_s=5.0, silence_distance_threshold_m=10000.0)

    # Test 1: With intruder, candidate should be dropped by the silence filter
    flight_dict_with_intruder = {"F1": f1_df, "F2": f2_df, "INTRUDER": intruder_df}
    surviving = filter_engine.filter_grid(grid, elevations, flight_dict_with_intruder, config)
    assert len(surviving) == 0

    # Test 2: Without intruder, target point survives
    flight_dict_clean = {"F1": f1_df, "F2": f2_df}
    surviving_clean = filter_engine.filter_grid(grid, elevations, flight_dict_clean, config)
    assert len(surviving_clean) == 1
    assert surviving_clean[0].flight_id_1 == "F1"
    assert surviving_clean[0].flight_id_2 == "F2"
    assert np.isclose(surviving_clean[0].lat, obs_lat)
    assert np.isclose(surviving_clean[0].lon, obs_lon)

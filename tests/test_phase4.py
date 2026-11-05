"""Unit tests for Phase 4: The Core Physics Engine (Doppler Modeler)."""

import pytest
import numpy as np
import pandas as pd

from doppler_geo.phase4_physics import (
    TrajectorySpline,
    DopplerModeler,
    DopplerCurve,
)


@pytest.fixture
def north_flying_aircraft():
    """Create a straight-and-level flight flying due North at 100 m/s at 1000m altitude."""
    # Origin at (40.0, -75.0, alt 0)
    # Flying along lon = -75.0 from lat 39.95 to 40.05
    # 1 degree lat ~ 111,000m, so 0.001 deg ~ 111m
    # At 100 m/s, aircraft travels ~100m per second
    t_cpa = 1000.0
    times = np.arange(900.0, 1101.0, 10.0)  # Telemetry every 10s
    dt = times - t_cpa
    y_meters = dt * 100.0  # North coordinate in meters
    lat_degrees = 40.0 + (y_meters / 111000.0)
    lon_degrees = np.full_like(lat_degrees, -75.0)
    alt_meters = np.full_like(lat_degrees, 1000.0)
    vel_m_s = np.full_like(lat_degrees, 100.0)

    df = pd.DataFrame({
        "flight_id": ["TEST_PROP"] * len(times),
        "timestamp_utc": times,
        "lat": lat_degrees,
        "lon": lon_degrees,
        "altitude_asl_meters": alt_meters,
        "velocity_m_s": vel_m_s,
        "heading": [0.0] * len(times),
    })
    return df, t_cpa


def test_trajectory_spline_interpolation(north_flying_aircraft):
    """Verify cubic spline resamples trajectory to 1.0s and computes exact velocity."""
    df, t_cpa = north_flying_aircraft
    spline = TrajectorySpline(df, ref_lat=40.0, ref_lon=-75.0, ref_alt=0.0)

    sampled = spline.sample_window(center_time_utc=t_cpa, window_half_sec=60.0, step_sec=1.0)
    assert len(sampled.times) == 121
    assert np.isclose(sampled.times[60], t_cpa)

    # Velocity in North direction should be ~100 m/s, East and Up should be ~0
    vel_east = sampled.enu_velocities[:, 0]
    vel_north = sampled.enu_velocities[:, 1]
    vel_up = sampled.enu_velocities[:, 2]

    assert np.allclose(vel_east, 0.0, atol=1.0)
    assert np.allclose(vel_north, 100.0, atol=2.0)
    assert np.allclose(vel_up, 0.0, atol=1.0)


def test_doppler_modeler_curve_physics(north_flying_aircraft):
    """Verify Doppler shift physics (higher freq approaching, F0 at CPA, lower freq receding)."""
    df, t_cpa = north_flying_aircraft
    spline = TrajectorySpline(df, ref_lat=40.0, ref_lon=-75.0, ref_alt=0.0)
    sampled = spline.sample_window(center_time_utc=t_cpa, window_half_sec=60.0, step_sec=1.0)

    modeler = DopplerModeler(speed_of_sound=343.0)
    f0 = 100.0  # 100 Hz baseline

    # Observer at (40.0, -75.0, elevation 0)
    curve = modeler.model_curve(
        trajectory=sampled,
        obs_lat=40.0,
        obs_lon=-75.0,
        obs_elevation_m=0.0,
        f0_baseline_hz=f0,
    )

    assert isinstance(curve, DopplerCurve)
    assert len(curve.t_obs) == 121

    # 1. Curve should be monotonically decreasing in frequency as plane approaches and passes
    diffs = np.diff(curve.f_obs)
    assert np.all(diffs <= 0.0)

    # 2. At start of window (approaching), frequency should be significantly higher than F0
    assert curve.f_obs[0] > f0

    # 3. At end of window (receding), frequency should be significantly lower than F0
    assert curve.f_obs[-1] < f0

    # 4. At CPA (minimum slant distance), radial velocity Vr should be ~0, so F_obs ~ F0
    min_idx = np.argmin(curve.slant_distances)
    assert np.isclose(curve.radial_velocities[min_idx], 0.0, atol=1.0)
    assert np.isclose(curve.f_obs[min_idx], f0, atol=0.5)

    # 5. Check acoustic propagation delay: overhead slant distance is 1000m
    # Delay = 1000 / 343 ~ 2.915 seconds
    cpa_emit_time = curve.t_emit[min_idx]
    cpa_obs_time = curve.t_obs[min_idx]
    assert np.isclose(cpa_obs_time - cpa_emit_time, 1000.0 / 343.0, atol=0.05)


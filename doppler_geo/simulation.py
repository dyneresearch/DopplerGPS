"""Simulation and Synthetic Dataset Generator for Project Doppler-Geo.

Generates realistic historical ADS-B flight logs, Digital Elevation Models,
and physical ambient audio recordings with Doppler frequency curves.
"""

from __future__ import annotations
import os
import pathlib
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import soundfile as sf

from doppler_geo.phase1_ingestion import BoundingBox, SyntheticDEMHandler
from doppler_geo.phase2_audio import AudioEventMeasurement, MeasurementConfig
from doppler_geo.phase4_physics.doppler_modeler import DopplerModeler
from doppler_geo.phase4_physics.trajectory_spline import TrajectorySpline


def create_simulation_data(tmp_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Setup full simulation datasets (ADSB parquet, audio file, measurement config)."""
    tmp_path = pathlib.Path(tmp_path)
    true_lat = 40.0000
    true_lon = -75.0000
    true_elevation = 120.0  # meters ASL
    c = 343.0

    # 1. Flight 1: Flying due East at 95 m/s at altitude 2620m ASL (relative height = 2500m)
    # Passes overhead / CPA at t_utc = 1015.0s
    t_cpa_1 = 1015.0
    f1_times = np.arange(950.0, 1081.0, 5.0)  # Telemetry every 5s
    dt1 = f1_times - t_cpa_1
    x_meters_1 = dt1 * 95.0
    f1_lats = np.full_like(f1_times, true_lat)
    f1_lons = true_lon + (x_meters_1 / (111412.0 * np.cos(np.radians(true_lat))))
    f1_alts = np.full_like(f1_times, 2620.0)
    f1_vels = np.full_like(f1_times, 95.0)

    # 2. Flight 2: Flying due North at 90 m/s at altitude 2120m ASL (relative height = 2000m)
    # Passes overhead / CPA at t_utc = 1075.0s (60s later for fast test duration)
    t_cpa_2 = 1075.0
    f2_times = np.arange(1010.0, 1141.0, 5.0)
    dt2 = f2_times - t_cpa_2
    y_meters_2 = dt2 * 90.0
    f2_lats = true_lat + (y_meters_2 / 111132.0)
    f2_lons = np.full_like(f2_times, true_lon)
    f2_alts = np.full_like(f2_times, 2120.0)
    f2_vels = np.full_like(f2_times, 90.0)

    # 3. Distractor Flight 3: Jet (B738) that should be filtered out
    jet_times = np.arange(950.0, 1141.0, 10.0)
    jet_lats = np.full_like(jet_times, true_lat)
    jet_lons = np.full_like(jet_times, true_lon)

    # Assemble combined ADSB parquet
    df_f1 = pd.DataFrame({
        "flight_id": ["N172_E1"] * len(f1_times),
        "type_code": ["C172"] * len(f1_times),
        "timestamp_utc": f1_times,
        "lat": f1_lats,
        "lon": f1_lons,
        "altitude_asl_meters": f1_alts,
        "velocity_m_s": f1_vels,
        "heading": [90.0] * len(f1_times),
    })
    df_f2 = pd.DataFrame({
        "flight_id": ["PA28_E2"] * len(f2_times),
        "type_code": ["PA28"] * len(f2_times),
        "timestamp_utc": f2_times,
        "lat": f2_lats,
        "lon": f2_lons,
        "altitude_asl_meters": f2_alts,
        "velocity_m_s": f2_vels,
        "heading": [0.0] * len(f2_times),
    })
    df_jet = pd.DataFrame({
        "flight_id": ["UAL999"] * len(jet_times),
        "type_code": ["B738"] * len(jet_times),
        "timestamp_utc": jet_times,
        "lat": jet_lats,
        "lon": jet_lons,
        "altitude_asl_meters": [10000.0] * len(jet_times),
        "velocity_m_s": [220.0] * len(jet_times),
        "heading": [45.0] * len(jet_times),
    })
    adsb_df = pd.concat([df_f1, df_f2, df_jet], ignore_index=True)
    parquet_path = str(tmp_path / "adsb_simulation.parquet")
    adsb_df.to_parquet(parquet_path, index=False)

    # 4. Generate Ground Truth Physical Doppler Curves
    spline1 = TrajectorySpline(df_f1, ref_lat=true_lat, ref_lon=true_lon)
    spline2 = TrajectorySpline(df_f2, ref_lat=true_lat, ref_lon=true_lon)
    traj1 = spline1.sample_window(t_cpa_1, window_half_sec=40.0, step_sec=0.25)
    traj2 = spline2.sample_window(t_cpa_2, window_half_sec=40.0, step_sec=0.25)

    modeler = DopplerModeler(speed_of_sound=c)
    f0_1 = 130.0
    f0_2 = 110.0
    curve1 = modeler.model_curve(traj1, true_lat, true_lon, true_elevation, f0_1)
    curve2 = modeler.model_curve(traj2, true_lat, true_lon, true_elevation, f0_2)

    # 5. Synthesize Audio Recording (simulated recording starts at t_utc = 1000.0)
    rec_start_utc = 1000.0
    sr = 2000
    audio_duration = 110.0  # 110 seconds of audio
    audio_t = np.linspace(0.0, audio_duration, int(sr * audio_duration), endpoint=False)
    audio_utc = rec_start_utc + audio_t

    # Interpolate instantaneous frequency onto audio time grid
    f1_interp = np.interp(audio_utc, curve1.t_obs, curve1.f_obs, left=f0_1, right=f0_1)
    f2_interp = np.interp(audio_utc, curve2.t_obs, curve2.f_obs, left=f0_2, right=f0_2)

    # Window signals around each event's pass
    win1 = np.exp(-((audio_utc - curve1.cpa_obs_time) / 12.0) ** 2)
    win2 = np.exp(-((audio_utc - curve2.cpa_obs_time) / 12.0) ** 2)

    phase1 = 2 * np.pi * np.cumsum(f1_interp) / sr
    phase2 = 2 * np.pi * np.cumsum(f2_interp) / sr

    audio_samples = 0.6 * np.sin(phase1) * win1 + 0.6 * np.sin(phase2) * win2
    np.random.seed(123)
    audio_samples += 0.02 * np.random.randn(len(audio_samples))

    audio_path = str(tmp_path / "ambient_flight_audio.wav")
    sf.write(audio_path, audio_samples, sr)

    # 6. Manual Measurement Config
    audio_cpa_1 = float(curve1.cpa_obs_time - rec_start_utc)
    audio_cpa_2 = float(curve2.cpa_obs_time - rec_start_utc)

    config = MeasurementConfig(events=[
        AudioEventMeasurement(
            event_id=1,
            cpa_time_in_audio=round(audio_cpa_1, 1),
            f0_baseline_hz=f0_1,
            est_velocity_range_m_s=(85.0, 105.0),
            est_slant_distance_range_m=(2000.0, 3000.0),
        ),
        AudioEventMeasurement(
            event_id=2,
            cpa_time_in_audio=round(audio_cpa_2, 1),
            f0_baseline_hz=f0_2,
            est_velocity_range_m_s=(80.0, 100.0),
            est_slant_distance_range_m=(1500.0, 2500.0),
        ),
    ])

    search_bbox = BoundingBox(
        min_lat=true_lat - 0.02,
        max_lat=true_lat + 0.02,
        min_lon=true_lon - 0.02,
        max_lon=true_lon + 0.02,
    )

    dem = SyntheticDEMHandler(base_elevation_m=true_elevation)

    return {
        "parquet_path": parquet_path,
        "audio_path": audio_path,
        "config": config,
        "bbox": search_bbox,
        "dem": dem,
        "true_lat": true_lat,
        "true_lon": true_lon,
        "true_elevation": true_elevation,
    }


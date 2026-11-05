"""Unit tests for Phase 6: Refinement & Visualization."""

import os
import json
import pytest
import numpy as np
import pandas as pd

from doppler_geo.phase2_audio.spectrogram import SpectrogramGenerator
from doppler_geo.phase4_physics.doppler_modeler import DopplerCurve
from doppler_geo.phase6_refinement import (
    generate_micro_grid,
    GeolocationCandidate,
    export_geojson,
    generate_folium_map,
    plot_spectrogram_with_curves,
)


def test_micro_grid_generation():
    """Verify micro-grid expansion generates fine-resolution grid around center."""
    center_lat, center_lon = 40.0, -75.0
    grid = generate_micro_grid(center_lat, center_lon, radius_m=300.0, resolution_m=20.0)

    assert grid.resolution_m == 20.0
    assert grid.num_points > 100
    assert np.all(np.abs(grid.lats - center_lat) <= 0.005)
    assert np.all(np.abs(grid.lons - center_lon) <= 0.005)


def test_geojson_export(tmp_path):
    """Verify GeoJSON output export structure and properties."""
    candidates = [
        GeolocationCandidate(
            rank=1,
            lat=40.001,
            lon=-75.002,
            elevation_m=150.0,
            total_score=520.5,
            score_event_1=260.0,
            score_event_2=260.5,
            flight_id_1="PROP_1",
            flight_id_2="PROP_2",
            locked_delta_t=0.2,
            applied_time_shift=-999.8,
            best_delta_f_1=0.1,
            best_delta_f_2=-0.2,
        )
    ]
    geojson_file = str(tmp_path / "test_results.geojson")
    export_geojson(candidates, geojson_file)

    assert os.path.exists(geojson_file)
    with open(geojson_file, "r") as f:
        data = json.load(f)

    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1
    feat = data["features"][0]
    assert feat["geometry"]["type"] == "Point"
    assert feat["geometry"]["coordinates"] == [-75.002, 40.001]
    assert feat["properties"]["rank"] == 1
    assert feat["properties"]["total_score"] == 520.5


def test_folium_map_generation(tmp_path):
    """Verify interactive HTML map generation."""
    candidates = [
        GeolocationCandidate(
            rank=1,
            lat=40.0,
            lon=-75.0,
            elevation_m=100.0,
            total_score=500.0,
            score_event_1=250.0,
            score_event_2=250.0,
            flight_id_1="F1",
            flight_id_2="F2",
            locked_delta_t=0.0,
            applied_time_shift=-1000.0,
            best_delta_f_1=0.0,
            best_delta_f_2=0.0,
        )
    ]
    f1_df = pd.DataFrame({
        "lat": [39.99, 40.0, 40.01],
        "lon": [-75.0, -75.0, -75.0],
    })
    map_file = str(tmp_path / "test_map.html")
    generate_folium_map(candidates, flight_dfs={"F1": f1_df}, output_path=map_file)

    assert os.path.exists(map_file)
    assert os.path.getsize(map_file) > 1000


def test_plot_spectrogram_with_curves(tmp_path):
    """Verify matplotlib spectrogram overlay plot creation."""
    sr = 2000
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 120.0 * t)

    gen = SpectrogramGenerator(n_fft=1024, hop_length=256, min_freq_hz=50.0, max_freq_hz=200.0)
    spec_data = gen.generate_from_samples(y, sr)

    t_curve = np.linspace(1000.0, 1010.0, 20)
    f_curve = np.linspace(130.0, 110.0, 20)
    curve1 = DopplerCurve(
        t_obs=t_curve,
        f_obs=f_curve,
        t_emit=t_curve,
        slant_distances=np.full_like(t_curve, 1000.0),
        radial_velocities=np.zeros_like(t_curve),
        f0_baseline_hz=120.0,
    )
    curve2 = DopplerCurve(
        t_obs=t_curve,
        f_obs=f_curve,
        t_emit=t_curve,
        slant_distances=np.full_like(t_curve, 1000.0),
        radial_velocities=np.zeros_like(t_curve),
        f0_baseline_hz=120.0,
    )

    plot_file = str(tmp_path / "test_spec.png")
    plot_spectrogram_with_curves(
        spec_data=spec_data,
        curve_e1=curve1,
        curve_e2=curve2,
        applied_time_shift=-1000.0,
        output_path=plot_file,
    )

    assert os.path.exists(plot_file)
    assert os.path.getsize(plot_file) > 5000


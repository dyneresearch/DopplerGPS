"""Tests for CLI subcommands, real data fetcher, and audio inspector."""

from __future__ import annotations
import json
import os
import pathlib
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import soundfile as sf

from doppler_geo.phase1_ingestion import BoundingBox, ADSBDataLoader
from doppler_geo.real_data_fetcher import (
    fetch_opensky_live_states,
    fetch_opensky_track,
    convert_adsbx_json_to_parquet,
)
from doppler_geo.phase2_audio import inspect_audio
from doppler_geo.cli import build_parser, main


def test_convert_adsbx_json_to_parquet(tmp_path: pathlib.Path):
    json_path = tmp_path / "aircraft.json"
    parquet_path = tmp_path / "flights.parquet"

    payload = {
        "now": 1700000000.0,
        "aircraft": [
            {
                "hex": "a00001",
                "flight": "N172AB",
                "lat": 40.10,
                "lon": -75.20,
                "alt_geom": 3000, # feet
                "gs": 120.0,       # knots
                "track": 90.0,
                "seen_pos": 1.5,
            },
            {
                "hex": "a00002",
                "flight": "PA28CD",
                "lat": 38.50, # Outside bbox
                "lon": -75.20,
                "alt_geom": 2500,
                "gs": 110.0,
                "track": 180.0,
                "seen_pos": 0.5,
            },
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    bbox = BoundingBox(min_lat=40.0, max_lat=41.0, min_lon=-76.0, max_lon=-74.0)
    df = convert_adsbx_json_to_parquet(str(json_path), str(parquet_path), bbox=bbox)

    assert len(df) == 1
    assert df["flight_id"].iloc[0] == "N172AB"
    assert pytest.approx(df["lat"].iloc[0]) == 40.10
    assert pytest.approx(df["lon"].iloc[0]) == -75.20
    # 3000 feet * 0.3048 = 914.4 meters
    assert pytest.approx(df["altitude_asl_meters"].iloc[0], rel=1e-2) == 914.4
    # 120 knots * 0.514444 = 61.73 m/s
    assert pytest.approx(df["velocity_m_s"].iloc[0], rel=1e-2) == 61.73

    # Verify queryability with ADSBDataLoader
    loader = ADSBDataLoader()
    loaded = loader.load_flight_data(str(parquet_path), bbox=bbox)
    assert len(loaded) == 1


def test_fetch_opensky_live_states_mock(tmp_path: pathlib.Path):
    mock_response = json.dumps({
        "time": 1700000000,
        "states": [
            ["a12345", "N172SP  ", "United States", 1700000000, 1700000000, -75.1, 40.05, 1000, False, 55.0, 90.0, 0.0, None, 1050.0, "1200", False, 0]
        ]
    }).encode("utf-8")

    parquet_path = tmp_path / "opensky_states.parquet"
    bbox = BoundingBox(min_lat=39.0, max_lat=41.0, min_lon=-76.0, max_lon=-74.0)

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_cm = MagicMock()
        mock_cm.read.return_value = mock_response
        mock_urlopen.return_value.__enter__.return_value = mock_cm

        df = fetch_opensky_live_states(bbox=bbox, output_parquet_path=str(parquet_path))

    assert len(df) == 1
    assert df["flight_id"].iloc[0] == "N172SP"
    assert pytest.approx(df["altitude_asl_meters"].iloc[0]) == 1050.0
    assert os.path.isfile(parquet_path)


def test_fetch_opensky_track_mock(tmp_path: pathlib.Path):
    mock_response = json.dumps({
        "icao24": "a12345",
        "callsign": "N172SP",
        "startTime": 1700000000,
        "endTime": 1700000100,
        "path": [
            [1700000000, 40.00, -75.00, 1000.0, 90.0, False],
            [1700000010, 40.01, -74.99, 1020.0, 90.0, False],
            [1700000020, 40.02, -74.98, 1040.0, 90.0, False],
        ]
    }).encode("utf-8")

    parquet_path = tmp_path / "opensky_track.parquet"
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_cm = MagicMock()
        mock_cm.read.return_value = mock_response
        mock_urlopen.return_value.__enter__.return_value = mock_cm

        df = fetch_opensky_track(
            icao24="a12345",
            target_time_utc=1700000010,
            output_parquet_path=str(parquet_path),
        )

    assert len(df) == 3
    assert df["flight_id"].iloc[0] == "N172SP"
    assert df["velocity_m_s"].iloc[1] > 0.0
    assert os.path.isfile(parquet_path)


def test_inspect_audio_peak_detection(tmp_path: pathlib.Path):
    sr = 4000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 120 Hz tone with noise
    signal = 0.5 * np.sin(2 * np.pi * 120.0 * t) + 0.02 * np.random.randn(len(t))
    wav_path = str(tmp_path / "test_tone.wav")
    sf.write(wav_path, signal, sr)

    plot_path = str(tmp_path / "diag.png")
    diag = inspect_audio(
        audio_path=wav_path,
        output_image_path=plot_path,
        min_freq_hz=80.0,
        max_freq_hz=160.0,
        n_fft=2048,
        hop_length=256,
    )

    assert diag.duration_s == pytest.approx(duration, abs=0.1)
    assert diag.sample_rate == sr
    assert os.path.isfile(plot_path)

    # Check that 120 Hz is among detected peaks
    detected_freqs = [f for f, _ in diag.detected_peaks_hz]
    assert any(abs(f - 120.0) < 3.0 for f in detected_freqs)


def test_cli_parser_subcommands():
    parser = build_parser()
    
    # demo
    demo_args = parser.parse_args(["demo", "--output-dir", "out"])
    assert demo_args.command == "demo"
    assert demo_args.output_dir == "out"

    # fetch-adsb
    fetch_args = parser.parse_args([
        "fetch-adsb",
        "--source", "opensky-live",
        "--bbox", "40.0", "41.0", "-75.0", "-74.0",
        "--output", "test.parquet"
    ])
    assert fetch_args.command == "fetch-adsb"
    assert fetch_args.source == "opensky-live"
    assert fetch_args.bbox == [40.0, 41.0, -75.0, -74.0]

    # inspect-audio
    inspect_args = parser.parse_args([
        "inspect-audio",
        "--audio", "audio.wav",
        "--fmin", "60.0",
        "--fmax", "300.0"
    ])
    assert inspect_args.command == "inspect-audio"
    assert inspect_args.fmin == 60.0
    assert inspect_args.fmax == 300.0


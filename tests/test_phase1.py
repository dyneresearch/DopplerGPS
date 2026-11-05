"""Unit tests for Phase 1: Data Ingestion & Preprocessing Engine."""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from doppler_geo.phase1_ingestion import (
    BoundingBox,
    TimeWindow,
    ADSBDataLoader,
    DEMHandler,
    SyntheticDEMHandler,
    CANONICAL_ADSB_COLUMNS,
)


@pytest.fixture
def sample_adsb_parquet(tmp_path):
    """Create a temporary parquet file with mixed jet and propeller aircraft."""
    records = [
        # Propeller 1: Cessna 172
        {"flight_id": "N172SP", "type_code": "C172", "timestamp_utc": 1000.0, "lat": 40.0, "lon": -75.0, "altitude_asl_meters": 1500.0, "velocity_m_s": 60.0, "heading": 90.0},
        {"flight_id": "N172SP", "type_code": "C172", "timestamp_utc": 1010.0, "lat": 40.005, "lon": -74.99, "altitude_asl_meters": 1510.0, "velocity_m_s": 61.0, "heading": 90.0},
        # Propeller 2: Piper PA-28
        {"flight_id": "N28PA", "type_code": "PA28", "timestamp_utc": 1920.0, "lat": 40.01, "lon": -75.01, "altitude_asl_meters": 1200.0, "velocity_m_s": 55.0, "heading": 180.0},
        {"flight_id": "N28PA", "type_code": "PA28", "timestamp_utc": 1930.0, "lat": 40.00, "lon": -75.01, "altitude_asl_meters": 1200.0, "velocity_m_s": 55.0, "heading": 180.0},
        # Jet 1: Boeing 737-800
        {"flight_id": "AAL123", "type_code": "B738", "timestamp_utc": 1500.0, "lat": 40.02, "lon": -75.02, "altitude_asl_meters": 10000.0, "velocity_m_s": 230.0, "heading": 45.0},
        # Jet 2: Airbus A320
        {"flight_id": "DAL456", "type_code": "A320", "timestamp_utc": 1600.0, "lat": 40.03, "lon": -75.03, "altitude_asl_meters": 11000.0, "velocity_m_s": 240.0, "heading": 45.0},
        # Propeller out of bounds
        {"flight_id": "N999XX", "type_code": "C182", "timestamp_utc": 1000.0, "lat": 45.0, "lon": -80.0, "altitude_asl_meters": 1000.0, "velocity_m_s": 70.0, "heading": 0.0},
    ]
    df = pd.DataFrame(records)
    parquet_file = str(tmp_path / "sample_flights.parquet")
    df.to_parquet(parquet_file, index=False)
    return parquet_file


@pytest.fixture
def sample_geotiff_dem(tmp_path):
    """Create a temporary 10x10 GeoTIFF raster file for DEM testing."""
    dem_file = str(tmp_path / "test_dem.tif")
    transform = from_origin(-75.1, 40.1, 0.01, 0.01)
    data = np.arange(100, dtype=np.float32).reshape(10, 10) + 120.0  # 120m to 219m

    with rasterio.open(
        dem_file,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    return dem_file


def test_adsb_loader_filters_jets(sample_adsb_parquet):
    """Verify that ADSBDataLoader filters out jets and retains propeller planes."""
    loader = ADSBDataLoader()
    df = loader.load_flight_data(sample_adsb_parquet, only_propellers=True)

    flight_ids = df["flight_id"].unique()
    assert "N172SP" in flight_ids
    assert "N28PA" in flight_ids
    assert "AAL123" not in flight_ids
    assert "DAL456" not in flight_ids
    assert list(df.columns) == CANONICAL_ADSB_COLUMNS


def test_adsb_loader_bounding_box_and_time(sample_adsb_parquet):
    """Verify that spatial bounding box and time window filtering work properly."""
    loader = ADSBDataLoader()
    bbox = BoundingBox(min_lat=39.9, max_lat=40.05, min_lon=-75.1, max_lon=-74.9)
    time_win = TimeWindow(start_utc=950.0, end_utc=1050.0)

    df = loader.load_flight_data(
        sample_adsb_parquet,
        bbox=bbox,
        time_window=time_win,
        only_propellers=True,
    )

    flight_ids = df["flight_id"].unique()
    assert list(flight_ids) == ["N172SP"]
    assert len(df) == 2


def test_dem_handler_with_rasterio(sample_geotiff_dem):
    """Test DEMHandler with a real rasterio GeoTIFF file and LRU caching."""
    with DEMHandler(sample_geotiff_dem) as dem:
        elev = dem.get_elevation(lat=40.05, lon=-75.05)
        assert 120.0 <= elev <= 220.0

        # Vectorized lookup
        lats = [40.05, 40.06]
        lons = [-75.05, -75.04]
        elevs = dem.get_elevations(lats, lons)
        assert len(elevs) == 2
        assert np.all(elevs >= 120.0)


def test_synthetic_dem_handler():
    """Test SyntheticDEMHandler fallback behavior."""
    synth_dem = SyntheticDEMHandler(base_elevation_m=250.0, slope_lat=0.01, ref_lat=40.0)
    elev1 = synth_dem.get_elevation(lat=40.0, lon=-75.0)
    assert elev1 == 250.0

    elev2 = synth_dem.get_elevation(lat=40.01, lon=-75.0)
    assert elev2 > 250.0



def test_adsb_loader_opensky_and_adsbx_schemas(tmp_path):
    """Verify ADSBDataLoader auto-detects OpenSky and ADS-B Exchange column schemas."""
    # OpenSky Network schema: time, icao24, lat, lon, geoaltitude (meters), velocity (m/s), heading
    opensky_df = pd.DataFrame([
        {"time": 1000.0, "icao24": "a12345", "lat": 40.0, "lon": -75.0, "geoaltitude": 2500.0, "velocity": 90.0, "heading": 180.0},
        {"time": 1010.0, "icao24": "a12345", "lat": 40.01, "lon": -75.0, "geoaltitude": 2500.0, "velocity": 90.0, "heading": 180.0},
    ])
    opensky_parquet = str(tmp_path / "opensky.parquet")
    opensky_df.to_parquet(opensky_parquet, index=False)

    loader = ADSBDataLoader()
    df_os = loader.load_flight_data(opensky_parquet, only_propellers=False)
    assert list(df_os.columns) == CANONICAL_ADSB_COLUMNS
    assert df_os["flight_id"].iloc[0] == "a12345"
    assert df_os["altitude_asl_meters"].iloc[0] == 2500.0
    assert df_os["velocity_m_s"].iloc[0] == 90.0

    # ADS-B Exchange schema: seen_pos, hex, t, lat, lon, alt_geom (feet), gs (knots), track
    adsbx_df = pd.DataFrame([
        {"seen_pos": 2000.0, "hex": "b67890", "t": "C172", "lat": 40.0, "lon": -75.0, "alt_geom": 3280.84, "gs": 100.0, "track": 90.0},
        {"seen_pos": 2010.0, "hex": "b67890", "t": "C172", "lat": 40.0, "lon": -74.99, "alt_geom": 3280.84, "gs": 100.0, "track": 90.0},
    ])
    adsbx_parquet = str(tmp_path / "adsbx.parquet")
    adsbx_df.to_parquet(adsbx_parquet, index=False)

    df_bx = loader.load_flight_data(adsbx_parquet, only_propellers=True)
    assert list(df_bx.columns) == CANONICAL_ADSB_COLUMNS
    assert df_bx["flight_id"].iloc[0] == "b67890"
    # 3280.84 feet * 0.3048 ~ 1000 meters
    assert np.isclose(df_bx["altitude_asl_meters"].iloc[0], 1000.0, atol=1.0)
    # 100 knots * 0.514444 ~ 51.44 m/s
    assert np.isclose(df_bx["velocity_m_s"].iloc[0], 51.4444, atol=0.1)

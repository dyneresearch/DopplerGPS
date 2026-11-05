"""Real Data Fetcher & Ingestion Utilities for ADS-B Telemetry.

Provides production utilities to:
1. Fetch live or recent state vectors from OpenSky Network's public REST API.
2. Fetch full historical flight track trajectories from OpenSky Network's Track API.
3. Ingest and convert raw readsb / ADS-B Exchange (ADSBx) JSON telemetry into
   DuckDB-compatible canonical Parquet files.
"""

from __future__ import annotations
import glob
import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from doppler_geo.phase1_ingestion.schemas import BoundingBox, CANONICAL_ADSB_COLUMNS

logger = logging.getLogger(__name__)


def fetch_opensky_live_states(
    bbox: BoundingBox,
    output_parquet_path: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    timeout_s: float = 20.0,
) -> pd.DataFrame:
    """Fetch live or recent aircraft state vectors from OpenSky Network's public REST API.

    Args:
        bbox: BoundingBox (min_lat, max_lat, min_lon, max_lon).
        output_parquet_path: Optional path to save as .parquet.
        username: Optional OpenSky account username.
        password: Optional OpenSky account password.
        timeout_s: Network timeout in seconds.

    Returns:
        DataFrame adhering to CANONICAL_ADSB_COLUMNS.
    """
    url = (
        f"https://opensky-network.org/api/states/all?"
        f"lamin={bbox.min_lat}&lamax={bbox.max_lat}&"
        f"lomin={bbox.min_lon}&lomax={bbox.max_lon}"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "DopplerGPS/1.0 (Dyne Research; connect@dyneresearch.com)"})
    if username and password:
        import base64
        auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {auth}")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        logger.error(f"OpenSky API HTTP Error {e.code}: {e.reason}")
        raise RuntimeError(f"OpenSky API returned HTTP {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        logger.error(f"Failed to reach OpenSky API: {e.reason}")
        raise RuntimeError(f"Network error querying OpenSky: {e.reason}") from e

    time_query = float(data.get("time", time.time()))
    raw_states = data.get("states", []) or []

    rows = []
    for s in raw_states:
        # OpenSky states vector format:
        # [0]=icao24, [1]=callsign, [2]=country, [3]=time_pos, [4]=last_contact,
        # [5]=lon, [6]=lat, [7]=baro_alt, [8]=on_ground, [9]=velocity, [10]=heading,
        # [11]=vert_rate, [12]=sensors, [13]=geo_altitude, [14]=squawk, [15]=spi, [16]=source
        if len(s) < 11:
            continue
        icao24 = s[0]
        callsign = (s[1] or "").strip()
        time_pos = s[3] or time_query
        lon = s[5]
        lat = s[6]
        geo_alt = s[13] if len(s) > 13 and s[13] is not None else s[7]
        vel = s[9]
        heading = s[10]

        if lat is None or lon is None or geo_alt is None or vel is None:
            continue

        flight_id = callsign if callsign else str(icao24)

        rows.append({
            "flight_id": flight_id,
            "timestamp_utc": float(time_pos),
            "lat": float(lat),
            "lon": float(lon),
            "altitude_asl_meters": float(geo_alt),
            "velocity_m_s": float(vel),
            "heading": float(heading) if heading is not None else 0.0,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=CANONICAL_ADSB_COLUMNS)

    if output_parquet_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
        df.to_parquet(output_parquet_path, index=False)

    return df


def fetch_opensky_track(
    icao24: str,
    target_time_utc: int,
    output_parquet_path: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    timeout_s: float = 30.0,
) -> pd.DataFrame:
    """Fetch complete historical 3D trajectory waypoints for an aircraft from OpenSky.

    Args:
        icao24: 24-bit ICAO transponder address (hex string, e.g. 'a1b2c3').
        target_time_utc: Epoch timestamp within the flight (0 for current).
        output_parquet_path: Optional destination .parquet file.
        username: Optional OpenSky account username.
        password: Optional OpenSky account password.
        timeout_s: Network timeout in seconds.

    Returns:
        DataFrame of flight waypoints in canonical schema.
    """
    url = f"https://opensky-network.org/api/tracks/all?icao24={icao24.lower()}&time={int(target_time_utc)}"
    req = urllib.request.Request(url, headers={"User-Agent": "DopplerGPS/1.0 (Dyne Research)"})
    if username and password:
        import base64
        auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {auth}")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.error(f"Failed to fetch track for {icao24}: {e}")
        raise RuntimeError(f"OpenSky Track API error: {e}") from e

    callsign = (data.get("callsign") or icao24).strip()
    path_points = data.get("path", []) or []

    rows = []
    # OpenSky track waypoint structure:
    # [0]=time, [1]=latitude, [2]=longitude, [3]=baro_altitude, [4]=true_track, [5]=on_ground
    prev_time = None
    prev_lat = None
    prev_lon = None

    for pt in path_points:
        if len(pt) < 4:
            continue
        t_utc = float(pt[0])
        lat = pt[1]
        lon = pt[2]
        alt = pt[3]
        track = pt[4] if len(pt) > 4 and pt[4] is not None else 0.0

        if lat is None or lon is None or alt is None:
            continue

        # Approximate velocity if not explicitly provided
        velocity_m_s = 0.0
        if prev_time is not None and t_utc > prev_time:
            dt = t_utc - prev_time
            from doppler_geo.coordinates import horizontal_haversine_distance
            dist = horizontal_haversine_distance(prev_lat, prev_lon, lat, lon)
            velocity_m_s = dist / dt if dt > 0 else 0.0

        prev_time = t_utc
        prev_lat = lat
        prev_lon = lon

        rows.append({
            "flight_id": callsign,
            "timestamp_utc": t_utc,
            "lat": float(lat),
            "lon": float(lon),
            "altitude_asl_meters": float(alt),
            "velocity_m_s": float(velocity_m_s),
            "heading": float(track),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=CANONICAL_ADSB_COLUMNS)
    else:
        # Forward-fill initial velocity if zero
        if df["velocity_m_s"].iloc[0] == 0.0 and len(df) > 1:
            df.loc[0, "velocity_m_s"] = df["velocity_m_s"].iloc[1]

    if output_parquet_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
        df.to_parquet(output_parquet_path, index=False)

    return df


def convert_adsbx_json_to_parquet(
    json_path_or_glob: str,
    output_parquet_path: str,
    bbox: Optional[BoundingBox] = None,
) -> pd.DataFrame:
    """Convert raw readsb / ADS-B Exchange (ADSBx) JSON telemetry files into Parquet.

    Compatible with standard aircraft.json or tar1090 history JSON files.

    Args:
        json_path_or_glob: Filepath or glob pattern matching JSON files.
        output_parquet_path: Path where canonical Parquet file will be saved.
        bbox: Optional BoundingBox filter to discard out-of-region fixes.

    Returns:
        Canonical DataFrame.
    """
    matched_files = glob.glob(json_path_or_glob)
    if not matched_files and os.path.isfile(json_path_or_glob):
        matched_files = [json_path_or_glob]

    if not matched_files:
        raise FileNotFoundError(f"No files matched pattern: {json_path_or_glob}")

    rows = []
    for filepath in matched_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping unparseable JSON file {filepath}: {e}")
            continue

        file_now = content.get("now", time.time())
        aircraft_list = content.get("aircraft", [])

        for ac in aircraft_list:
            lat = ac.get("lat")
            lon = ac.get("lon")
            if lat is None or lon is None:
                continue

            if bbox is not None:
                if not (bbox.min_lat <= lat <= bbox.max_lat and bbox.min_lon <= lon <= bbox.max_lon):
                    continue

            # Altitude: prefer geometric altitude (alt_geom), fallback to barometric (alt_baro)
            # In readsb / ADSBx, altitude is typically in feet
            alt_ft = ac.get("alt_geom") or ac.get("alt_baro")
            if alt_ft is None:
                continue
            if isinstance(alt_ft, str):
                try:
                    alt_ft = float(alt_ft)
                except ValueError:
                    continue
            alt_m = alt_ft * 0.3048

            # Ground speed in knots -> m/s
            gs_kt = ac.get("gs") or 0.0
            vel_m_s = float(gs_kt) * 0.514444

            heading = float(ac.get("track") or ac.get("true_heading") or 0.0)
            seen_pos = ac.get("seen_pos") or 0.0
            timestamp_utc = float(file_now) - float(seen_pos)

            flight_id = (ac.get("flight") or ac.get("hex") or "UNKNOWN").strip()

            rows.append({
                "flight_id": flight_id,
                "timestamp_utc": timestamp_utc,
                "lat": float(lat),
                "lon": float(lon),
                "altitude_asl_meters": float(alt_m),
                "velocity_m_s": float(vel_m_s),
                "heading": heading,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=CANONICAL_ADSB_COLUMNS)
    else:
        df = df.sort_values(by=["flight_id", "timestamp_utc"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
    df.to_parquet(output_parquet_path, index=False)
    logger.info(f"Successfully converted {len(df)} ADS-B fixes to {output_parquet_path}")

    return df

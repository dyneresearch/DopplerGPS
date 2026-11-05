"""Real Data Fetcher & Converter.

Utility to fetch real flight state vectors from OpenSky Network or convert
raw readsb/ADSBx JSON archives into DuckDB-compatible Parquet files.
"""

from __future__ import annotations
import os
import json
import time
import urllib.request
from typing import Optional, List, Dict, Any
import pandas as pd
import duckdb

from doppler_geo.phase1_ingestion.schemas import BoundingBox, CANONICAL_ADSB_COLUMNS


def fetch_opensky_live_states(
    bbox: BoundingBox,
    output_parquet_path: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch live or recent aircraft state vectors from OpenSky Network's public REST API.

    Args:
        bbox: BoundingBox (min_lat, max_lat, min_lon, max_lon).
        output_parquet_path: Optional path to save as .parquet.
        username: Optional OpenSky username.
        password: Optional OpenSky password.

    Returns:
        DataFrame in canonical schema.
    """
    url = (
        f"https://opensky-network.org/api/states/all?"
        f"lamin={bbox.min_lat}&lamax={bbox.max_lat}&"
        f"lomin={bbox.min_lon}&lomax={bbox.max_lon}"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "DopplerGPS/1.0"})
    if username and password:
        import base64
        auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {auth}")

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())

    time_query = float(data.get("time", time.time()))
    raw_states = data.get("states", []) or []

    rows = []
    for s in raw_states:
        # OpenSky states vector layout:
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
        os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
        df.to_parquet(output_parquet_path, index=False)

    return df


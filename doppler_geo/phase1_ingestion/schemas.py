"""Data schemas and type definitions for Phase 1 Data Ingestion."""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd


@dataclass(frozen=True)
class BoundingBox:
    """Geographic search bounding box in WGS84 degrees."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon


@dataclass(frozen=True)
class TimeWindow:
    """Temporal window in UTC epoch seconds."""
    start_utc: float
    end_utc: float

    def contains(self, timestamp: float) -> bool:
        return self.start_utc <= timestamp <= self.end_utc


# Required canonical columns for the ADSB output dataframe
CANONICAL_ADSB_COLUMNS = [
    "flight_id",
    "timestamp_utc",
    "lat",
    "lon",
    "altitude_asl_meters",
    "velocity_m_s",
    "heading",
]


def validate_adsb_dataframe(df: pd.DataFrame) -> None:
    """Validate that dataframe strictly adheres to Phase 1 output schema."""
    missing = [c for c in CANONICAL_ADSB_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ADSB DataFrame is missing required columns: {missing}")
    if df.empty:
        return
    # Check nulls in essential columns
    for col in ["flight_id", "timestamp_utc", "lat", "lon"]:
        if df[col].isnull().any():
            raise ValueError(f"Column '{col}' contains NaN/null values.")


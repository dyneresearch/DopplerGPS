"""ADSB Data Loader (The Parquet Processor).

Uses DuckDB to lazily query large .parquet flight logs with predicate pushdown.
Supports standard canonical schema, OpenSky Network schema, and ADS-B Exchange schemas.
Filters out known jet engines using aircraft type codes to strictly isolate
propeller aircraft.
"""

from __future__ import annotations
import os
from typing import Optional, List, Union
import duckdb
import pandas as pd
from doppler_geo.phase1_ingestion.schemas import (
    BoundingBox,
    TimeWindow,
    CANONICAL_ADSB_COLUMNS,
    validate_adsb_dataframe,
)

# Common known jet aircraft ICAO type code prefixes/patterns to filter out
KNOWN_JET_PREFIXES = (
    "A3",   # Airbus A318-A380
    "B7",   # Boeing 707-787
    "MD",   # MD-80, MD-90, MD-11
    "DC",   # DC-8, DC-9, DC-10
    "CRJ",  # Bombardier CRJ
    "ERJ",  # Embraer ERJ
    "E17",  # Embraer 170/175
    "E19",  # Embraer 190/195
    "E29",  # Embraer 190-E2
    "C5",   # Cessna Citation
    "C6",   # Cessna Citation
    "C7",   # Cessna Citation
    "GLF",  # Gulfstream
    "GLEX", # Bombardier Global Express
    "CL3",  # Challenger 300
    "CL6",  # Challenger 600
    "F90",  # Falcon 900
    "FA7",  # Falcon 7X
    "FA8",  # Falcon 8X
    "HA42", # HondaJet
    "LJ",   # Learjet
    "B46",  # BAe 146
    "RJ",   # Avro RJ
)


class ADSBDataLoader:
    """Lazy DuckDB loader for massive historical ADSB parquet flight files."""

    def __init__(self, db_connection: Optional[duckdb.DuckDBPyConnection] = None):
        self.con = db_connection or duckdb.connect(":memory:")

    def _inspect_parquet_columns(self, path_str: str) -> List[str]:
        """Inspect column names present in the parquet file(s)."""
        try:
            desc = self.con.execute(f"DESCRIBE SELECT * FROM read_parquet({path_str}) LIMIT 0").fetchall()
            return [str(row[0]).lower() for row in desc]
        except Exception:
            return []

    def load_flight_data(
        self,
        parquet_path: Union[str, List[str]],
        bbox: Optional[BoundingBox] = None,
        time_window: Optional[TimeWindow] = None,
        only_propellers: bool = True,
        alt_in_feet: Optional[bool] = None,
        speed_in_knots: Optional[bool] = None,
        custom_filter_sql: Optional[str] = None,
    ) -> pd.DataFrame:
        """Lazily load, filter, and extract canonical flight trajectory records.

        Supports Canonical, OpenSky Network, and ADS-B Exchange schemas automatically.

        Returns:
            A pandas DataFrame adhering strictly to CANONICAL_ADSB_COLUMNS:
            [flight_id, timestamp_utc, lat, lon, altitude_asl_meters, velocity_m_s, heading]
        """
        if isinstance(parquet_path, list):
            path_str = "[" + ", ".join(f"'{p}'" for p in parquet_path) + "]"
            probe_path = f"'{parquet_path[0]}'"
        else:
            path_str = f"'{parquet_path}'"
            probe_path = path_str

        cols = self._inspect_parquet_columns(probe_path)

        # 1. Resolve Time Column
        if "timestamp_utc" in cols:
            raw_time_col = "timestamp_utc"
        elif "time" in cols:
            raw_time_col = "time"
        elif "timestamp" in cols:
            raw_time_col = "timestamp"
        elif "seen_pos" in cols:
            raw_time_col = "seen_pos"
        else:
            raw_time_col = "timestamp_utc"

        # 2. Resolve Flight ID Column
        if "flight_id" in cols:
            flight_id_col = "flight_id"
        elif "icao24" in cols:
            flight_id_col = "icao24"
        elif "hex" in cols:
            flight_id_col = "hex"
        elif "callsign" in cols:
            flight_id_col = "callsign"
        else:
            flight_id_col = "flight_id"

        # 3. Resolve Altitude Column and Unit
        if "altitude_asl_meters" in cols:
            alt_source = "altitude_asl_meters"
            is_feet = False if alt_in_feet is None else alt_in_feet
        elif "geoaltitude" in cols:
            alt_source = "geoaltitude"
            is_feet = False if alt_in_feet is None else alt_in_feet
        elif "baroaltitude" in cols:
            alt_source = "baroaltitude"
            is_feet = False if alt_in_feet is None else alt_in_feet
        elif "alt_geom" in cols:
            alt_source = "alt_geom"
            is_feet = True if alt_in_feet is None else alt_in_feet
        elif "alt_baro" in cols:
            alt_source = "alt_baro"
            is_feet = True if alt_in_feet is None else alt_in_feet
        elif "altitude" in cols:
            alt_source = "altitude"
            is_feet = False if alt_in_feet is None else alt_in_feet
        else:
            alt_source = "altitude_asl_meters"
            is_feet = False if alt_in_feet is None else alt_in_feet

        alt_expr = f"CAST({alt_source} AS DOUBLE) * 0.3048" if is_feet else f"CAST({alt_source} AS DOUBLE)"

        # 4. Resolve Velocity Column and Unit
        if "velocity_m_s" in cols:
            vel_source = "velocity_m_s"
            is_knots = False if speed_in_knots is None else speed_in_knots
        elif "velocity" in cols:
            vel_source = "velocity"
            is_knots = False if speed_in_knots is None else speed_in_knots
        elif "gs" in cols:
            vel_source = "gs"
            is_knots = True if speed_in_knots is None else speed_in_knots
        elif "groundspeed" in cols:
            vel_source = "groundspeed"
            is_knots = True if speed_in_knots is None else speed_in_knots
        else:
            vel_source = "velocity_m_s"
            is_knots = False if speed_in_knots is None else speed_in_knots

        vel_expr = f"CAST({vel_source} AS DOUBLE) * 0.514444" if is_knots else f"CAST({vel_source} AS DOUBLE)"

        # 5. Resolve Heading Column
        if "heading" in cols:
            heading_expr = "CAST(heading AS DOUBLE)"
        elif "track" in cols:
            heading_expr = "CAST(track AS DOUBLE)"
        else:
            heading_expr = "0.0"

        # 6. Resolve Aircraft Type Column for Jet Filtering
        if "type_code" in cols:
            type_col = "type_code"
        elif "t" in cols:
            type_col = "t"
        elif "type" in cols:
            type_col = "type"
        else:
            type_col = None

        where_clauses = ["lat IS NOT NULL", "lon IS NOT NULL", f"{raw_time_col} IS NOT NULL"]

        if bbox is not None:
            where_clauses.append(
                f"lat >= {bbox.min_lat} AND lat <= {bbox.max_lat} "
                f"AND lon >= {bbox.min_lon} AND lon <= {bbox.max_lon}"
            )

        if time_window is not None:
            where_clauses.append(
                f"{raw_time_col} >= {time_window.start_utc} AND {raw_time_col} <= {time_window.end_utc}"
            )

        if only_propellers and type_col is not None:
            jet_conditions = " AND ".join(
                [f"{type_col} NOT LIKE '{prefix}%'" for prefix in KNOWN_JET_PREFIXES]
            )
            where_clauses.append(f"({jet_conditions})")

        if custom_filter_sql:
            where_clauses.append(f"({custom_filter_sql})")

        where_stmt = " AND ".join(where_clauses)

        query = f"""
            SELECT
                CAST({flight_id_col} AS VARCHAR) AS flight_id,
                CAST({raw_time_col} AS DOUBLE) AS timestamp_utc,
                CAST(lat AS DOUBLE) AS lat,
                CAST(lon AS DOUBLE) AS lon,
                {alt_expr} AS altitude_asl_meters,
                {vel_expr} AS velocity_m_s,
                {heading_expr} AS heading
            FROM read_parquet({path_str})
            WHERE {where_stmt}
            ORDER BY flight_id, timestamp_utc ASC
        """

        df = self.con.execute(query).df()
        validate_adsb_dataframe(df)
        return df

    def get_unique_flights(self, df: pd.DataFrame) -> List[str]:
        """Return list of distinct flight IDs present in the dataset."""
        return df["flight_id"].unique().tolist()

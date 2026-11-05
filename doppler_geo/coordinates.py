"""Coordinate conversion and local Cartesian (ENU) geometry module.

Strictly manages Coordinate Reference Systems (CRS). Converts between
WGS84 (Lat, Lon, Altitude ASL in meters) and local East-North-Up (ENU)
Cartesian coordinates in meters to ensure high-accuracy physics and metric distance math.
"""

from __future__ import annotations
from typing import Tuple, Union
import numpy as np
import pyproj

# WGS84 Ellipsoid constants
WGS84_A = 6378137.0  # Semi-major axis in meters
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # Semi-minor axis in meters
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2  # First eccentricity squared


def geodetic_to_ecef(
    lat_deg: Union[float, np.ndarray],
    lon_deg: Union[float, np.ndarray],
    alt_m: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic coordinates (WGS84 lat, lon, alt ASL) to ECEF (X, Y, Z in meters).
    
    Fully vectorized for NumPy arrays or scalars.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    # Prime vertical radius of curvature
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    
    return x, y, z


def ecef_to_enu(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF coordinates to local East, North, Up (ENU) in meters relative to a reference origin."""
    if "ref_alt_m" in kwargs:
        ref_alt = kwargs["ref_alt_m"]
    if "ref_lat_deg" in kwargs:
        ref_lat = kwargs["ref_lat_deg"]
    if "ref_lon_deg" in kwargs:
        ref_lon = kwargs["ref_lon_deg"]

    ref_x, ref_y, ref_z = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)
    
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z
    
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)
    
    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)
    
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return east, north, up


def geodetic_to_enu(
    lat_deg: Union[float, np.ndarray],
    lon_deg: Union[float, np.ndarray],
    alt_m: Union[float, np.ndarray],
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directly convert geodetic coordinates (lat, lon, alt) to local ENU (East, North, Up) in meters."""
    if "ref_alt_m" in kwargs:
        ref_alt = kwargs["ref_alt_m"]
    if "ref_lat_deg" in kwargs:
        ref_lat = kwargs["ref_lat_deg"]
    if "ref_lon_deg" in kwargs:
        ref_lon = kwargs["ref_lon_deg"]

    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt)


def compute_speed_of_sound(temperature_c: float = 20.0) -> float:
    """Compute speed of sound in dry air in m/s as a function of temperature in Celsius.
    
    Formula: c = 331.3 * sqrt(1 + T / 273.15)
    At 20C, c ~ 343.2 m/s. At 0C, c ~ 331.3 m/s. At 35C, c ~ 351.9 m/s.
    """
    return float(331.3 * np.sqrt(max(1.0 + (temperature_c / 273.15), 0.1)))


def calculate_slant_distance(
    p1_enu: np.ndarray,
    p2_enu: np.ndarray,
) -> np.ndarray:
    """Calculate 3D Euclidean distance (slant distance in meters) between ENU points.
    
    p1_enu, p2_enu can be shapes (3,) or (N, 3).
    """
    diff = p1_enu - p2_enu
    return np.linalg.norm(diff, axis=-1)


def get_utm_epsg(lon: float, lat: float) -> int:
    """Calculate UTM EPSG code for a given WGS84 coordinate."""
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def horizontal_haversine_distance(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> np.ndarray:
    """Fast vectorized great-circle distance in meters using Haversine formula."""
    r = 6371000.0  # Earth mean radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


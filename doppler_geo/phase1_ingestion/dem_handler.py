"""Digital Elevation Model (DEM) Terrain Handler.

Wraps rasterio to accept (lat, lon) coordinates and return ground elevation
above sea level (ASL) in meters. Includes LRU caching to eliminate I/O bottlenecks.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Union, Optional, Tuple
import numpy as np
import rasterio
from rasterio.sample import sample_gen


class DEMHandler:
    """Terrain model handler using rasterio GeoTIFF datasets with LRU caching."""

    def __init__(self, dem_path: Optional[str] = None, default_elevation_m: float = 0.0):
        self.dem_path = dem_path
        self.default_elevation_m = default_elevation_m
        self._dataset = None
        if dem_path is not None:
            self._dataset = rasterio.open(dem_path)

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def __enter__(self) -> DEMHandler:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @lru_cache(maxsize=262144)
    def get_elevation_cached(self, lat_rounded: float, lon_rounded: float) -> float:
        """Cached lookup for ground elevation at rounded (lat, lon) in meters."""
        if self._dataset is None:
            return self.default_elevation_m

        coords = [(lon_rounded, lat_rounded)]  # rasterio expects (x, y) = (lon, lat)
        try:
            sample = list(self._dataset.sample(coords))
            val = float(sample[0][0])
            # Check for rasterio nodata
            if self._dataset.nodata is not None and np.isclose(val, self._dataset.nodata):
                return self.default_elevation_m
            return val
        except Exception:
            return self.default_elevation_m

    def get_elevation(self, lat: float, lon: float, precision: int = 5) -> float:
        """Get elevation in meters for a single (lat, lon) coordinate."""
        lat_r = round(float(lat), precision)
        lon_r = round(float(lon), precision)
        return self.get_elevation_cached(lat_r, lon_r)

    def get_elevations(
        self,
        lats: Union[np.ndarray, list],
        lons: Union[np.ndarray, list],
        precision: int = 5,
    ) -> np.ndarray:
        """Vectorized elevation lookup for arrays of latitudes and longitudes."""
        lats_arr = np.asarray(lats, dtype=np.float64)
        lons_arr = np.asarray(lons, dtype=np.float64)

        if self._dataset is None:
            return np.full_like(lats_arr, self.default_elevation_m, dtype=np.float64)

        # Use cache for unique rounded points to optimize batch lookups
        lats_r = np.round(lats_arr, precision)
        lons_r = np.round(lons_arr, precision)

        # Build coordinate pairs
        coords = np.column_stack((lons_r.ravel(), lats_r.ravel()))
        samples = np.fromiter(
            (val[0] for val in self._dataset.sample(coords)),
            dtype=np.float64,
            count=len(coords),
        )

        if self._dataset.nodata is not None:
            samples[np.isclose(samples, self._dataset.nodata)] = self.default_elevation_m

        return samples.reshape(lats_arr.shape)


class SyntheticDEMHandler(DEMHandler):
    """Synthetic terrain elevation generator for testing and simulations without raster files."""

    def __init__(
        self,
        base_elevation_m: float = 150.0,
        slope_lat: float = 0.0,
        slope_lon: float = 0.0,
        ref_lat: float = 0.0,
        ref_lon: float = 0.0,
    ):
        super().__init__(dem_path=None, default_elevation_m=base_elevation_m)
        self.base_elevation_m = base_elevation_m
        self.slope_lat = slope_lat
        self.slope_lon = slope_lon
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon

    def get_elevation(self, lat: float, lon: float, precision: int = 5) -> float:
        return float(
            self.base_elevation_m
            + self.slope_lat * (lat - self.ref_lat) * 111000.0
            + self.slope_lon * (lon - self.ref_lon) * 111000.0
        )

    def get_elevations(
        self,
        lats: Union[np.ndarray, list],
        lons: Union[np.ndarray, list],
        precision: int = 5,
    ) -> np.ndarray:
        lats_arr = np.asarray(lats, dtype=np.float64)
        lons_arr = np.asarray(lons, dtype=np.float64)
        return (
            self.base_elevation_m
            + self.slope_lat * (lats_arr - self.ref_lat) * 111000.0
            + self.slope_lon * (lons_arr - self.ref_lon) * 111000.0
        )


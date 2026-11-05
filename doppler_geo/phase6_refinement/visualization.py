"""Visualization and Output Generation Module.

Exports geolocation results to GeoJSON, creates interactive Folium maps,
and renders spectrogram overlay plots showing theoretical Doppler curves against audio.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import folium

from doppler_geo.phase2_audio.spectrogram import SpectrogramData
from doppler_geo.phase4_physics.doppler_modeler import DopplerCurve


@dataclass
class GeolocationCandidate:
    """Final ranked geolocation candidate with full physical metadata."""
    rank: int
    lat: float
    lon: float
    elevation_m: float
    total_score: float
    score_event_1: float
    score_event_2: float
    flight_id_1: str
    flight_id_2: str
    locked_delta_t: float
    applied_time_shift: float
    best_delta_f_1: float
    best_delta_f_2: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def export_geojson(
    candidates: List[GeolocationCandidate],
    output_path: str,
) -> str:
    """Export candidates to a GeoJSON FeatureCollection file."""
    features = []
    for c in candidates:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [c.lon, c.lat],  # GeoJSON is [lon, lat]
            },
            "properties": c.to_dict(),
        }
        features.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)

    return output_path


def generate_folium_map(
    candidates: List[GeolocationCandidate],
    flight_dfs: Optional[Dict[str, Any]] = None,
    output_path: str = "doppler_geolocation_map.html",
) -> str:
    """Create an interactive Folium HTML map showing candidate fixes and flight tracks."""
    if not candidates:
        center_lat, center_lon = 0.0, 0.0
    else:
        center_lat = candidates[0].lat
        center_lon = candidates[0].lon

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
    )

    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"]

    # Add flight trajectory lines if provided
    if flight_dfs:
        for i, (fid, fdf) in enumerate(flight_dfs.items()):
            color = colors[i % len(colors)]
            coords = list(zip(fdf["lat"].to_numpy(), fdf["lon"].to_numpy()))
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=3,
                opacity=0.8,
                tooltip=f"Flight {fid}",
            ).add_to(m)

    # Add candidate markers
    for c in candidates:
        rank_color = "red" if c.rank == 1 else ("orange" if c.rank == 2 else "blue")
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 180px;">
            <h4>Rank #{c.rank} Candidate</h4>
            <p><b>Score:</b> {c.total_score:.2f}</p>
            <p><b>Coords:</b> {c.lat:.5f}, {c.lon:.5f}</p>
            <p><b>Elevation:</b> {c.elevation_m:.1f} m</p>
            <p><b>Flights:</b> {c.flight_id_1} & {c.flight_id_2}</p>
            <p><b>Time Offset:</b> {c.locked_delta_t:+.2f} s</p>
        </div>
        """
        folium.CircleMarker(
            location=[c.lat, c.lon],
            radius=9 if c.rank == 1 else 6,
            color=rank_color,
            fill=True,
            fill_color=rank_color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Rank #{c.rank} (Score: {c.total_score:.1f})",
        ).add_to(m)

    m.save(output_path)
    return output_path


def plot_spectrogram_with_curves(
    spec_data: SpectrogramData,
    curve_e1: DopplerCurve,
    curve_e2: DopplerCurve,
    applied_time_shift: float,
    delta_f1: float = 0.0,
    delta_f2: float = 0.0,
    output_path: str = "spectrogram_overlay.png",
    title: str = "Doppler Physics Model Fit on Spectrogram",
) -> str:
    """Plot the audio spectrogram matrix with theoretical Doppler curves overlaid."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    # Plot spectrogram matrix
    times = spec_data.times
    freqs = spec_data.frequencies
    db_matrix = spec_data.db

    img = ax.pcolormesh(
        times,
        freqs,
        db_matrix,
        shading="auto",
        cmap="magma",
        vmin=-80.0,
        vmax=0.0,
    )
    plt.colorbar(img, ax=ax, label="Normalized Magnitude (dB)")

    # Overlay Event 1 theoretical curve
    t_audio_e1 = curve_e1.t_obs + applied_time_shift
    f_audio_e1 = curve_e1.f_obs + delta_f1
    valid_e1 = (t_audio_e1 >= times[0]) & (t_audio_e1 <= times[-1])
    ax.plot(
        t_audio_e1[valid_e1],
        f_audio_e1[valid_e1],
        color="cyan",
        linewidth=2.5,
        linestyle="-",
        label=f"Event 1 Model (F0={curve_e1.f0_baseline_hz + delta_f1:.1f}Hz)",
    )

    # Overlay Event 2 theoretical curve
    t_audio_e2 = curve_e2.t_obs + applied_time_shift
    f_audio_e2 = curve_e2.f_obs + delta_f2
    valid_e2 = (t_audio_e2 >= times[0]) & (t_audio_e2 <= times[-1])
    ax.plot(
        t_audio_e2[valid_e2],
        f_audio_e2[valid_e2],
        color="lime",
        linewidth=2.5,
        linestyle="-",
        label=f"Event 2 Model (F0={curve_e2.f0_baseline_hz + delta_f2:.1f}Hz)",
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time in Audio (seconds)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_ylim(spec_data.frequencies[0], spec_data.frequencies[-1])
    ax.legend(loc="upper right", framealpha=0.85)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close(fig)
    return output_path


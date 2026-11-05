"""Target Measurement Interface & Configuration Schema.

Parses, validates, and manages manual audio event measurements (CPA time,
f0 baseline, velocity range, slant distance range) and traced curve points.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict, Any


@dataclass
class TracedPoint:
    """A human-traced coordinate (timestamp in seconds, observed frequency in Hz)."""
    t: float
    f: float


@dataclass
class AudioEventMeasurement:
    """Manual measurement constraints for a single aircraft flyover event in the audio."""
    event_id: int
    cpa_time_in_audio: float
    f0_baseline_hz: float
    est_velocity_range_m_s: Tuple[float, float]
    est_slant_distance_range_m: Tuple[float, float]
    traced_points: Optional[List[TracedPoint]] = None

    def __post_init__(self):
        if self.cpa_time_in_audio < 0:
            raise ValueError(f"cpa_time_in_audio cannot be negative, got {self.cpa_time_in_audio}")
        if self.f0_baseline_hz <= 0:
            raise ValueError(f"f0_baseline_hz must be positive, got {self.f0_baseline_hz}")
        vmin, vmax = self.est_velocity_range_m_s
        if vmin > vmax or vmin <= 0:
            raise ValueError(f"Invalid velocity range: {self.est_velocity_range_m_s}")
        dmin, dmax = self.est_slant_distance_range_m
        if dmin > dmax or dmin < 0:
            raise ValueError(f"Invalid slant distance range: {self.est_slant_distance_range_m}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AudioEventMeasurement:
        v_range = tuple(data["est_velocity_range_m_s"])
        d_range = tuple(data["est_slant_distance_range_m"])
        traced = None
        if "traced_points" in data and data["traced_points"] is not None:
            traced = []
            for p in data["traced_points"]:
                if isinstance(p, (list, tuple)):
                    traced.append(TracedPoint(t=float(p[0]), f=float(p[1])))
                elif isinstance(p, dict):
                    traced.append(TracedPoint(t=float(p["t"]), f=float(p["f"])))
        return cls(
            event_id=int(data["event_id"]),
            cpa_time_in_audio=float(data["cpa_time_in_audio"]),
            f0_baseline_hz=float(data["f0_baseline_hz"]),
            est_velocity_range_m_s=(float(v_range[0]), float(v_range[1])),
            est_slant_distance_range_m=(float(d_range[0]), float(d_range[1])),
            traced_points=traced,
        )

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "event_id": self.event_id,
            "cpa_time_in_audio": self.cpa_time_in_audio,
            "f0_baseline_hz": self.f0_baseline_hz,
            "est_velocity_range_m_s": list(self.est_velocity_range_m_s),
            "est_slant_distance_range_m": list(self.est_slant_distance_range_m),
        }
        if self.traced_points:
            res["traced_points"] = [{"t": pt.t, "f": pt.f} for pt in self.traced_points]
        return res


@dataclass
class MeasurementConfig:
    """Container for multi-event measurements extracted from the audio."""
    events: List[AudioEventMeasurement] = field(default_factory=list)

    @classmethod
    def from_json(cls, json_str_or_path: str) -> MeasurementConfig:
        """Load configuration from a JSON string or file path."""
        try:
            # Check if it's a file path
            with open(json_str_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, FileNotFoundError):
            data = json.loads(json_str_or_path)

        events = [AudioEventMeasurement.from_dict(e) for e in data.get("events", [])]
        events.sort(key=lambda e: e.cpa_time_in_audio)
        return cls(events=events)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({"events": [e.to_dict() for e in self.events]}, indent=indent)

    def get_time_offset_between_events(self, idx1: int = 0, idx2: int = 1) -> float:
        """Return the audio time delta in seconds between two flyover events."""
        if len(self.events) <= max(idx1, idx2):
            raise IndexError("MeasurementConfig does not contain enough events.")
        return self.events[idx2].cpa_time_in_audio - self.events[idx1].cpa_time_in_audio


"""Phase 3: The Coarse Spatial Filter (Grid Generation)."""

from doppler_geo.phase3_coarse_filter.grid import (
    CandidateGrid,
    generate_candidate_grid,
)
from doppler_geo.phase3_coarse_filter.cpa_calculator import (
    SPEED_OF_SOUND,
    FlightCPAResults,
    CPACalculator,
)
from doppler_geo.phase3_coarse_filter.multi_event_filter import (
    CoarseCandidateMatch,
    MultiEventFilter,
)

__all__ = [
    "CandidateGrid",
    "generate_candidate_grid",
    "SPEED_OF_SOUND",
    "FlightCPAResults",
    "CPACalculator",
    "CoarseCandidateMatch",
    "MultiEventFilter",
]


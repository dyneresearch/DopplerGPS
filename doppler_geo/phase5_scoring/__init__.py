"""Phase 5: The Alignment and Scoring Engine."""

from doppler_geo.phase5_scoring.interpolation import (
    create_curve_interpolator,
    sample_spectrogram_on_curve,
)
from doppler_geo.phase5_scoring.locked_scorer import (
    EventScoreResult,
    CandidateScoreResult,
    LockedOffsetScorer,
)
from doppler_geo.phase5_scoring.traced_scorer import (
    TracedScoreResult,
    TracedLineScorer,
)

__all__ = [
    "create_curve_interpolator",
    "sample_spectrogram_on_curve",
    "EventScoreResult",
    "CandidateScoreResult",
    "LockedOffsetScorer",
    "TracedScoreResult",
    "TracedLineScorer",
]


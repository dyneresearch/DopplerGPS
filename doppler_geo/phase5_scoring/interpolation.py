"""Curve-to-Spectrogram Interpolation.

Uses scipy.interpolate.interp1d to map theoretical Doppler physics curves
onto continuous spectrogram coordinates and evaluate intensity profiles.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d

from doppler_geo.phase2_audio.spectrogram import SpectrogramData
from doppler_geo.phase4_physics.doppler_modeler import DopplerCurve


def create_curve_interpolator(
    curve: DopplerCurve,
    time_shift: float = 0.0,
    freq_shift: float = 0.0,
    kind: str = "linear",
) -> interp1d:
    """Create a 1D continuous function f_obs(t) for a Doppler curve with shifts."""
    shifted_times = curve.t_obs + time_shift
    shifted_freqs = curve.f_obs + freq_shift
    return interp1d(
        shifted_times,
        shifted_freqs,
        kind=kind,
        bounds_error=False,
        fill_value=np.nan,
    )


def sample_spectrogram_on_curve(
    spec_data: SpectrogramData,
    curve: DopplerCurve,
    time_shift: float = 0.0,
    freq_shift: float = 0.0,
    eval_step_s: Optional[float] = None,
    eval_window: Optional[Tuple[float, float]] = None,
    use_db: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample spectrogram intensity values along a theoretical Doppler curve.

    Args:
        spec_data: SpectrogramData containing magnitude/db matrices and axes.
        curve: Theoretical DopplerCurve.
        time_shift: Delta T shift applied to curve in seconds.
        freq_shift: Delta F shift applied to curve in Hz.
        eval_step_s: Time step for sampling (defaults to spectrogram hop duration).
        eval_window: Optional (start_time, end_time) to restrict evaluation to flyover window.
        use_db: If True, samples dB values; if False, samples linear magnitude.

    Returns:
        (eval_times, eval_freqs, intensities)
    """
    interp_curve = create_curve_interpolator(curve, time_shift, freq_shift)

    t_min = curve.t_obs[0] + time_shift
    t_max = curve.t_obs[-1] + time_shift

    # Restrict to spectrogram boundaries and optional window
    spec_t_min = spec_data.times[0]
    spec_t_max = spec_data.times[-1]
    t_start = max(t_min, spec_t_min)
    t_end = min(t_max, spec_t_max)

    if eval_window is not None:
        t_start = max(t_start, eval_window[0])
        t_end = min(t_end, eval_window[1])

    if t_start >= t_end:
        return np.array([]), np.array([]), np.array([])

    if eval_step_s is None:
        eval_step_s = float(spec_data.hop_length / spec_data.sr)

    eval_times = np.arange(t_start, t_end, eval_step_s)
    eval_freqs = interp_curve(eval_times)

    valid_mask = ~np.isnan(eval_freqs)
    eval_times = eval_times[valid_mask]
    eval_freqs = eval_freqs[valid_mask]

    if len(eval_times) == 0:
        return np.array([]), np.array([]), np.array([])

    intensities = spec_data.sample_curve_intensity(
        time_points=eval_times,
        frequency_points=eval_freqs,
        use_db=use_db,
    )

    return eval_times, eval_freqs, intensities


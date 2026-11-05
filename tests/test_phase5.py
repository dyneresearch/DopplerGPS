"""Unit tests for Phase 5: The Alignment and Scoring Engine."""

import pytest
import numpy as np

from doppler_geo.phase2_audio.spectrogram import SpectrogramGenerator
from doppler_geo.phase2_audio.measurement_interface import TracedPoint
from doppler_geo.phase4_physics.doppler_modeler import DopplerCurve
from doppler_geo.phase5_scoring import (
    create_curve_interpolator,
    sample_spectrogram_on_curve,
    LockedOffsetScorer,
    TracedLineScorer,
)


@pytest.fixture
def synthetic_dual_doppler_audio():
    """Create a synthetic audio signal containing two Doppler S-curves."""
    sr = 4000
    duration = 60.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Event 1 near t = 15s, frequency sweeps from 140 Hz down to 120 Hz, F0 = 130 Hz
    # Sigmoid transition centered at t = 15.0s
    f1_inst = 130.0 - 10.0 * np.tanh((t - 15.0) / 2.0)
    phase1 = 2 * np.pi * np.cumsum(f1_inst) / sr
    sig1 = 0.5 * np.sin(phase1)

    # Event 2 near t = 45s, frequency sweeps from 120 Hz down to 100 Hz, F0 = 110 Hz
    f2_inst = 110.0 - 10.0 * np.tanh((t - 45.0) / 2.0)
    phase2 = 2 * np.pi * np.cumsum(f2_inst) / sr
    sig2 = 0.5 * np.sin(phase2)

    # Combined signal with low background noise
    np.random.seed(42)
    noise = 0.05 * np.random.randn(len(t))
    audio = sig1 + sig2 + noise

    gen = SpectrogramGenerator(n_fft=2048, hop_length=512, min_freq_hz=80.0, max_freq_hz=160.0)
    spec_data = gen.generate_from_samples(audio, sr)

    return spec_data, f1_inst, f2_inst, t


def test_curve_interpolation():
    """Verify create_curve_interpolator accurately interpolates frequencies."""
    t_obs = np.array([10.0, 15.0, 20.0])
    f_obs = np.array([140.0, 130.0, 120.0])
    curve = DopplerCurve(
        t_obs=t_obs,
        f_obs=f_obs,
        t_emit=t_obs - 1.0,
        slant_distances=np.array([3000.0, 1000.0, 3000.0]),
        radial_velocities=np.array([-80.0, 0.0, 80.0]),
        f0_baseline_hz=130.0,
    )

    interp = create_curve_interpolator(curve, time_shift=0.0, freq_shift=0.0)
    assert np.isclose(interp(15.0), 130.0)
    assert np.isclose(interp(12.5), 135.0)


def test_locked_offset_scorer(synthetic_dual_doppler_audio):
    """Verify that true geometry candidate outscores misaligned candidate under locked delta T."""
    spec_data, _, _, _ = synthetic_dual_doppler_audio

    # Create matching DopplerCurve for Event 1 (centered at t = 1015s UTC, audio CPA nominal 15s)
    t_window_e1 = np.linspace(1005.0, 1025.0, 41)
    f_e1 = 130.0 - 10.0 * np.tanh((t_window_e1 - 1015.0) / 2.0)
    curve_e1 = DopplerCurve(
        t_obs=t_window_e1,
        f_obs=f_e1,
        t_emit=t_window_e1 - 1.0,
        slant_distances=np.abs(t_window_e1 - 1015.0) * 100.0 + 1000.0,
        radial_velocities=(t_window_e1 - 1015.0) * 5.0,
        f0_baseline_hz=130.0,
    )

    # Create matching DopplerCurve for Event 2 (centered at t = 1045s UTC, audio CPA nominal 45s)
    t_window_e2 = np.linspace(1035.0, 1055.0, 41)
    f_e2 = 110.0 - 10.0 * np.tanh((t_window_e2 - 1045.0) / 2.0)
    curve_e2_true = DopplerCurve(
        t_obs=t_window_e2,
        f_obs=f_e2,
        t_emit=t_window_e2 - 1.0,
        slant_distances=np.abs(t_window_e2 - 1045.0) * 100.0 + 1000.0,
        radial_velocities=(t_window_e2 - 1045.0) * 5.0,
        f0_baseline_hz=110.0,
    )

    # Bad candidate point: Event 2 passes 10 seconds later than true geometry
    t_window_e2_bad = np.linspace(1045.0, 1065.0, 41)
    curve_e2_bad = DopplerCurve(
        t_obs=t_window_e2_bad,
        f_obs=f_e2,
        t_emit=t_window_e2_bad - 1.0,
        slant_distances=np.abs(t_window_e2_bad - 1055.0) * 100.0 + 1000.0,
        radial_velocities=(t_window_e2_bad - 1055.0) * 5.0,
        f0_baseline_hz=110.0,
    )

    scorer = LockedOffsetScorer(dt_range=(-2.0, 2.0), dt_step=0.5, df_range=(-1.0, 1.0), df_step=0.5, use_db=False)

    # Score true candidate
    true_res = scorer.score_candidate(
        spec_data=spec_data,
        curve_e1=curve_e1,
        curve_e2=curve_e2_true,
        nominal_audio_cpa_1=15.0,
        nominal_audio_cpa_2=45.0,
    )

    # Score bad candidate
    bad_res = scorer.score_candidate(
        spec_data=spec_data,
        curve_e1=curve_e1,
        curve_e2=curve_e2_bad,
        nominal_audio_cpa_1=15.0,
        nominal_audio_cpa_2=45.0,
    )

    # True candidate should have significantly higher Event 2 score and total score
    assert true_res.score_event_2 > bad_res.score_event_2 * 1.5
    assert true_res.total_score > bad_res.total_score + 100.0


def test_traced_line_scorer():
    """Verify coordinate distance scoring against user-traced curve points."""
    t_obs = np.linspace(10.0, 20.0, 11)
    f_obs = 130.0 - 2.0 * (t_obs - 15.0)  # Linear slope for test
    slant_dist = (t_obs - 15.0)**2 + 1000.0
    curve = DopplerCurve(
        t_obs=t_obs,
        f_obs=f_obs,
        t_emit=t_obs,
        slant_distances=slant_dist,
        radial_velocities=np.zeros_like(t_obs),
        f0_baseline_hz=130.0,
    )

    # Traced points lying perfectly on the theoretical curve
    traced_perfect = [
        TracedPoint(t=12.0, f=136.0),
        TracedPoint(t=15.0, f=130.0),
        TracedPoint(t=18.0, f=124.0),
    ]

    # Traced points with 10 Hz error
    traced_deviant = [
        TracedPoint(t=12.0, f=146.0),
        TracedPoint(t=15.0, f=140.0),
        TracedPoint(t=18.0, f=134.0),
    ]

    scorer = TracedLineScorer(dt_range=(-1.0, 1.0), dt_step=0.5, df_range=(0.0, 0.0), df_step=1.0)

    res_perfect = scorer.score_curve_against_traced_points(
        curve=curve,
        nominal_audio_cpa_time=15.0,
        traced_points=traced_perfect,
        sweep_alignment=False,
    )
    assert np.isclose(res_perfect.mse, 0.0, atol=1e-4)

    res_deviant = scorer.score_curve_against_traced_points(
        curve=curve,
        nominal_audio_cpa_time=15.0,
        traced_points=traced_deviant,
        sweep_alignment=False,
    )
    assert np.isclose(res_deviant.mse, 100.0, atol=1e-2)

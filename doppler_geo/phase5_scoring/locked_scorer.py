"""The "Locked Offset" Alignment and Scoring Engine.

Implements the multi-dimensional grid search error-correction sweep for Event 1
and strictly locks the discovered timeline offset when scoring Event 2.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from doppler_geo.phase2_audio.spectrogram import SpectrogramData
from doppler_geo.phase4_physics.doppler_modeler import DopplerCurve
from doppler_geo.phase5_scoring.interpolation import sample_spectrogram_on_curve


@dataclass
class EventScoreResult:
    """Alignment and fitness score result for a single event."""
    best_score: float
    best_delta_t: float
    best_delta_f: float
    applied_time_shift: float


@dataclass
class CandidateScoreResult:
    """Combined fitness score for a candidate flight pair at a grid point."""
    total_score: float
    score_event_1: float
    score_event_2: float
    locked_delta_t: float
    applied_time_shift: float
    best_delta_f_1: float
    best_delta_f_2: float


class LockedOffsetScorer:
    """Evaluates candidate Doppler curves against audio spectrogram using locked offset logic."""

    def __init__(
        self,
        dt_range: Tuple[float, float] = (-5.0, 5.0),
        dt_step: float = 0.25,
        df_range: Tuple[float, float] = (-2.0, 2.0),
        df_step: float = 0.25,
        eval_window_half_sec: float = 15.0,
        use_db: bool = False,
    ):
        self.dt_range = dt_range
        self.dt_step = dt_step
        self.df_range = df_range
        self.df_step = df_step
        self.eval_window_half_sec = eval_window_half_sec
        self.use_db = use_db

    def sweep_event_1(
        self,
        spec_data: SpectrogramData,
        curve_e1: DopplerCurve,
        nominal_audio_cpa_time: float,
    ) -> EventScoreResult:
        """Grid search over delta T and delta F to find optimal alignment for Event 1."""
        base_shift_t = nominal_audio_cpa_time - curve_e1.cpa_obs_time

        dt_offsets = np.arange(self.dt_range[0], self.dt_range[1] + self.dt_step * 0.5, self.dt_step)
        df_offsets = np.arange(self.df_range[0], self.df_range[1] + self.df_step * 0.5, self.df_step)

        eval_window = (
            nominal_audio_cpa_time - self.eval_window_half_sec,
            nominal_audio_cpa_time + self.eval_window_half_sec,
        )

        best_score = -np.inf
        best_dt = 0.0
        best_df = 0.0

        for dt in dt_offsets:
            total_shift_t = base_shift_t + dt
            for df in df_offsets:
                _, _, intensities = sample_spectrogram_on_curve(
                    spec_data=spec_data,
                    curve=curve_e1,
                    time_shift=total_shift_t,
                    freq_shift=df,
                    eval_window=eval_window,
                    use_db=self.use_db,
                )
                if len(intensities) == 0:
                    continue
                score = float(np.mean(intensities))
                if score > best_score:
                    best_score = score
                    best_dt = float(dt)
                    best_df = float(df)

        best_applied_shift = base_shift_t + best_dt

        return EventScoreResult(
            best_score=best_score,
            best_delta_t=best_dt,
            best_delta_f=best_df,
            applied_time_shift=best_applied_shift,
        )

    def score_event_2_locked(
        self,
        spec_data: SpectrogramData,
        curve_e2: DopplerCurve,
        applied_time_shift: float,
        nominal_audio_cpa_2: Optional[float] = None,
    ) -> EventScoreResult:
        """Score Event 2 using the strictly LOCKED timeline shift discovered from Event 1.

        Args:
            spec_data: Audio SpectrogramData.
            curve_e2: Theoretical DopplerCurve for Event 2.
            applied_time_shift: The exact global UTC-to-audio time shift from Event 1.
            nominal_audio_cpa_2: Optional audio CPA time to focus scoring window.

        Returns:
            EventScoreResult with best_score, best_delta_f, and applied_time_shift.
        """
        df_offsets = np.arange(self.df_range[0], self.df_range[1] + self.df_step * 0.5, self.df_step)

        eval_window = None
        if nominal_audio_cpa_2 is not None:
            eval_window = (
                nominal_audio_cpa_2 - self.eval_window_half_sec,
                nominal_audio_cpa_2 + self.eval_window_half_sec,
            )

        best_score = -np.inf
        best_df = 0.0

        for df in df_offsets:
            _, _, intensities = sample_spectrogram_on_curve(
                spec_data=spec_data,
                curve=curve_e2,
                time_shift=applied_time_shift,
                freq_shift=df,
                eval_window=eval_window,
                use_db=self.use_db,
            )
            if len(intensities) == 0:
                continue
            score = float(np.mean(intensities))
            if score > best_score:
                best_score = score
                best_df = float(df)

        return EventScoreResult(
            best_score=best_score,
            best_delta_t=0.0,
            best_delta_f=best_df,
            applied_time_shift=applied_time_shift,
        )

    def score_candidate(
        self,
        spec_data: SpectrogramData,
        curve_e1: DopplerCurve,
        curve_e2: DopplerCurve,
        nominal_audio_cpa_1: float,
        nominal_audio_cpa_2: Optional[float] = None,
    ) -> CandidateScoreResult:
        """Run the complete locked scoring pipeline on a candidate grid point."""
        res_e1 = self.sweep_event_1(spec_data, curve_e1, nominal_audio_cpa_1)
        res_e2 = self.score_event_2_locked(
            spec_data,
            curve_e2,
            applied_time_shift=res_e1.applied_time_shift,
            nominal_audio_cpa_2=nominal_audio_cpa_2,
        )

        total_score = res_e1.best_score + res_e2.best_score

        return CandidateScoreResult(
            total_score=total_score,
            score_event_1=res_e1.best_score,
            score_event_2=res_e2.best_score,
            locked_delta_t=res_e1.best_delta_t,
            applied_time_shift=res_e1.applied_time_shift,
            best_delta_f_1=res_e1.best_delta_f,
            best_delta_f_2=res_e2.best_delta_f,
        )

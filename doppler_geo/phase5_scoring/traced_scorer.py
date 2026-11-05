"""Alternative Scoring: The "Traced Line" Method (Coordinate Distance Scoring).

Computes Mean Squared Error (MSE) or Mean Absolute Error (MAE) between the
theoretical Doppler curve and human-traced (timestamp, frequency) coordinates.
Designed for faint signals where background acoustic noise overpowers pixel brightness.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from doppler_geo.phase2_audio.measurement_interface import TracedPoint
from doppler_geo.phase4_physics.doppler_modeler import DopplerCurve
from doppler_geo.phase5_scoring.interpolation import create_curve_interpolator


@dataclass
class TracedScoreResult:
    """Distance error metrics against human-traced coordinate curve."""
    mse: float
    mae: float
    aligned_delta_t: float
    aligned_delta_f: float
    num_points_matched: int


class TracedLineScorer:
    """Calculates coordinate distance error between theoretical physics curves and traced points."""

    def __init__(
        self,
        dt_range: Tuple[float, float] = (-5.0, 5.0),
        dt_step: float = 0.2,
        df_range: Tuple[float, float] = (-2.0, 2.0),
        df_step: float = 0.2,
    ):
        self.dt_range = dt_range
        self.dt_step = dt_step
        self.df_range = df_range
        self.df_step = df_step

    def score_curve_against_traced_points(
        self,
        curve: DopplerCurve,
        nominal_audio_cpa_time: float,
        traced_points: List[TracedPoint],
        sweep_alignment: bool = True,
        locked_dt: Optional[float] = None,
    ) -> TracedScoreResult:
        """Calculate MSE and MAE between theoretical curve and traced points.

        Args:
            curve: Theoretical DopplerCurve.
            nominal_audio_cpa_time: Nominal CPA time in audio.
            traced_points: List of TracedPoint(t, f) from user interface.
            sweep_alignment: If True and locked_dt is None, sweeps dt and df to minimize MSE.
            locked_dt: If provided, locks dt to this exact value and sweeps only df.

        Returns:
            TracedScoreResult with mse, mae, and alignment offsets.
        """
        if not traced_points:
            raise ValueError("traced_points list cannot be empty.")

        t_traced = np.array([pt.t for pt in traced_points], dtype=np.float64)
        f_traced = np.array([pt.f for pt in traced_points], dtype=np.float64)

        base_shift_t = nominal_audio_cpa_time - curve.cpa_obs_time

        if locked_dt is not None:
            dt_offsets = [locked_dt]
        elif sweep_alignment:
            dt_offsets = np.arange(self.dt_range[0], self.dt_range[1] + self.dt_step * 0.5, self.dt_step)
        else:
            dt_offsets = [0.0]

        df_offsets = np.arange(self.df_range[0], self.df_range[1] + self.df_step * 0.5, self.df_step)

        best_mse = np.inf
        best_mae = np.inf
        best_dt = 0.0
        best_df = 0.0
        best_matched = 0

        for dt in dt_offsets:
            total_shift_t = base_shift_t + dt
            interp = create_curve_interpolator(curve, time_shift=total_shift_t, freq_shift=0.0)
            f_model_base = interp(t_traced)

            valid_mask = ~np.isnan(f_model_base)
            if np.sum(valid_mask) < len(t_traced) // 2:
                continue

            for df in df_offsets:
                f_model = f_model_base[valid_mask] + df
                diff = f_model - f_traced[valid_mask]
                mse = float(np.mean(diff**2))
                mae = float(np.mean(np.abs(diff)))

                if mse < best_mse:
                    best_mse = mse
                    best_mae = mae
                    best_dt = float(dt)
                    best_df = float(df)
                    best_matched = int(np.sum(valid_mask))

        return TracedScoreResult(
            mse=best_mse,
            mae=best_mae,
            aligned_delta_t=best_dt,
            aligned_delta_f=best_df,
            num_points_matched=best_matched,
        )


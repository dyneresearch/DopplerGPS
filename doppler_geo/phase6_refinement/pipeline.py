"""End-to-End Orchestration Pipeline for Project Doppler-Geo.

Sequentially executes Phases 1 through 6:
1. Ingestion of ADSB parquet & DEM elevation data.
2. Audio STFT spectrogram generation & measurement schema parsing.
3. Coarse 200m spatial grid filtering with delayed CPA and 10km silence checks.
4. Trajectory cubic spline interpolation and 3D line-of-sight Doppler modeling.
5. Spectrogram alignment with locked delta-T offset scoring.
6. High-resolution micro-grid refinement (10-25m) and visualization generation.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from doppler_geo.phase1_ingestion import (
    BoundingBox,
    TimeWindow,
    ADSBDataLoader,
    DEMHandler,
    SyntheticDEMHandler,
)
from doppler_geo.phase2_audio import (
    SpectrogramGenerator,
    SpectrogramData,
    MeasurementConfig,
)
from doppler_geo.phase3_coarse_filter import (
    CandidateGrid,
    generate_candidate_grid,
    MultiEventFilter,
    CoarseCandidateMatch,
    CPACalculator,
)
from doppler_geo.phase4_physics import (
    TrajectorySpline,
    DopplerModeler,
    DopplerCurve,
)
from doppler_geo.phase5_scoring import (
    LockedOffsetScorer,
    CandidateScoreResult,
)
from doppler_geo.phase6_refinement.micro_grid import generate_micro_grid
from doppler_geo.coordinates import compute_speed_of_sound, horizontal_haversine_distance
from doppler_geo.phase6_refinement.visualization import (
    GeolocationCandidate,
    export_geojson,
    generate_folium_map,
    plot_spectrogram_with_curves,
)


@dataclass
class PipelineResult:
    """Execution output from DopplerGeoPipeline."""
    top_candidates: List[GeolocationCandidate]
    coarse_matches_count: int
    confidence_status: str = "HIGH_CONFIDENCE"  # HIGH_CONFIDENCE, MODERATE_CONFIDENCE, INCONCLUSIVE
    diagnostics: str = "Success"
    geojson_path: Optional[str] = None
    map_html_path: Optional[str] = None
    plot_path: Optional[str] = None


class DopplerGeoPipeline:
    """Production-grade pipeline executing Phases 1 through 6."""

    def __init__(
        self,
        adsb_loader: Optional[ADSBDataLoader] = None,
        dem_handler: Optional[DEMHandler] = None,
        spec_generator: Optional[SpectrogramGenerator] = None,
        time_tolerance_s: float = 5.0,
        silence_distance_threshold_m: float = 10000.0,
        coarse_resolution_m: float = 200.0,
        micro_resolution_m: float = 20.0,
        micro_radius_m: float = 400.0,
        temperature_c: float = 20.0,
        speed_of_sound: Optional[float] = None,
    ):
        self.loader = adsb_loader or ADSBDataLoader()
        self.dem = dem_handler or SyntheticDEMHandler()
        self.spec_gen = spec_generator or SpectrogramGenerator()
        self.time_tolerance_s = time_tolerance_s
        self.silence_threshold_m = silence_distance_threshold_m
        self.coarse_resolution_m = coarse_resolution_m
        self.micro_resolution_m = micro_resolution_m
        self.micro_radius_m = micro_radius_m
        self.temperature_c = temperature_c
        self.speed_of_sound = speed_of_sound or compute_speed_of_sound(temperature_c)

        self.multi_filter = MultiEventFilter(
            time_tolerance_s=self.time_tolerance_s,
            silence_distance_threshold_m=self.silence_threshold_m,
        )
        self.multi_filter.cpa_calculator = CPACalculator(speed_of_sound=self.speed_of_sound)
        self.modeler = DopplerModeler(speed_of_sound=self.speed_of_sound)
        self.scorer = LockedOffsetScorer()

    def run(
        self,
        adsb_parquet_path: Union[str, List[str]],
        measurement_config: MeasurementConfig,
        search_bbox: BoundingBox,
        audio_path_or_spec: Union[str, SpectrogramData],
        output_dir: str = "output",
        top_k: int = 3,
    ) -> PipelineResult:
        """Run the full end-to-end Doppler geolocation pipeline."""
        os.makedirs(output_dir, exist_ok=True)

        # -------------------------------------------------------------
        # Phase 1: ADSB Data Ingestion & Terrain Preparation
        # -------------------------------------------------------------
        # Estimate broad time window covering the events
        e1 = measurement_config.events[0]
        e2 = measurement_config.events[1]

        df_all = self.loader.load_flight_data(
            parquet_path=adsb_parquet_path,
            bbox=search_bbox,
            only_propellers=True,
        )
        if df_all.empty:
            raise ValueError("No propeller aircraft found in ADSB data matching the search bounding box.")

        flight_dfs: Dict[str, pd.DataFrame] = {}
        for fid, group in df_all.groupby("flight_id"):
            flight_dfs[str(fid)] = group.reset_index(drop=True)

        # -------------------------------------------------------------
        # Phase 2: Audio Spectrogram Generation
        # -------------------------------------------------------------
        if isinstance(audio_path_or_spec, SpectrogramData):
            spec_data = audio_path_or_spec
        else:
            spec_data = self.spec_gen.generate_from_audio(audio_path_or_spec)

        # -------------------------------------------------------------
        # Phase 3: Coarse Spatial Filter (200m Grid)
        # -------------------------------------------------------------
        coarse_grid = generate_candidate_grid(search_bbox, resolution_m=self.coarse_resolution_m)
        coarse_elevations = self.dem.get_elevations(coarse_grid.lats, coarse_grid.lons)

        coarse_matches = self.multi_filter.filter_grid(
            grid=coarse_grid,
            grid_elevations=coarse_elevations,
            flight_dfs=flight_dfs,
            measurement_config=measurement_config,
        )

        if not coarse_matches:
            return PipelineResult(
                top_candidates=[],
                coarse_matches_count=0,
                confidence_status="INCONCLUSIVE",
                diagnostics="No candidate grid points satisfied the dual-event flight timing and 10km silence criteria.",
            )

        # -------------------------------------------------------------
        # Phase 4 & 5: Coarse Doppler Modeling & Locked Offset Scoring
        # -------------------------------------------------------------
        # Pre-fit splines for unique flights involved in coarse matches
        unique_fids = set()
        for m in coarse_matches:
            unique_fids.add(m.flight_id_1)
            unique_fids.add(m.flight_id_2)

        splines: Dict[str, TrajectorySpline] = {}
        for fid in unique_fids:
            splines[fid] = TrajectorySpline(
                flight_df=flight_dfs[fid],
                ref_lat=coarse_grid.ref_lat,
                ref_lon=coarse_grid.ref_lon,
            )

        scored_coarse: List[Tuple[float, CoarseCandidateMatch, CandidateScoreResult]] = []

        for match in coarse_matches:
            spline1 = splines[match.flight_id_1]
            spline2 = splines[match.flight_id_2]

            traj1 = spline1.sample_window(center_time_utc=match.flight_cpa_1, window_half_sec=60.0)
            traj2 = spline2.sample_window(center_time_utc=match.flight_cpa_2, window_half_sec=60.0)

            curve1 = self.modeler.model_curve(
                trajectory=traj1,
                obs_lat=match.lat,
                obs_lon=match.lon,
                obs_elevation_m=match.elevation_m,
                f0_baseline_hz=e1.f0_baseline_hz,
            )
            curve2 = self.modeler.model_curve(
                trajectory=traj2,
                obs_lat=match.lat,
                obs_lon=match.lon,
                obs_elevation_m=match.elevation_m,
                f0_baseline_hz=e2.f0_baseline_hz,
            )

            score_res = self.scorer.score_candidate(
                spec_data=spec_data,
                curve_e1=curve1,
                curve_e2=curve2,
                nominal_audio_cpa_1=e1.cpa_time_in_audio,
                nominal_audio_cpa_2=e2.cpa_time_in_audio,
            )
            scored_coarse.append((score_res.total_score, match, score_res))

        # Sort coarse candidates descending by score
        scored_coarse.sort(key=lambda x: x[0], reverse=True)
        best_coarse_match = scored_coarse[0][1]

        # -------------------------------------------------------------
        # Phase 6: High-Resolution Refinement (Micro-Grid) & Visualization
        # -------------------------------------------------------------
        micro_grid = generate_micro_grid(
            center_lat=best_coarse_match.lat,
            center_lon=best_coarse_match.lon,
            radius_m=self.micro_radius_m,
            resolution_m=self.micro_resolution_m,
        )
        micro_elevations = self.dem.get_elevations(micro_grid.lats, micro_grid.lons)

        # Micro-score points for the winning flight pair
        fid1 = best_coarse_match.flight_id_1
        fid2 = best_coarse_match.flight_id_2
        spline1 = splines[fid1]
        spline2 = splines[fid2]

        traj1 = spline1.sample_window(center_time_utc=best_coarse_match.flight_cpa_1, window_half_sec=60.0)
        traj2 = spline2.sample_window(center_time_utc=best_coarse_match.flight_cpa_2, window_half_sec=60.0)

        micro_scores: List[Tuple[float, float, float, float, CandidateScoreResult, DopplerCurve, DopplerCurve]] = []

        for i in range(micro_grid.num_points):
            lat_pt = float(micro_grid.lats[i])
            lon_pt = float(micro_grid.lons[i])
            elev_pt = float(micro_elevations[i])

            curve1 = self.modeler.model_curve(
                trajectory=traj1,
                obs_lat=lat_pt,
                obs_lon=lon_pt,
                obs_elevation_m=elev_pt,
                f0_baseline_hz=e1.f0_baseline_hz,
            )
            curve2 = self.modeler.model_curve(
                trajectory=traj2,
                obs_lat=lat_pt,
                obs_lon=lon_pt,
                obs_elevation_m=elev_pt,
                f0_baseline_hz=e2.f0_baseline_hz,
            )

            score_res = self.scorer.score_candidate(
                spec_data=spec_data,
                curve_e1=curve1,
                curve_e2=curve2,
                nominal_audio_cpa_1=e1.cpa_time_in_audio,
                nominal_audio_cpa_2=e2.cpa_time_in_audio,
            )
            micro_scores.append((score_res.total_score, lat_pt, lon_pt, elev_pt, score_res, curve1, curve2))

        micro_scores.sort(key=lambda x: x[0], reverse=True)

        final_candidates: List[GeolocationCandidate] = []
        for rank_idx, (tot_score, lat_pt, lon_pt, elev_pt, s_res, _, _) in enumerate(micro_scores[:top_k], start=1):
            final_candidates.append(
                GeolocationCandidate(
                    rank=rank_idx,
                    lat=lat_pt,
                    lon=lon_pt,
                    elevation_m=elev_pt,
                    total_score=tot_score,
                    score_event_1=s_res.score_event_1,
                    score_event_2=s_res.score_event_2,
                    flight_id_1=fid1,
                    flight_id_2=fid2,
                    locked_delta_t=s_res.locked_delta_t,
                    applied_time_shift=s_res.applied_time_shift,
                    best_delta_f_1=s_res.best_delta_f_1,
                    best_delta_f_2=s_res.best_delta_f_2,
                )
            )

        # Output Generation
        geojson_file = os.path.join(output_dir, "results.geojson")
        export_geojson(final_candidates, geojson_file)

        map_file = os.path.join(output_dir, "map.html")
        generate_folium_map(
            final_candidates,
            flight_dfs={fid1: flight_dfs[fid1], fid2: flight_dfs[fid2]},
            output_path=map_file,
        )

        plot_file = os.path.join(output_dir, "spectrogram_overlay.png")
        best_peak = micro_scores[0]
        plot_spectrogram_with_curves(
            spec_data=spec_data,
            curve_e1=best_peak[5],
            curve_e2=best_peak[6],
            applied_time_shift=best_peak[4].applied_time_shift,
            delta_f1=best_peak[4].best_delta_f_1,
            delta_f2=best_peak[4].best_delta_f_2,
            output_path=plot_file,
        )

        # Confidence assessment
        best_score = final_candidates[0].total_score
        runner_up = final_candidates[1].total_score if len(final_candidates) > 1 else 0.0

        if best_score < 5.0:
            confidence = "INCONCLUSIVE"
            diagnostics = "Acoustic correlation is too weak to confirm candidate geolocation."
        elif best_score > runner_up * 1.05 or best_score > 100.0:
            confidence = "HIGH_CONFIDENCE"
            diagnostics = "Strong mathematical peak identified with clear multi-event temporal and Doppler lock."
        else:
            confidence = "MODERATE_CONFIDENCE"
            diagnostics = "Close score cluster between adjacent spatial candidates."

        return PipelineResult(
            top_candidates=final_candidates,
            coarse_matches_count=len(coarse_matches),
            confidence_status=confidence,
            diagnostics=diagnostics,
            geojson_path=geojson_file,
            map_html_path=map_file,
            plot_path=plot_file,
        )


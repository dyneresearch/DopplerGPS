"""Command Line Interface (CLI) for Project DopplerGPS.

Supports:
- Running full geolocation jobs from ADS-B parquet files and audio recordings ('run')
- Launching an automated end-to-end synthetic demo ('demo')
- Fetching & converting real-world ADS-B telemetry ('fetch-adsb')
- Inspecting audio recordings for acoustic harmonics and CPA peaks ('inspect-audio')
"""

from __future__ import annotations
import argparse
import os
import sys
import tempfile
import pathlib

from doppler_geo.phase1_ingestion import BoundingBox, SyntheticDEMHandler, DEMHandler
from doppler_geo.phase2_audio import MeasurementConfig, SpectrogramGenerator, inspect_audio
from doppler_geo.phase6_refinement import DopplerGeoPipeline
from doppler_geo.coordinates import horizontal_haversine_distance
from doppler_geo.real_data_fetcher import (
    fetch_opensky_live_states,
    fetch_opensky_track,
    convert_adsbx_json_to_parquet,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dopplergps",
        description="DopplerGPS: Acoustic Geolocation Engine (Dyne Research)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: demo
    demo_parser = subparsers.add_parser("demo", help="Run automated end-to-end synthetic demo")
    demo_parser.add_argument("--output-dir", default="demo_output", help="Directory to save output files")

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Run Doppler geolocation on audio and ADS-B data")
    run_parser.add_argument("--adsb", required=True, help="Path or glob to ADSB parquet files")
    run_parser.add_argument("--audio", required=True, help="Path to ambient audio file (.wav, .mp3, .m4a)")
    run_parser.add_argument("--config", required=True, help="Path to manual measurement JSON config")
    run_parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON"),
        required=True,
        help="Search bounding box: min_lat max_lat min_lon max_lon",
    )
    run_parser.add_argument("--dem", default=None, help="Path to DEM GeoTIFF file (optional)")
    run_parser.add_argument("--temperature", type=float, default=20.0, help="Ambient temperature in Celsius (default 20.0)")
    run_parser.add_argument("--output-dir", default="output", help="Output directory")
    run_parser.add_argument("--coarse-res", type=float, default=200.0, help="Coarse grid resolution in meters")
    run_parser.add_argument("--micro-res", type=float, default=20.0, help="Micro-grid resolution in meters")
    run_parser.add_argument("--micro-radius", type=float, default=400.0, help="Micro-grid radius in meters")
    run_parser.add_argument("--top-k", type=int, default=3, help="Number of top candidates to export")

    # Subcommand: fetch-adsb
    fetch_parser = subparsers.add_parser("fetch-adsb", help="Fetch or convert real-world ADS-B telemetry")
    fetch_parser.add_argument(
        "--source",
        choices=["opensky-live", "opensky-track", "adsbx-json"],
        default="opensky-live",
        help="Telemetry source or conversion mode (default: opensky-live)",
    )
    fetch_parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON"),
        default=None,
        help="Bounding box (required for opensky-live, optional for adsbx-json)",
    )
    fetch_parser.add_argument("--output", required=True, help="Destination .parquet file path")
    fetch_parser.add_argument("--icao24", help="24-bit ICAO transponder hex (for opensky-track)")
    fetch_parser.add_argument("--time", type=int, default=0, help="Target UTC epoch seconds (for opensky-track)")
    fetch_parser.add_argument("--input-json", help="Input JSON file or glob pattern (for adsbx-json)")
    fetch_parser.add_argument("--username", help="OpenSky username (optional)")
    fetch_parser.add_argument("--password", help="OpenSky password (optional)")

    # Subcommand: inspect-audio
    inspect_parser = subparsers.add_parser("inspect-audio", help="Inspect audio spectrum and detect candidate harmonics")
    inspect_parser.add_argument("--audio", required=True, help="Path to raw audio recording")
    inspect_parser.add_argument("--output-plot", default="audio_diagnostic.png", help="Path to save diagnostic plot PNG")
    inspect_parser.add_argument("--fmin", type=float, default=50.0, help="Lower frequency bound in Hz (default: 50.0)")
    inspect_parser.add_argument("--fmax", type=float, default=400.0, help="Upper frequency bound in Hz (default: 400.0)")
    inspect_parser.add_argument("--n-fft", type=int, default=8192, help="FFT window size (default: 8192)")
    inspect_parser.add_argument("--hop", type=int, default=512, help="FFT hop size (default: 512)")

    return parser


def run_demo(output_dir: str) -> None:
    print("=" * 70)
    print("  DOPPLERGPS: AUTOMATED SYNTHETIC DEMONSTRATION")
    print("  Dyne Research (https://dyneresearch.com)")
    print("=" * 70)
    print(f"Creating realistic flight telemetry and audio simulation in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    from doppler_geo.simulation import create_simulation_data
    with tempfile.TemporaryDirectory() as tmp_dir:
        env = create_simulation_data(pathlib.Path(tmp_dir))
        print(f"-> True Ground Location: {env['true_lat']:.5f}, {env['true_lon']:.5f}")
        print(f"-> ADSB Data: Propeller flights N172_E1, PA28_E2, and jet distractor UAL999")
        print(f"-> Audio: Synthesized ambient audio with physical Doppler sweeps")

        pipeline = DopplerGeoPipeline(
            dem_handler=env["dem"],
            coarse_resolution_m=100.0,
            micro_resolution_m=15.0,
            micro_radius_m=200.0,
            time_tolerance_s=5.0,
            spec_generator=SpectrogramGenerator(
                n_fft=4096,
                hop_length=256,
                min_freq_hz=90.0,
                max_freq_hz=150.0,
            ),
        )

        print("\nExecuting Pipeline (Phases 1-6)...")
        result = pipeline.run(
            adsb_parquet_path=env["parquet_path"],
            measurement_config=env["config"],
            search_bbox=env["bbox"],
            audio_path_or_spec=env["audio_path"],
            output_dir=output_dir,
            top_k=3,
        )

        print("\n" + "=" * 70)
        print("  GEOLOCATION RESULTS")
        print("=" * 70)
        for cand in result.top_candidates:
            dist = horizontal_haversine_distance(cand.lat, cand.lon, env["true_lat"], env["true_lon"])
            print(f"Rank #{cand.rank}:")
            print(f"  Coordinates:    {cand.lat:.5f}, {cand.lon:.5f} (Elevation: {cand.elevation_m:.1f}m)")
            print(f"  Distance Error: {dist:.1f} meters from ground truth")
            print(f"  Fitness Score:  {cand.total_score:.2f} (E1: {cand.score_event_1:.2f}, E2: {cand.score_event_2:.2f})")
            print(f"  Flight Pair:    {cand.flight_id_1} & {cand.flight_id_2}")
            print(f"  Confidence:     {result.confidence_status}")
            print(f"  Locked Delta T: {cand.locked_delta_t:+.2f}s (applied UTC shift: {cand.applied_time_shift:+.2f}s)")
            print("-" * 70)

        print(f"\nGenerated Deliverables in '{output_dir}':")
        print(f"  1. GeoJSON Results:      {result.geojson_path}")
        print(f"  2. Interactive Map:      {result.map_html_path}")
        print(f"  3. Spectrogram Overlay:  {result.plot_path}")


def run_fetch_adsb(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("  DOPPLERGPS: REAL-WORLD ADS-B TELEMETRY INGESTION")
    print("=" * 70)

    if args.source == "opensky-live":
        if not args.bbox:
            print("[ERROR] --bbox min_lat max_lat min_lon max_lon is required for opensky-live mode.")
            sys.exit(1)
        min_lat, max_lat, min_lon, max_lon = args.bbox
        bbox = BoundingBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)
        print(f"Querying OpenSky Network live state vectors for BBox: {bbox}...")
        df = fetch_opensky_live_states(
            bbox=bbox,
            output_parquet_path=args.output,
            username=args.username,
            password=args.password,
        )
        print(f"✓ Retrieved {len(df)} aircraft states. Saved to: {args.output}")

    elif args.source == "opensky-track":
        if not args.icao24:
            print("[ERROR] --icao24 (hex address) is required for opensky-track mode.")
            sys.exit(1)
        print(f"Querying historical track for ICAO24 {args.icao24} at UTC {args.time}...")
        df = fetch_opensky_track(
            icao24=args.icao24,
            target_time_utc=args.time,
            output_parquet_path=args.output,
            username=args.username,
            password=args.password,
        )
        print(f"✓ Retrieved {len(df)} track waypoints. Saved to: {args.output}")

    elif args.source == "adsbx-json":
        if not args.input_json:
            print("[ERROR] --input-json is required for adsbx-json mode.")
            sys.exit(1)
        bbox = None
        if args.bbox:
            min_lat, max_lat, min_lon, max_lon = args.bbox
            bbox = BoundingBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)
        print(f"Converting ADS-B Exchange / readsb JSON from: {args.input_json}...")
        df = convert_adsbx_json_to_parquet(
            json_path_or_glob=args.input_json,
            output_parquet_path=args.output,
            bbox=bbox,
        )
        print(f"✓ Converted {len(df)} fixes to canonical Parquet: {args.output}")


def run_inspect_audio(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("  DOPPLERGPS: AUDIO SPECTRAL INSPECTOR & PEAK DETECTOR")
    print("=" * 70)
    diag = inspect_audio(
        audio_path=args.audio,
        output_image_path=args.output_plot,
        min_freq_hz=args.fmin,
        max_freq_hz=args.fmax,
        n_fft=args.n_fft,
        hop_length=args.hop,
    )
    print(diag.summary())
    print("\nNext Steps for Measurement Configuration:")
    print("  1. Open the generated plot to locate the Doppler S-curve.")
    print("  2. Identify the inflection timestamp (t_cpa) and center frequency (f0).")
    print("  3. Insert these values into your measurements.json config.")


LEGAL_NOTICE = """
[LEGAL NOTICE] DopplerGPS is published strictly for academic research, atmospheric acoustics
study, and authorized defensive OSINT verification. Malicious surveillance, unlawful tracking,
and privacy violations are strictly prohibited. Dyne Research assumes no liability for misuse.
See DISCLAIMER.md for full terms.
"""


def main() -> None:
    print(LEGAL_NOTICE.strip())
    print("-" * 70)
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "demo":
        run_demo(args.output_dir)
    elif args.command == "fetch-adsb":
        run_fetch_adsb(args)
    elif args.command == "inspect-audio":
        run_inspect_audio(args)
    elif args.command == "run":
        min_lat, max_lat, min_lon, max_lon = args.bbox
        bbox = BoundingBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)
        config = MeasurementConfig.from_json(args.config)

        dem = DEMHandler(args.dem) if args.dem else SyntheticDEMHandler()
        pipeline = DopplerGeoPipeline(
            dem_handler=dem,
            coarse_resolution_m=args.coarse_res,
            micro_resolution_m=args.micro_res,
            micro_radius_m=args.micro_radius,
            temperature_c=args.temperature,
        )

        result = pipeline.run(
            adsb_parquet_path=args.adsb,
            measurement_config=config,
            search_bbox=bbox,
            audio_path_or_spec=args.audio,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )
        print(f"Pipeline completed successfully. {len(result.top_candidates)} candidates exported to {args.output_dir}.")


if __name__ == "__main__":
    main()

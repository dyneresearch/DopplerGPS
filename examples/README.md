# DopplerGPS Operational Field Guide

This directory contains configuration templates and end-to-end instructions for running DopplerGPS on **real-world outdoor audio recordings and ADS-B flight logs**.

---

## 🧭 Step-by-Step Field Methodology

```mermaid
flowchart LR
    A[Raw Ambient Audio] -->|inspect-audio| B[Diagnostic Spectrogram]
    B -->|Identify F0 & t_CPA| C[measurements.json]
    D[OpenSky / ADSBx] -->|fetch-adsb| E[flights.parquet]
    C --> F[dopplergps run]
    E --> F
    F --> G[Folium Map & GeoJSON Fix]
```

### Step 1: Pre-Inspect Audio & Detect Propeller Harmonics

Outdoor audio typically contains wind, traffic, and background noise. Run the spectral inspector to identify blade-pass fundamental frequencies ($F_0$) and Closest Point of Approach ($t_{\text{CPA}}$):

```bash
dopplergps inspect-audio \
  --audio "field_recording.wav" \
  --output-plot "diagnostic_spectrum.png" \
  --fmin 60.0 \
  --fmax 350.0
```

1. Open `diagnostic_spectrum.png`.
2. Locate the characteristic **S-curve** (pitch drops from high to low).
3. The inflection point (where the slope is steepest) is your **$t_{\text{CPA}}$** (in audio seconds).
4. The frequency at this inflection point is the unshifted fundamental frequency **$F_0$** (in Hz).

---

### Step 2: Prepare Your Measurement Configuration

Copy `examples/measurement_template.json` to your workspace:

```bash
cp examples/measurement_template.json my_measurements.json
```

Populate the parameters for both flyover events:
* `cpa_time_in_audio`: Inflection timestamp in seconds.
* `f0_baseline_hz`: Fundamental frequency at CPA.
* `est_velocity_range_m_s`: Conservative aircraft speed bounds (e.g. $[70, 140]$ m/s for general aviation).
* `est_slant_distance_range_m`: Conservative distance bounds (e.g. $[1000, 5000]$ m).

> **Note on Dual Events:** Doppler triangulation requires two flyovers (either two different propeller aircraft or the same aircraft on a return leg). The shared clock offset invariant will lock both events to a single physical ground coordinate.

---

### Step 3: Fetch or Convert ADS-B Flight Telemetry

#### Option A: Fetch Live States from OpenSky Network
```bash
dopplergps fetch-adsb \
  --source opensky-live \
  --bbox 39.80 40.20 -75.30 -74.70 \
  --output "flights.parquet"
```

#### Option B: Fetch a Specific Historical Flight Track
If you know the aircraft's transponder ICAO24 hex address (e.g. `a12345`):
```bash
dopplergps fetch-adsb \
  --source opensky-track \
  --icao24 "a12345" \
  --time 1700000000 \
  --output "flights.parquet"
```

#### Option C: Ingest Raw ADS-B Exchange / readsb JSON Archives
If you downloaded tar1090/readsb `aircraft.json` or history snapshots:
```bash
dopplergps fetch-adsb \
  --source adsbx-json \
  --input-json "/path/to/history/*.json" \
  --bbox 39.80 40.20 -75.30 -74.70 \
  --output "flights.parquet"
```

---

### Step 4: Execute the Geolocation Pipeline

Run the solver over your bounding box:

```bash
dopplergps run \
  --adsb "flights.parquet" \
  --audio "field_recording.wav" \
  --config "my_measurements.json" \
  --bbox 39.80 40.20 -75.30 -74.70 \
  --output-dir "case_output" \
  --temperature 18.5 \
  --coarse-res 200.0 \
  --micro-res 20.0
```

---

### Step 5: Deliverables & Verification

The pipeline outputs:
1. `case_output/map.html`: Interactive web map with candidate coordinate markers, terrain contours, and 3D flight paths.
2. `case_output/results.geojson`: Standard GIS vector file with confidence classifications and fitness rankings.
3. `case_output/spectrogram_overlay.png`: Visual verification proving exact alignment between theoretical Doppler curves and audio spectral energy.


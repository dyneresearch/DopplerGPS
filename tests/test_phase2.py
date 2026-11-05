"""Unit tests for Phase 2: Audio Feature Extraction (The Spectrogram Module)."""

import json
import pytest
import numpy as np
import soundfile as sf

from doppler_geo.phase2_audio import (
    SpectrogramGenerator,
    SpectrogramData,
    MeasurementConfig,
    AudioEventMeasurement,
    TracedPoint,
)


@pytest.fixture
def synthetic_tone_audio(tmp_path):
    """Generate a 5-second 130 Hz tone audio file."""
    sr = 8000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave at 130 Hz
    y = 0.5 * np.sin(2 * np.pi * 130.0 * t)

    audio_file = str(tmp_path / "tone_130hz.wav")
    sf.write(audio_file, y, sr)
    return audio_file, y, sr


def test_spectrogram_generation_and_peak_frequency(synthetic_tone_audio):
    """Verify STFT spectrogram computes correct frequency peak at 130 Hz."""
    audio_file, y, sr = synthetic_tone_audio
    gen = SpectrogramGenerator(n_fft=4096, hop_length=1024, min_freq_hz=50.0, max_freq_hz=300.0)
    spec_data = gen.generate_from_samples(y, sr)

    assert isinstance(spec_data, SpectrogramData)
    assert spec_data.magnitude.shape[0] == len(spec_data.frequencies)
    assert spec_data.magnitude.shape[1] == len(spec_data.times)

    # Find the frequency bin with highest average power
    mean_power = np.mean(spec_data.magnitude, axis=1)
    peak_freq = spec_data.frequencies[np.argmax(mean_power)]
    assert np.isclose(peak_freq, 130.0, atol=2.0)


def test_spectrogram_curve_sampling(synthetic_tone_audio):
    """Verify sample_curve_intensity accurately extracts intensity along a track."""
    _, y, sr = synthetic_tone_audio
    gen = SpectrogramGenerator(n_fft=4096, hop_length=1024, min_freq_hz=50.0, max_freq_hz=300.0)
    spec_data = gen.generate_from_samples(y, sr)

    times = np.array([1.0, 2.0, 3.0, 4.0])
    freqs_on_tone = np.full_like(times, 130.0)
    freqs_off_tone = np.full_like(times, 200.0)

    intensity_on = spec_data.sample_curve_intensity(times, freqs_on_tone, use_db=False)
    intensity_off = spec_data.sample_curve_intensity(times, freqs_off_tone, use_db=False)

    assert np.all(intensity_on > intensity_off * 10.0)


def test_measurement_interface_json_schema():
    """Verify loading and parsing the blueprint's defined JSON configuration."""
    raw_json = """
    {
      "events": [
        {
          "event_id": 1,
          "cpa_time_in_audio": 15.3, 
          "f0_baseline_hz": 130.5,
          "est_velocity_range_m_s": [85, 115],
          "est_slant_distance_range_m": [2500, 4000]
        },
        {
          "event_id": 2,
          "cpa_time_in_audio": 935.3, 
          "f0_baseline_hz": 110.2,
          "est_velocity_range_m_s": [90, 110],
          "est_slant_distance_range_m": [1500, 3000]
        }
      ]
    }
    """
    config = MeasurementConfig.from_json(raw_json)
    assert len(config.events) == 2
    assert config.events[0].event_id == 1
    assert config.events[0].cpa_time_in_audio == 15.3
    assert config.events[0].f0_baseline_hz == 130.5
    assert config.events[0].est_velocity_range_m_s == (85.0, 115.0)

    # Test time offset between events
    dt = config.get_time_offset_between_events(0, 1)
    assert np.isclose(dt, 920.0)


def test_measurement_interface_validation_errors():
    """Verify that invalid event measurements raise ValueError."""
    with pytest.raises(ValueError, match="cpa_time_in_audio"):
        AudioEventMeasurement(
            event_id=1,
            cpa_time_in_audio=-5.0,
            f0_baseline_hz=100.0,
            est_velocity_range_m_s=(80.0, 100.0),
            est_slant_distance_range_m=(1000.0, 2000.0),
        )

    with pytest.raises(ValueError, match="Invalid velocity range"):
        AudioEventMeasurement(
            event_id=1,
            cpa_time_in_audio=10.0,
            f0_baseline_hz=100.0,
            est_velocity_range_m_s=(120.0, 80.0),
            est_slant_distance_range_m=(1000.0, 2000.0),
        )


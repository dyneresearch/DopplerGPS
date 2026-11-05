"""Audio Diagnostic & Spectral Inspector.

Provides high-resolution acoustic visualization, harmonic peak detection,
and diagnostic reporting to assist analysts in extracting F0 baseline frequencies
and CPA timestamps from real-world outdoor audio recordings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


@dataclass
class AudioDiagnostics:
    """Diagnostic profile of an acoustic recording."""
    audio_path: str
    duration_s: float
    sample_rate: int
    num_channels: int
    rms_db: float
    detected_peaks_hz: List[Tuple[float, float]] = field(default_factory=list)
    spectrogram_plot_path: Optional[str] = None

    def summary(self) -> str:
        peaks_str = ", ".join([f"{f:.1f} Hz ({p:.1f} dB)" for f, p in self.detected_peaks_hz[:5]]) or "None detected"
        return (
            f"Audio Diagnostics for: {os.path.basename(self.audio_path)}\n"
            f"  Duration:         {self.duration_s:.2f} s ({self.duration_s / 60.0:.2f} min)\n"
            f"  Sample Rate:      {self.sample_rate} Hz\n"
            f"  Channels:         {self.num_channels}\n"
            f"  RMS Level:        {self.rms_db:.1f} dBFS\n"
            f"  Candidate Peaks:  {peaks_str}\n"
            f"  Diagnostic Plot:  {self.spectrogram_plot_path or 'Not generated'}"
        )


def inspect_audio(
    audio_path: str,
    output_image_path: Optional[str] = None,
    min_freq_hz: float = 50.0,
    max_freq_hz: float = 400.0,
    n_fft: int = 8192,
    hop_length: int = 512,
    top_k_peaks: int = 5,
) -> AudioDiagnostics:
    """Inspect an audio recording and generate diagnostic spectral visualizations.

    Args:
        audio_path: Path to .wav, .mp3, .m4a, or .flac file.
        output_image_path: Destination path for diagnostic plot PNG.
        min_freq_hz: Lower bound of frequency band of interest (Hz).
        max_freq_hz: Upper bound of frequency band of interest (Hz).
        n_fft: FFT window size for high frequency resolution.
        hop_length: Step size between successive FFT windows.
        top_k_peaks: Maximum number of prominent tonal peaks to extract.

    Returns:
        AudioDiagnostics instance with extracted metrics and peak frequencies.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Inspect file metadata
    try:
        info = sf.info(audio_path)
        sr_native = info.samplerate
        channels = info.channels
        duration = info.duration
    except Exception:
        # Fallback to librosa header inspection
        duration = librosa.get_duration(path=audio_path)
        sr_native = 22050
        channels = 1

    # Load audio (downmix to mono for analysis)
    y, sr = librosa.load(audio_path, sr=sr_native, mono=True)
    if duration <= 0:
        duration = float(len(y)) / float(sr)

    # Compute RMS energy level
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    rms_db = float(20.0 * np.log10(rms))

    # Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
    magnitudes = np.abs(D)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)

    # Filter to requested frequency range
    freq_mask = (frequencies >= min_freq_hz) & (frequencies <= max_freq_hz)
    sub_freqs = frequencies[freq_mask]
    sub_mag = magnitudes[freq_mask, :]

    # Convert to log scale (dB)
    spec_db = librosa.amplitude_to_db(sub_mag, ref=np.max)

    # Compute time-averaged power spectrum across the band
    avg_spectrum_db = np.mean(spec_db, axis=1)

    # Peak detection on time-averaged spectrum
    # Minimum peak distance: ~10 Hz in bins
    freq_resolution = sr / n_fft
    min_dist_bins = max(1, int(10.0 / freq_resolution))
    peak_indices, properties = find_peaks(avg_spectrum_db, distance=min_dist_bins, prominence=1.5)

    detected_peaks: List[Tuple[float, float]] = []
    if len(peak_indices) > 0:
        peak_freqs = sub_freqs[peak_indices]
        peak_powers = avg_spectrum_db[peak_indices]
        # Sort by prominence/power descending
        sorted_order = np.argsort(peak_powers)[::-1][:top_k_peaks]
        for idx in sorted_order:
            detected_peaks.append((float(peak_freqs[idx]), float(peak_powers[idx])))

    # Generate diagnostic plot if requested
    if output_image_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
        fig, (ax_spec, ax_pow) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            dpi=180,
        )

        # Plot 1: High-Resolution Spectrogram
        extent = [times[0], times[-1], sub_freqs[0], sub_freqs[-1]]
        im = ax_spec.imshow(
            spec_db,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="inferno",
            vmin=-50,
            vmax=0,
        )
        ax_spec.set_title(
            f"DopplerGPS Audio Diagnostic Spectrogram: {os.path.basename(audio_path)}\n"
            f"Band: {min_freq_hz:.0f}–{max_freq_hz:.0f} Hz | Window N_fft={n_fft} | SR={sr} Hz",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax_spec.set_ylabel("Frequency (Hz)", fontsize=11)
        ax_spec.grid(True, linestyle="--", alpha=0.3, color="white")

        # Overlay detected candidate peak frequencies as horizontal lines
        for pf, pp in detected_peaks[:3]:
            ax_spec.axhline(pf, color="cyan", linestyle=":", alpha=0.7, label=f"Peak {pf:.1f} Hz")
        if detected_peaks:
            ax_spec.legend(loc="upper right", framealpha=0.8, fontsize=9)

        cbar = plt.colorbar(im, ax=ax_spec, pad=0.01)
        cbar.set_label("Relative Amplitude (dB)", fontsize=10)

        # Plot 2: Average Spectrum & Peaks
        ax_pow.plot(sub_freqs, avg_spectrum_db, color="#1f77b4", linewidth=1.2, label="Mean Spectral Density")
        for pf, pp in detected_peaks:
            ax_pow.scatter(pf, pp, color="red", s=35, zorder=5)
            ax_pow.annotate(
                f"{pf:.1f} Hz",
                (pf, pp),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                weight="bold",
                color="darkred",
            )
        ax_pow.set_xlabel("Time (s) [Top Plot] / Frequency (Hz) [Bottom Plot]", fontsize=11)
        ax_pow.set_ylabel("Power (dB)", fontsize=10)
        ax_pow.set_xlim(min_freq_hz, max_freq_hz)
        ax_pow.grid(True, linestyle=":", alpha=0.6)
        ax_pow.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_image_path, bbox_inches="tight")
        plt.close(fig)

    return AudioDiagnostics(
        audio_path=audio_path,
        duration_s=duration,
        sample_rate=sr,
        num_channels=channels,
        rms_db=rms_db,
        detected_peaks_hz=detected_peaks,
        spectrogram_plot_path=output_image_path,
    )


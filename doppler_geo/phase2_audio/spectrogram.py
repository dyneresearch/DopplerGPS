"""Audio Feature Extraction: Spectrogram Generator.

Uses librosa.load and librosa.stft to generate high-resolution spectrogram matrices
with extreme frequency resolution suitable for resolving subtle aircraft Doppler curves.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import librosa
from scipy.interpolate import RegularGridInterpolator


@dataclass
class SpectrogramData:
    """Container for the computed STFT spectrogram matrix and its coordinate axes."""
    magnitude: np.ndarray      # Shape (n_freqs, n_times)
    db: np.ndarray             # Shape (n_freqs, n_times) in decibels
    times: np.ndarray          # Time in seconds for each column
    frequencies: np.ndarray    # Frequency in Hz for each row
    sr: int                    # Sampling rate
    hop_length: int            # Hop length used
    n_fft: int                 # FFT size used

    def create_interpolator(self, use_db: bool = True) -> RegularGridInterpolator:
        """Create a 2D continuous interpolator (frequency_hz, time_s) -> intensity."""
        values = self.db if use_db else self.magnitude
        # Note: RegularGridInterpolator expects (axis0, axis1) = (frequencies, times)
        return RegularGridInterpolator(
            (self.frequencies, self.times),
            values,
            bounds_error=False,
            fill_value=-120.0 if use_db else 0.0,
            method="linear",
        )

    def sample_curve_intensity(
        self,
        time_points: np.ndarray,
        frequency_points: np.ndarray,
        interpolator: Optional[RegularGridInterpolator] = None,
        use_db: bool = True,
    ) -> np.ndarray:
        """Sample spectrogram intensity values along arbitrary (time, frequency) coordinates."""
        interp = interpolator or self.create_interpolator(use_db=use_db)
        pts = np.column_stack((frequency_points, time_points))
        return interp(pts)


class SpectrogramGenerator:
    """Configurable STFT spectrogram generator designed for narrow-band propeller signals."""

    def __init__(
        self,
        n_fft: int = 16384,
        hop_length: int = 2048,
        min_freq_hz: Optional[float] = 40.0,
        max_freq_hz: Optional[float] = 500.0,
        target_sr: Optional[int] = None,
        normalize_background: bool = False,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.target_sr = target_sr
        self.normalize_background = normalize_background

    def generate_from_audio(
        self,
        audio_path: str,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> SpectrogramData:
        """Load audio file and generate high-resolution spectrogram."""
        y, sr = librosa.load(
            audio_path,
            sr=self.target_sr,
            mono=True,
            offset=offset,
            duration=duration,
        )
        return self.generate_from_samples(y, sr, time_offset=offset)

    def generate_from_samples(
        self,
        y: np.ndarray,
        sr: int,
        time_offset: float = 0.0,
    ) -> SpectrogramData:
        """Generate high-resolution STFT spectrogram from raw numpy audio samples."""
        # STFT
        stft_matrix = librosa.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window="hann",
            center=True,
        )
        magnitude = np.abs(stft_matrix)
        db = librosa.amplitude_to_db(magnitude, ref=np.max)

        # Spectral whitening / background noise floor subtraction
        if self.normalize_background and db.shape[1] > 1:
            median_floor = np.median(db, axis=1, keepdims=True)
            db = db - median_floor
            # Update magnitude to reflect normalized contrast
            magnitude = np.maximum(db, 0.0)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        times = librosa.frames_to_time(
            np.arange(stft_matrix.shape[1]),
            sr=sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        ) + time_offset

        # Filter frequency band if requested
        if self.min_freq_hz is not None or self.max_freq_hz is not None:
            f_min = self.min_freq_hz if self.min_freq_hz is not None else 0.0
            f_max = self.max_freq_hz if self.max_freq_hz is not None else freqs[-1]
            mask = (freqs >= f_min) & (freqs <= f_max)
            freqs = freqs[mask]
            magnitude = magnitude[mask, :]
            db = db[mask, :]

        return SpectrogramData(
            magnitude=magnitude,
            db=db,
            times=times,
            frequencies=freqs,
            sr=sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )


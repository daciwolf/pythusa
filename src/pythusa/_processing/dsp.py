from __future__ import annotations

from typing import Callable

import numpy as np


DSP_KERNEL_NAMES = (
    "passthrough",
    "gain",
    "window",
    "fir32",
    "fir128",
    "rfft",
    "power_spectrum",
    "stft",
)


def normalized_hann(length: int, dtype: np.dtype) -> np.ndarray:
    return np.hanning(length).astype(dtype)


def normalized_fir_taps(length: int, dtype: np.dtype) -> np.ndarray:
    taps = np.hanning(length).astype(dtype)
    taps /= taps.sum()
    return taps


def passthrough(frame: np.ndarray, out: np.ndarray) -> np.ndarray:
    np.copyto(out, frame)
    return out


def gain(frame: np.ndarray, out: np.ndarray, scale: float = 0.5) -> np.ndarray:
    np.multiply(frame, np.asarray(scale, dtype=out.dtype), out=out)
    return out


def window(frame: np.ndarray, out: np.ndarray, window_2d: np.ndarray) -> np.ndarray:
    np.multiply(frame, window_2d, out=out)
    return out


def fir_same_direct(frame: np.ndarray, out: np.ndarray, taps: np.ndarray) -> np.ndarray:
    for channel_index in range(frame.shape[1]):
        out[:, channel_index] = np.convolve(frame[:, channel_index], taps, mode="same")
    return out


def fir_same_fft(frame: np.ndarray, out: np.ndarray, taps: np.ndarray) -> np.ndarray:
    rows = frame.shape[0]
    fft_size = 1 << (rows + taps.size - 2).bit_length()
    spectrum_dtype = np.result_type(frame.dtype, np.complex64)
    tap_spectrum = np.fft.rfft(taps, n=fft_size).astype(spectrum_dtype, copy=False)
    same_start = (taps.size - 1) // 2
    spectrum = np.fft.rfft(frame, n=fft_size, axis=0)
    spectrum *= tap_spectrum[:, None]
    filtered = np.fft.irfft(spectrum, n=fft_size, axis=0)
    np.copyto(out, filtered[same_start:same_start + rows, :])
    return out


def rfft_spectrum(frame: np.ndarray) -> np.ndarray:
    return np.fft.rfft(frame, axis=0)


def power_spectrum(frame: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(frame, axis=0)
    return np.square(np.abs(spectrum))


def stft_spectrum(frame: np.ndarray, scratch: np.ndarray, window_2d: np.ndarray) -> np.ndarray:
    np.multiply(frame, window_2d, out=scratch)
    return np.fft.rfft(scratch, axis=0)


def make_benchmark_processor(
    kernel_name: str,
    *,
    rows: int,
    channels: int,
    dtype: np.dtype,
) -> Callable[[np.ndarray], float]:
    scratch = np.empty((rows, channels), dtype=dtype)
    window_1d = normalized_hann(rows, dtype)
    window_2d = window_1d[:, None]

    if kernel_name == "passthrough":

        def _process(frame: np.ndarray) -> float:
            return float(passthrough(frame, scratch)[0, 0])

    elif kernel_name == "gain":

        def _process(frame: np.ndarray) -> float:
            return float(gain(frame, scratch)[0, 0])

    elif kernel_name == "window":

        def _process(frame: np.ndarray) -> float:
            return float(window(frame, scratch, window_2d)[0, 0])

    elif kernel_name == "fir32":
        taps = normalized_fir_taps(32, dtype)

        def _process(frame: np.ndarray) -> float:
            return float(fir_same_direct(frame, scratch, taps)[0, 0])

    elif kernel_name == "fir128":
        taps = normalized_fir_taps(128, dtype)

        def _process(frame: np.ndarray) -> float:
            return float(fir_same_fft(frame, scratch, taps)[0, 0])

    elif kernel_name == "rfft":

        def _process(frame: np.ndarray) -> float:
            return float(np.abs(rfft_spectrum(frame)).max())

    elif kernel_name == "power_spectrum":

        def _process(frame: np.ndarray) -> float:
            return float(power_spectrum(frame).max())

    elif kernel_name == "stft":

        def _process(frame: np.ndarray) -> float:
            return float(np.abs(stft_spectrum(frame, scratch, window_2d)).max())

    else:  # pragma: no cover - defensive for benchmark env overrides.
        raise ValueError(f"unknown kernel '{kernel_name}'")

    return _process


def validate_kernel_name(kernel_name: str) -> None:
    if kernel_name not in DSP_KERNEL_NAMES:
        raise ValueError(f"unknown kernel '{kernel_name}'")

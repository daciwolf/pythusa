"""Run with: pytest tests/test_dsp_kernels.py -q"""

from __future__ import annotations

import unittest

import numpy as np

from pythusa._processing.dsp import (
    DSP_KERNEL_NAMES,
    fir_same_direct,
    fir_same_fft,
    gain,
    make_benchmark_processor,
    normalized_fir_taps,
    normalized_hann,
    passthrough,
    power_spectrum,
    rfft_spectrum,
    stft_spectrum,
    window,
)


def _make_frame(rows: int, channels: int, dtype: np.dtype) -> np.ndarray:
    base = np.linspace(0.0, 2.0 * np.pi, rows, endpoint=False, dtype=dtype)
    frame = np.empty((rows, channels), dtype=dtype)
    for channel_index in range(channels):
        freq = dtype.type(channel_index + 1)
        frame[:, channel_index] = (
            np.sin(base * freq)
            + np.cos(base * (freq + dtype.type(0.5)))
            + dtype.type(0.1 * (channel_index + 1))
        )
    return frame


class DSPKernelCorrectnessTests(unittest.TestCase):
    def test_passthrough_matches_input(self) -> None:
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                frame = _make_frame(64, 2, np.dtype(dtype))
                out = np.empty_like(frame)
                result = passthrough(frame, out)
                np.testing.assert_array_equal(result, frame)

    def test_gain_scales_by_half(self) -> None:
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                frame = _make_frame(64, 2, np.dtype(dtype))
                out = np.empty_like(frame)
                result = gain(frame, out)
                expected = frame * np.asarray(0.5, dtype=frame.dtype)
                np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)

    def test_window_matches_hann_weighting(self) -> None:
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                frame = _make_frame(64, 2, np.dtype(dtype))
                out = np.empty_like(frame)
                window_2d = normalized_hann(frame.shape[0], frame.dtype)[:, None]
                result = window(frame, out, window_2d)
                expected = frame * window_2d
                np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)

    def test_fir32_matches_direct_convolution_reference(self) -> None:
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                frame = _make_frame(256, 2, np.dtype(dtype))
                out = np.empty_like(frame)
                taps = normalized_fir_taps(32, frame.dtype)
                result = fir_same_direct(frame, out, taps)
                expected = np.empty_like(frame)
                for channel_index in range(frame.shape[1]):
                    expected[:, channel_index] = np.convolve(frame[:, channel_index], taps, mode="same")
                np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)

    def test_fir128_fft_matches_direct_convolution_reference(self) -> None:
        tolerances = {
            np.dtype(np.float32): dict(rtol=2e-4, atol=2e-5),
            np.dtype(np.float64): dict(rtol=1e-10, atol=1e-12),
        }
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                frame = _make_frame(256, 2, np.dtype(dtype))
                out = np.empty_like(frame)
                taps = normalized_fir_taps(128, frame.dtype)
                result = fir_same_fft(frame, out, taps)
                expected = np.empty_like(frame)
                for channel_index in range(frame.shape[1]):
                    expected[:, channel_index] = np.convolve(frame[:, channel_index], taps, mode="same")
                np.testing.assert_allclose(result, expected, **tolerances[np.dtype(dtype)])

    def test_rfft_matches_numpy_reference(self) -> None:
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                frame = _make_frame(128, 2, np.dtype(dtype))
                result = rfft_spectrum(frame)
                expected = np.fft.rfft(frame, axis=0)
                np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_power_spectrum_matches_numpy_reference(self) -> None:
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                frame = _make_frame(128, 2, np.dtype(dtype))
                result = power_spectrum(frame)
                expected = np.square(np.abs(np.fft.rfft(frame, axis=0)))
                np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)

    def test_stft_matches_windowed_rfft_reference(self) -> None:
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                frame = _make_frame(128, 2, np.dtype(dtype))
                scratch = np.empty_like(frame)
                window_2d = normalized_hann(frame.shape[0], frame.dtype)[:, None]
                result = stft_spectrum(frame, scratch, window_2d)
                expected = np.fft.rfft(frame * window_2d, axis=0)
                np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_benchmark_processors_emit_expected_tokens(self) -> None:
        rows = 256
        channels = 2
        for dtype in (np.float32, np.float64):
            frame = _make_frame(rows, channels, np.dtype(dtype))
            window_2d = normalized_hann(rows, frame.dtype)[:, None]
            fir32_taps = normalized_fir_taps(32, frame.dtype)
            fir128_taps = normalized_fir_taps(128, frame.dtype)
            expectations = {
                "passthrough": lambda: float(frame[0, 0]),
                "gain": lambda: float((frame * np.asarray(0.5, dtype=frame.dtype))[0, 0]),
                "window": lambda: float((frame * window_2d)[0, 0]),
                "fir32": lambda: float(np.convolve(frame[:, 0], fir32_taps, mode="same")[0]),
                "fir128": lambda: float(np.convolve(frame[:, 0], fir128_taps, mode="same")[0]),
                "rfft": lambda: float(np.abs(np.fft.rfft(frame, axis=0)).max()),
                "power_spectrum": lambda: float(np.square(np.abs(np.fft.rfft(frame, axis=0))).max()),
                "stft": lambda: float(np.abs(np.fft.rfft(frame * window_2d, axis=0)).max()),
            }
            for kernel_name in DSP_KERNEL_NAMES:
                with self.subTest(dtype=np.dtype(dtype).name, kernel=kernel_name):
                    processor = make_benchmark_processor(
                        kernel_name,
                        rows=rows,
                        channels=channels,
                        dtype=np.dtype(dtype),
                    )
                    self.assertAlmostEqual(processor(frame), expectations[kernel_name](), places=6)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from typing import Sequence

import numpy as np


def graph_display_series(
    series: np.ndarray | Sequence[np.ndarray],
    *,
    title: str | None = None,
    size: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    color: tuple[float, float, float, float] = (0.20, 0.80, 1.00, 1.00),
    overlay_color: tuple[float, float, float, float] = (1.00, 0.45, 0.20, 0.80),
    background_color: tuple[float, float, float, float] = (0.07, 0.08, 0.09, 1.00),
    border_color: tuple[float, float, float, float] = (0.32, 0.34, 0.36, 1.00),
    thickness: float = 1.5,
    overlay_thickness: float | None = None,
    show_axes: bool = True,
) -> None:
    imgui = _require_imgui()

    if isinstance(series, np.ndarray) and series.ndim == 2:
        frames = [series]
    else:
        frames = list(series)

    if not frames:
        raise ValueError("series must contain at least one frame")

    if overlay_thickness is None:
        overlay_thickness = max(1.0, thickness * 0.65)

    chunks: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        data = np.asarray(frame)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"each frame must have shape (n, 3), got {data.shape}")
        if data.shape[0] == 0:
            continue
        chunks.append(data)

    if not chunks:
        raise ValueError("series must contain at least one non-empty frame")

    data = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    n = data.shape[0]
    timestamps = np.arange(n, dtype=np.float32)
    values = np.asarray(data[:, 1], dtype=np.float32)
    overlay_values = np.asarray(data[:, 2], dtype=np.float32)

    x_min = 0.0
    x_max = float(max(n - 1, 1))
    combined_values = np.concatenate((values, overlay_values))
    y_min = float(np.min(combined_values))
    y_max = float(np.max(combined_values))

    if size is None:
        avail = imgui.get_content_region_available()
        width = float(avail[0] if isinstance(avail, tuple) else avail.x)
        height = float(avail[1] if isinstance(avail, tuple) else avail.y)
        if width <= 0.0:
            width = 1200.0
        if height <= 0.0:
            height = 650.0
    else:
        width = float(size[0])
        height = float(size[1])

    if title is not None:
        imgui.text_unformatted(title)

    draw_pos = imgui.get_cursor_screen_pos()
    imgui.dummy(width, height)

    draw_list = imgui.get_window_draw_list()
    x0, y0 = _xy(draw_pos)
    x1 = x0 + width
    y1 = y0 + height

    bg = _rgba_u32(imgui, background_color)
    border = _rgba_u32(imgui, border_color)
    line_color = _rgba_u32(imgui, color)
    overlay_line_color = _rgba_u32(imgui, overlay_color)

    draw_list.add_rect_filled(x0, y0, x1, y1, bg)
    draw_list.add_rect(x0, y0, x1, y1, border)

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min = 0.0
        x_max = 1.0
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min = -1.0
        y_max = 1.0
    if y_limits is not None:
        stable_y_min, stable_y_max = y_limits
        if np.isfinite(stable_y_min) and np.isfinite(stable_y_max) and stable_y_max > stable_y_min:
            y_min = float(stable_y_min)
            y_max = float(stable_y_max)

    inner_left = x0 + 3.0
    inner_top = y0 + 3.0
    inner_right = x1 - 3.0
    inner_bottom = y1 - 3.0
    inner_width = max(1.0, inner_right - inner_left)
    inner_height = max(1.0, inner_bottom - inner_top)

    x_scale = inner_width / (x_max - x_min)
    y_scale = inner_height / (y_max - y_min)

    if show_axes and y_min <= 0.0 <= y_max:
        zero_y = inner_bottom - ((0.0 - y_min) * y_scale)
        draw_list.add_line(inner_left, zero_y, inner_right, zero_y, border, 1.0)

    overlay_points = _map_segment_to_screen(
        timestamps,
        overlay_values,
        x_min=x_min,
        y_min=y_min,
        x_scale=x_scale,
        y_scale=y_scale,
        inner_left=inner_left,
        inner_bottom=inner_bottom,
    )
    main_points = _map_segment_to_screen(
        timestamps,
        values,
        x_min=x_min,
        y_min=y_min,
        x_scale=x_scale,
        y_scale=y_scale,
        inner_left=inner_left,
        inner_bottom=inner_bottom,
    )
    if overlay_points.shape[0] >= 2:
        _draw_polyline(draw_list, overlay_points, overlay_line_color, overlay_thickness)
    if main_points.shape[0] >= 2:
        _draw_polyline(draw_list, main_points, line_color, thickness)


def graph_data_arrays(
    arrays: Sequence[np.ndarray],
    *,
    overlay_arrays: Sequence[np.ndarray] | None = None,
    title: str | None = None,
    size: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    color: tuple[float, float, float, float] = (0.20, 0.80, 1.00, 1.00),
    overlay_color: tuple[float, float, float, float] = (1.00, 0.45, 0.20, 1.00),
    background_color: tuple[float, float, float, float] = (0.07, 0.08, 0.09, 1.00),
    border_color: tuple[float, float, float, float] = (0.32, 0.34, 0.36, 1.00),
    thickness: float = 1.5,
    overlay_thickness: float | None = None,
    show_axes: bool = True,
) -> None:
    imgui = _require_imgui()

    if not arrays:
        raise ValueError("arrays must contain at least one array")

    segments, x_min, x_max, y_min, y_max = _prepare_segments(arrays)
    overlay_segments: list[tuple[np.ndarray, np.ndarray]] = []
    if overlay_arrays:
        overlay_segments, ox_min, ox_max, oy_min, oy_max = _prepare_segments(overlay_arrays)
        x_min = min(x_min, ox_min)
        x_max = max(x_max, ox_max)
        y_min = min(y_min, oy_min)
        y_max = max(y_max, oy_max)

    if size is None:
        avail = imgui.get_content_region_available()
        width = float(avail[0] if isinstance(avail, tuple) else avail.x)
        height = float(avail[1] if isinstance(avail, tuple) else avail.y)
        if width <= 0.0:
            width = 1200.0
        if height <= 0.0:
            height = 650.0
    else:
        width = float(size[0])
        height = float(size[1])

    if title is not None:
        imgui.text_unformatted(title)

    draw_pos = imgui.get_cursor_screen_pos()
    imgui.dummy(width, height)

    draw_list = imgui.get_window_draw_list()
    x0, y0 = _xy(draw_pos)
    x1 = x0 + width
    y1 = y0 + height

    bg = _rgba_u32(imgui, background_color)
    border = _rgba_u32(imgui, border_color)
    line_color = _rgba_u32(imgui, color)
    overlay_line_color = _rgba_u32(imgui, overlay_color)
    if overlay_thickness is None:
        overlay_thickness = max(1.0, thickness * 0.65)

    draw_list.add_rect_filled(x0, y0, x1, y1, bg)
    draw_list.add_rect(x0, y0, x1, y1, border)

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min = 0.0
        x_max = 1.0
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min = -1.0
        y_max = 1.0
    if y_limits is not None:
        stable_y_min, stable_y_max = y_limits
        if np.isfinite(stable_y_min) and np.isfinite(stable_y_max) and stable_y_max > stable_y_min:
            y_min = float(stable_y_min)
            y_max = float(stable_y_max)

    inner_left = x0 + 3.0
    inner_top = y0 + 3.0
    inner_right = x1 - 3.0
    inner_bottom = y1 - 3.0
    inner_width = max(1.0, inner_right - inner_left)
    inner_height = max(1.0, inner_bottom - inner_top)

    x_scale = inner_width / (x_max - x_min)
    y_scale = inner_height / (y_max - y_min)

    if show_axes and y_min <= 0.0 <= y_max:
        zero_y = inner_bottom - ((0.0 - y_min) * y_scale)
        draw_list.add_line(inner_left, zero_y, inner_right, zero_y, border, 1.0)

    _draw_segments(
        draw_list,
        overlay_segments,
        color=overlay_line_color,
        thickness=overlay_thickness,
        x_min=x_min,
        y_min=y_min,
        x_scale=x_scale,
        y_scale=y_scale,
        inner_left=inner_left,
        inner_bottom=inner_bottom,
    )
    _draw_segments(
        draw_list,
        segments,
        color=line_color,
        thickness=thickness,
        x_min=x_min,
        y_min=y_min,
        x_scale=x_scale,
        y_scale=y_scale,
        inner_left=inner_left,
        inner_bottom=inner_bottom,
    )


def _prepare_segments(
    arrays: Sequence[np.ndarray],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], float, float, float, float]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    previous_end = None
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf

    for index, array in enumerate(arrays):
        data = np.asarray(array)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"arrays[{index}] must have shape (n, 2), got {data.shape}")

        timestamps = np.asarray(data[:, 0], dtype=np.float64)
        values = np.asarray(data[:, 1], dtype=np.float64)
        if timestamps.size == 0:
            continue

        if previous_end is not None:
            start = float(timestamps[0])
            if timestamps.size > 1:
                step = float(np.median(np.diff(timestamps)))
                if not np.isfinite(step) or step <= 0.0:
                    step = 1.0
            else:
                step = 1.0
            timestamps = timestamps + (previous_end + step - start)

        previous_end = float(timestamps[-1])
        segments.append((timestamps, values))

        x_min = min(x_min, float(timestamps[0]))
        x_max = max(x_max, float(timestamps[-1]))
        y_min = min(y_min, float(np.min(values)))
        y_max = max(y_max, float(np.max(values)))

    if not segments:
        raise ValueError("arrays must contain at least one non-empty array")

    return segments, x_min, x_max, y_min, y_max


def _map_segment_to_screen(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    x_min: float,
    y_min: float,
    x_scale: float,
    y_scale: float,
    inner_left: float,
    inner_bottom: float,
) -> np.ndarray:
    points = np.empty((timestamps.size, 2), dtype=np.float32)
    points[:, 0] = inner_left + (timestamps - x_min) * x_scale
    points[:, 1] = inner_bottom - (values - y_min) * y_scale
    return points


def _draw_polyline(draw_list, points: np.ndarray, color: int, thickness: float) -> None:
    polyline = getattr(draw_list, "add_polyline", None)
    if polyline is not None:
        polyline(points.tolist(), color, False, thickness)
        return
    for start, end in zip(points[:-1], points[1:]):
        draw_list.add_line(
            float(start[0]),
            float(start[1]),
            float(end[0]),
            float(end[1]),
            color,
            thickness,
        )


def _draw_segments(
    draw_list,
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    color: int,
    thickness: float,
    x_min: float,
    y_min: float,
    x_scale: float,
    y_scale: float,
    inner_left: float,
    inner_bottom: float,
) -> None:
    for timestamps, values in segments:
        points = _map_segment_to_screen(
            timestamps,
            values,
            x_min=x_min,
            y_min=y_min,
            x_scale=x_scale,
            y_scale=y_scale,
            inner_left=inner_left,
            inner_bottom=inner_bottom,
        )
        if points.shape[0] >= 2:
            _draw_polyline(draw_list, points, color, thickness)


def _rgba_u32(imgui, rgba: tuple[float, float, float, float]) -> int:
    converter = getattr(imgui, "get_color_u32_rgba", None)
    if converter is not None:
        return converter(*rgba)
    r, g, b, a = (max(0, min(255, int(channel * 255.0))) for channel in rgba)
    return (a << 24) | (b << 16) | (g << 8) | r


def _xy(pos) -> tuple[float, float]:
    if isinstance(pos, tuple):
        return float(pos[0]), float(pos[1])
    return float(pos.x), float(pos.y)


def _require_imgui():
    try:
        import imgui
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "imgui is required for graphing. Install a Dear ImGui Python binding such as 'imgui'."
        ) from exc
    return imgui

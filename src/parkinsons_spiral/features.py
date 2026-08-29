"""Feature extraction for digitized spiral trajectories.

The expected input format is the seven-column, semicolon-delimited format used by
the UCI ParkinsonHW dataset: x, y, z, pressure, grip angle, timestamp, test id.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = ("x", "y", "z", "pressure", "grip_angle", "timestamp", "test_id")
TEST_NAMES = {0: "static", 1: "dynamic"}
EPSILON = 1e-9


def read_trajectory(path: str | Path) -> pd.DataFrame:
    """Read and validate one participant's UCI-style trajectory file."""
    frame = pd.read_csv(path, sep=";", names=COLUMNS, header=None)
    for column in COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["x", "y", "timestamp", "test_id"]).copy()
    if frame.empty:
        raise ValueError(f"No valid trajectory rows found in {path}")
    frame["test_id"] = frame["test_id"].astype(int)
    return frame


def _safe_cv(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite) / (abs(np.mean(finite)) + EPSILON))


def _rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))


def _tremor_power_ratio(
    timestamps_s: np.ndarray, radial_residual: np.ndarray
) -> float:
    """Return the fraction of residual radial power in the 4-7 Hz band.

    This is a research feature, not a clinical tremor measurement. The signal is
    interpolated to a regular grid before applying a Hann-windowed FFT.
    """
    if timestamps_s.size < 32:
        return float("nan")
    unique_t, unique_indices = np.unique(timestamps_s, return_index=True)
    if unique_t.size < 32 or unique_t[-1] - unique_t[0] < 0.5:
        return float("nan")

    median_dt = float(np.median(np.diff(unique_t)))
    if not np.isfinite(median_dt) or median_dt <= 0:
        return float("nan")
    sample_rate = float(np.clip(1.0 / median_dt, 20.0, 200.0))
    regular_t = np.arange(unique_t[0], unique_t[-1], 1.0 / sample_rate)
    if regular_t.size < 32:
        return float("nan")

    signal = np.interp(regular_t, unique_t, radial_residual[unique_indices])
    signal = signal - np.mean(signal)
    signal = signal * np.hanning(signal.size)
    power = np.abs(np.fft.rfft(signal)) ** 2
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    usable = (frequencies >= 1.0) & (frequencies <= 15.0)
    tremor = (frequencies >= 4.0) & (frequencies <= 7.0)
    denominator = float(np.sum(power[usable]))
    if denominator <= EPSILON:
        return 0.0
    return float(np.sum(power[tremor]) / denominator)


def extract_test_features(frame: pd.DataFrame, test_id: int) -> dict[str, float]:
    """Extract scale-normalized spatial, kinematic, and pressure features."""
    part = frame.loc[frame["test_id"] == test_id].copy()
    if len(part) < 10:
        return {}

    x = part["x"].to_numpy(dtype=float)
    y = part["y"].to_numpy(dtype=float)
    timestamps_s = part["timestamp"].to_numpy(dtype=float) / 1000.0
    timestamps_s = timestamps_s - timestamps_s[0]

    dx = np.diff(x)
    dy = np.diff(y)
    step = np.hypot(dx, dy)
    dt = np.diff(timestamps_s)
    valid_dt = (dt > 0) & np.isfinite(dt) & (dt < 2.0)
    speed = np.divide(
        step,
        dt,
        out=np.full_like(step, np.nan, dtype=float),
        where=valid_dt,
    )

    extent = float(np.hypot(np.ptp(x), np.ptp(y)))
    scale = max(extent, EPSILON)
    duration_s = float(max(timestamps_s[-1] - timestamps_s[0], 0.0))

    valid_speed = speed[np.isfinite(speed)]
    accel = np.array([], dtype=float)
    if valid_speed.size > 2:
        # The device has an approximately constant sampling interval. Using its
        # median makes this robust to the occasional duplicate timestamp.
        sample_dt = float(np.median(dt[valid_dt]))
        accel = np.diff(valid_speed) / max(sample_dt, EPSILON)

    origin_x, origin_y = x[0], y[0]
    radius = np.hypot(x - origin_x, y - origin_y)
    theta = np.unwrap(np.arctan2(y - origin_y, x - origin_x))
    dtheta = np.diff(theta)
    nonzero_turn = dtheta[np.abs(dtheta) > 1e-6]
    direction = float(np.sign(np.median(nonzero_turn))) if nonzero_turn.size else 1.0
    if direction == 0:
        direction = 1.0
    progress = direction * (theta - theta[0])

    design = np.column_stack([np.ones(progress.size), progress])
    coefficients, *_ = np.linalg.lstsq(design, radius, rcond=None)
    fitted_radius = design @ coefficients
    radial_residual = radius - fitted_radius
    radius_range = max(float(np.ptp(radius)), EPSILON)

    pressure = part["pressure"].to_numpy(dtype=float)
    grip_angle = part["grip_angle"].to_numpy(dtype=float)
    angle_motion = direction * dtheta

    return {
        "duration_s": duration_s,
        "path_length_norm": float(np.sum(step) / scale),
        "speed_mean_norm": float(np.nanmean(speed) / scale),
        "speed_cv": _safe_cv(speed),
        "accel_rms_norm": _rms(accel) / scale,
        "stationary_fraction": float(np.mean(step <= max(scale * 1e-4, 1e-6))),
        "pressure_cv": _safe_cv(pressure),
        "grip_angle_cv": _safe_cv(grip_angle),
        "turns": float(max(progress[-1], 0.0) / (2.0 * np.pi)),
        "loop_spacing_norm": float(abs(coefficients[1]) * 2.0 * np.pi / scale),
        "archimedean_rmse_norm": _rms(radial_residual) / radius_range,
        "radial_monotonicity": float(np.mean(np.diff(radius) >= 0)),
        "angle_backtrack_fraction": float(np.mean(angle_motion < 0)),
        "tremor_power_4_7_ratio": _tremor_power_ratio(
            timestamps_s, radial_residual
        ),
    }


def extract_participant_features(frame: pd.DataFrame) -> dict[str, float]:
    """Create one feature vector per participant to prevent subject leakage."""
    combined: dict[str, float] = {}
    by_test: dict[str, dict[str, float]] = {}
    for test_id, test_name in TEST_NAMES.items():
        features = extract_test_features(frame, test_id)
        by_test[test_name] = features
        combined.update({f"{test_name}__{key}": value for key, value in features.items()})

    # Relative change in the harder blinking-template task can reduce dependence
    # on a participant's absolute drawing style.
    for feature in (
        "duration_s",
        "speed_mean_norm",
        "speed_cv",
        "archimedean_rmse_norm",
        "stationary_fraction",
    ):
        if feature in by_test["static"] and feature in by_test["dynamic"]:
            static_value = by_test["static"][feature]
            dynamic_value = by_test["dynamic"][feature]
            combined[f"delta__{feature}"] = float(
                (dynamic_value - static_value) / (abs(static_value) + EPSILON)
            )
    return combined


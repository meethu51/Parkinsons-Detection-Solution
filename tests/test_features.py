from __future__ import annotations

import numpy as np
import pandas as pd

from parkinsons_spiral.features import COLUMNS, extract_participant_features


def _spiral(test_id: int, wobble: float = 0.0) -> pd.DataFrame:
    samples = 600
    theta = np.linspace(0.01, 6.0 * np.pi, samples)
    radius = np.linspace(1.0, 100.0, samples) + wobble * np.sin(20.0 * theta)
    time_ms = np.arange(samples) * 10 + test_id * 10_000
    values = np.column_stack(
        [
            200 + radius * np.cos(theta),
            200 + radius * np.sin(theta),
            np.zeros(samples),
            np.full(samples, 400),
            np.full(samples, 900),
            time_ms,
            np.full(samples, test_id),
        ]
    )
    return pd.DataFrame(values, columns=COLUMNS)


def test_extracts_both_tests_and_delta_features() -> None:
    frame = pd.concat([_spiral(0), _spiral(1, wobble=2.0)], ignore_index=True)
    features = extract_participant_features(frame)

    assert features["static__turns"] > 2.5
    assert features["dynamic__turns"] > 2.5
    assert "delta__archimedean_rmse_norm" in features
    assert features["dynamic__archimedean_rmse_norm"] > features["static__archimedean_rmse_norm"]


def test_scale_normalized_geometry_is_size_invariant() -> None:
    original = _spiral(0)
    scaled = original.copy()
    scaled[["x", "y"]] *= 3.0

    first = extract_participant_features(original)
    second = extract_participant_features(scaled)

    assert np.isclose(
        first["static__archimedean_rmse_norm"],
        second["static__archimedean_rmse_norm"],
        rtol=1e-5,
    )
    assert np.isclose(
        first["static__loop_spacing_norm"],
        second["static__loop_spacing_norm"],
        rtol=1e-5,
    )


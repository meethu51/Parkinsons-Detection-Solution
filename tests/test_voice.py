from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from parkinsons_spiral.voice import (
    VOICE_FEATURES,
    extract_voice_features,
    score_voice_features,
)


def test_live_extractor_returns_complete_feature_vector() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate * 6, dtype=float) / sample_rate
    samples = 0.2 * np.sin(2 * np.pi * 140 * time)
    features, quality = extract_voice_features(samples.astype(np.float32), sample_rate)

    assert quality["valid"] is True
    assert set(features) == set(VOICE_FEATURES)
    assert all(np.isfinite(value) for value in features.values())


def test_out_of_domain_voice_score_is_suppressed(tmp_path: Path) -> None:
    classifier = DummyClassifier(strategy="prior")
    classifier.fit(
        pd.DataFrame([[0.0] * len(VOICE_FEATURES)] * 2, columns=VOICE_FEATURES),
        [0, 1],
    )
    model_path = tmp_path / "voice-model.joblib"
    joblib.dump(
        {
            "pipeline": classifier,
            "feature_names": list(VOICE_FEATURES),
            "training_quantiles": {
                name: {"low": 0.0, "high": 0.5} for name in VOICE_FEATURES
            },
            "version": 1,
        },
        model_path,
    )

    result = score_voice_features(
        {name: 1.0 for name in VOICE_FEATURES}, model_path
    )
    assert result["domain_match"] is False
    assert result["screening_score"] is None
    assert result["pattern_signal"] == "unscorable"

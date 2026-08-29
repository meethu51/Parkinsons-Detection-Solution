from __future__ import annotations

import math
import io
import wave
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from parkinsons_spiral.app import create_app
from parkinsons_spiral.features import COLUMNS, extract_participant_features
from parkinsons_spiral.voice import VOICE_FEATURES


def _points(mode: str) -> list[dict]:
    points = []
    samples = 500
    for index in range(samples):
        theta = 0.01 + (index / (samples - 1)) * 6.0 * math.pi
        radius = 8.0 + (index / (samples - 1)) * 235.0
        if mode == "dynamic":
            radius += 1.5 * math.sin(theta * 12.0)
        points.append(
            {
                "x": 320.0 + radius * math.cos(theta),
                "y": 320.0 + radius * math.sin(theta),
                "t": index * 12.0,
                "pressure": 0.45 + 0.05 * math.sin(theta),
                "tilt_x": 5.0,
                "tilt_y": 2.0,
                "pointer_type": "pen",
                "stroke": 1,
            }
        )
    return points


def _feature_names() -> list[str]:
    frames = []
    for test_id, mode in ((0, "static"), (1, "dynamic")):
        rows = [
            (
                point["x"],
                point["y"],
                0,
                point["pressure"] * 1024,
                850,
                point["t"],
                test_id,
            )
            for point in _points(mode)
        ]
        frames.append(pd.DataFrame(rows, columns=COLUMNS))
    return list(extract_participant_features(pd.concat(frames, ignore_index=True)))


def _client(tmp_path: Path) -> TestClient:
    feature_names = _feature_names()
    classifier = DummyClassifier(strategy="prior")
    classifier.fit(pd.DataFrame([[0.0] * len(feature_names)] * 2, columns=feature_names), [0, 1])
    model_path = tmp_path / "model.joblib"
    joblib.dump(
        {"pipeline": classifier, "feature_names": feature_names, "version": 1},
        model_path,
    )
    voice_classifier = DummyClassifier(strategy="prior")
    voice_classifier.fit(
        pd.DataFrame([[0.0] * len(VOICE_FEATURES)] * 2, columns=VOICE_FEATURES), [0, 1]
    )
    voice_model_path = tmp_path / "voice-model.joblib"
    joblib.dump(
        {
            "pipeline": voice_classifier,
            "feature_names": list(VOICE_FEATURES),
            "training_quantiles": {
                name: {"low": -1_000.0, "high": 1_000.0} for name in VOICE_FEATURES
            },
            "version": 1,
        },
        voice_model_path,
    )
    return TestClient(
        create_app(tmp_path / "sessions.sqlite3", model_path, voice_model_path)
    )


def _sustained_vowel_wav() -> bytes:
    sample_rate = 16_000
    time = np.arange(sample_rate * 6, dtype=float) / sample_rate
    samples = 0.2 * np.sin(2 * np.pi * 140 * time)
    samples *= 1.0 + 0.01 * np.sin(2 * np.pi * 3 * time)
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return target.getvalue()


def _create_session(client: TestClient) -> str:
    response = client.post(
        "/api/sessions",
        json={
            "participant_code": "TEST-001",
            "handedness": "right",
            "age_band": "not_shared",
            "medication_state": "not_applicable",
            "consent_research": True,
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_complete_capture_score_export_and_delete(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json()["model_ready"] is True
    session_id = _create_session(client)

    for mode in ("static", "dynamic"):
        response = client.post(
            f"/api/sessions/{session_id}/trials",
            json={
                "mode": mode,
                "hand": "right",
                "repetition": 1,
                "canvas_width": 640,
                "canvas_height": 640,
                "points": _points(mode),
            },
        )
        assert response.status_code == 201
        assert response.json()["quality"]["valid"] is True
        assert response.json()["quality"]["sample_rate_hz"] > 80
        assert response.json()["quality"]["pressure_range"][1] > response.json()["quality"]["pressure_range"][0]

    result = client.post(f"/api/sessions/{session_id}/score")
    assert result.status_code == 200
    assert result.json()["experimental_screening_score"] == 0.5
    assert result.json()["pattern_signal"] == "lower"
    assert result.json()["decision_threshold"] == 0.75
    assert len(result.json()["pair_scores"]) == 1

    exported = client.get(f"/api/sessions/{session_id}/export")
    assert exported.status_code == 200
    assert len(exported.json()["trials"]) == 2
    assert len(exported.json()["trials"][0]["points"]) == 500

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_rejects_session_without_consent(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/api/sessions",
        json={
            "participant_code": "TEST-002",
            "handedness": "left",
            "age_band": "50_59",
            "medication_state": "unknown",
            "consent_research": False,
        },
    )
    assert response.status_code == 422


def test_requires_a_complete_static_dynamic_pair(tmp_path: Path) -> None:
    client = _client(tmp_path)
    session_id = _create_session(client)
    response = client.post(
        f"/api/sessions/{session_id}/trials",
        json={
            "mode": "static",
            "hand": "right",
            "repetition": 1,
            "canvas_width": 640,
            "canvas_height": 640,
            "points": _points("static"),
        },
    )
    assert response.status_code == 201
    scored = client.post(f"/api/sessions/{session_id}/score")
    assert scored.status_code == 422
    assert "static and dynamic spiral pair" in scored.json()["detail"]


def test_rejects_non_dominant_hand_trial(tmp_path: Path) -> None:
    client = _client(tmp_path)
    session_id = _create_session(client)
    response = client.post(
        f"/api/sessions/{session_id}/trials",
        json={
            "mode": "static",
            "hand": "left",
            "repetition": 1,
            "canvas_width": 640,
            "canvas_height": 640,
            "points": _points("static"),
        },
    )
    assert response.status_code == 422
    assert "dominant hand" in response.json()["detail"]


def test_three_voice_recordings_are_scored_without_storing_audio(tmp_path: Path) -> None:
    client = _client(tmp_path)
    session_id = _create_session(client)
    recording = _sustained_vowel_wav()
    for repetition in range(1, 4):
        response = client.post(
            f"/api/sessions/{session_id}/voice?repetition={repetition}",
            content=recording,
            headers={"Content-Type": "audio/wav"},
        )
        assert response.status_code == 201
        assert response.json()["quality"]["valid"] is True
        assert response.json()["result"]["screening_score"] == 0.5

    score = client.post(f"/api/sessions/{session_id}/voice-score")
    assert score.status_code == 200
    assert score.json()["experimental_voice_score"] == 0.5
    assert score.json()["pattern_signal"] == "lower"
    assert score.json()["decision_threshold"] == 0.75
    assert score.json()["valid_recordings"] == 3

    for mode in ("static", "dynamic"):
        response = client.post(
            f"/api/sessions/{session_id}/trials",
            json={
                "mode": mode,
                "hand": "right",
                "repetition": 1,
                "canvas_width": 640,
                "canvas_height": 640,
                "points": _points(mode),
            },
        )
        assert response.status_code == 201

    report = client.get(f"/api/sessions/{session_id}/report.pdf")
    assert report.status_code == 200
    assert report.headers["content-type"] == "application/pdf"
    assert "parkinsons-research-report" in report.headers["content-disposition"]
    assert report.content.startswith(b"%PDF-")
    assert len(report.content) > 10_000

    exported = client.get(f"/api/sessions/{session_id}/export").json()
    assert len(exported["voice_trials"]) == 3
    assert "audio" not in exported["voice_trials"][0]

"""Local tablet-capture web application and API."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .features import COLUMNS, extract_participant_features, extract_test_features
from .storage import SessionStore
from .settings import ELEVATED_SIGNAL_THRESHOLD
from .report import build_session_report
from .voice import decode_audio, extract_voice_features, score_voice_features


PACKAGE_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PACKAGE_ROOT / "web"
DEFAULT_DB = Path("data/local/sessions.sqlite3")
DEFAULT_MODEL = Path("artifacts/model.joblib")
DEFAULT_VOICE_MODEL = Path("artifacts/voice/model.joblib")


class SessionCreate(BaseModel):
    participant_code: Annotated[str, Field(min_length=1, max_length=64)]
    handedness: Literal["left", "right"]
    age_band: Literal["under_40", "40_49", "50_59", "60_69", "70_79", "80_plus", "not_shared"]
    medication_state: Literal["not_applicable", "before_pd_medication", "after_pd_medication", "unknown"]
    consent_research: bool

    @field_validator("participant_code")
    @classmethod
    def clean_participant_code(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("participant code cannot be blank")
        return value


class CapturePoint(BaseModel):
    x: float
    y: float
    t: Annotated[float, Field(ge=0)]
    pressure: Annotated[float, Field(ge=0, le=1)] = 0
    tilt_x: Annotated[float, Field(ge=-90, le=90)] = 0
    tilt_y: Annotated[float, Field(ge=-90, le=90)] = 0
    pointer_type: Literal["pen", "touch", "mouse", "unknown"] = "unknown"
    stroke: Annotated[int, Field(ge=0)] = 0


class TrialCreate(BaseModel):
    mode: Literal["static", "dynamic"]
    hand: Literal["left", "right"]
    repetition: Annotated[int, Field(ge=1, le=3)] = 1
    canvas_width: Annotated[float, Field(gt=100, le=4000)]
    canvas_height: Annotated[float, Field(gt=100, le=4000)]
    points: Annotated[list[CapturePoint], Field(min_length=20, max_length=50_000)]


def _trial_frame(trial: dict, test_id: int) -> pd.DataFrame:
    rows = []
    for point in trial["points"]:
        tilt = min(float(np.hypot(point["tilt_x"], point["tilt_y"])), 90.0)
        rows.append(
            (
                point["x"],
                point["y"],
                0,
                point["pressure"] * 1024.0,
                (90.0 - tilt) * 10.0,
                point["t"],
                test_id,
            )
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def assess_quality(trial: dict) -> dict:
    """Apply device and geometry checks before a trial can be scored."""
    test_id = 0 if trial["mode"] == "static" else 1
    frame = _trial_frame(trial, test_id)
    points = trial["points"]
    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    timestamps = frame["timestamp"].to_numpy(dtype=float)
    duration_s = float((np.max(timestamps) - np.min(timestamps)) / 1000.0)
    extent = float(np.hypot(np.ptp(x), np.ptp(y)))
    pointer_types = sorted({point["pointer_type"] for point in points})
    pressure = np.asarray([point["pressure"] for point in points], dtype=float)
    pressure_range = float(np.ptp(pressure))
    sample_rate_hz = float(len(points) / duration_s) if duration_s > 0 else 0.0
    errors: list[str] = []
    warnings: list[str] = []

    if len(points) < 80:
        errors.append("Too few recorded points; draw continuously and more slowly.")
    if duration_s < 1.0:
        errors.append("Drawing was completed too quickly to assess.")
    if duration_s > 90.0:
        errors.append("Drawing took longer than the supported capture window.")
    if extent < min(trial["canvas_width"], trial["canvas_height"]) * 0.25:
        errors.append("Spiral is too small; follow the full template.")
    if "pen" not in pointer_types:
        warnings.append("A pen was not detected. Mouse or touch results are demonstration-only.")
    elif pressure_range < 0.02:
        warnings.append(
            "Pen pressure did not vary. Check that Windows Ink and pressure are enabled."
        )
    if sample_rate_hz < 40.0:
        warnings.append(
            "Effective sampling rate is below 40 Hz; some dynamic features may be unreliable."
        )

    features = extract_test_features(frame, test_id)
    turns = float(features.get("turns", 0.0))
    if turns < 1.8:
        errors.append("Fewer than two spiral turns were detected.")
    if turns > 5.0:
        warnings.append("More turns than expected were detected; check the tracing.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "point_count": len(points),
        "duration_s": round(duration_s, 3),
        "extent_px": round(extent, 3),
        "turns": round(turns, 3),
        "sample_rate_hz": round(sample_rate_hz, 1),
        "pressure_range": [round(float(np.min(pressure)), 3), round(float(np.max(pressure)), 3)],
        "pointer_types": pointer_types,
    }


def _score_session(session: dict, model_path: Path) -> dict:
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifact is missing. Run 'spiral-pd train' first.",
        )
    bundle = joblib.load(model_path)
    trials = [trial for trial in session["trials"] if trial["quality"]["valid"]]
    indexed = {(trial["hand"], trial["repetition"], trial["mode"]): trial for trial in trials}
    pair_scores: list[dict] = []

    for hand in ("left", "right"):
        for repetition in range(1, 4):
            static = indexed.get((hand, repetition, "static"))
            dynamic = indexed.get((hand, repetition, "dynamic"))
            if static is None or dynamic is None:
                continue
            frame = pd.concat(
                [_trial_frame(static, 0), _trial_frame(dynamic, 1)], ignore_index=True
            )
            features = extract_participant_features(frame)
            row = pd.DataFrame([features]).reindex(columns=bundle["feature_names"])
            score = float(bundle["pipeline"].predict_proba(row)[0, 1])
            pair_scores.append(
                {"hand": hand, "repetition": repetition, "screening_score": score}
            )

    if not pair_scores:
        raise HTTPException(
            status_code=422,
            detail=(
                "At least one valid static and dynamic spiral pair from the same hand "
                "and repetition is required."
            ),
        )

    scores = [item["screening_score"] for item in pair_scores]
    aggregate = float(median(scores))
    return {
        "experimental_screening_score": aggregate,
        "pattern_signal": (
            "elevated" if aggregate >= ELEVATED_SIGNAL_THRESHOLD else "lower"
        ),
        "decision_threshold": ELEVATED_SIGNAL_THRESHOLD,
        "pair_scores": pair_scores,
        "score_range": [float(min(scores)), float(max(scores))],
        "valid_trials": len(trials),
        "warning": (
            "Research-only motor-pattern score. It is not a diagnosis and is not a "
            "population-calibrated probability of Parkinson's disease. Concerning "
            "symptoms require assessment by a qualified clinician."
        ),
    }


def _score_voice_session(session: dict) -> dict:
    valid_recordings = [
        trial
        for trial in session["voice_trials"]
        if trial["quality"]["valid"] and trial["result"] is not None
    ]
    if len(valid_recordings) < 3:
        raise HTTPException(
            status_code=422,
            detail="Three valid sustained-'ah' recordings are required.",
        )
    usable = [
        trial
        for trial in valid_recordings
        if trial["result"].get("domain_match")
        and trial["result"].get("screening_score") is not None
    ]
    if len(usable) < 3:
        outside = sorted(
            {
                name
                for trial in valid_recordings
                for name in trial["result"].get("out_of_distribution_features", [])
            }
        )
        return {
            "status": "unscorable",
            "experimental_voice_score": None,
            "pattern_signal": "unscorable",
            "valid_recordings": len(valid_recordings),
            "domain_match": False,
            "out_of_distribution_features": outside,
            "decision_threshold": ELEVATED_SIGNAL_THRESHOLD,
            "warning": (
                "The recordings were clear enough to analyze but did not match the "
                "training domain. No numeric voice score was produced."
            ),
        }
    scores = [float(trial["result"]["screening_score"]) for trial in usable]
    aggregate = float(median(scores))
    return {
        "experimental_voice_score": aggregate,
        "status": "scored",
        "pattern_signal": (
            "elevated" if aggregate >= ELEVATED_SIGNAL_THRESHOLD else "lower"
        ),
        "recording_scores": scores,
        "valid_recordings": len(usable),
        "domain_match": all(bool(trial["result"]["domain_match"]) for trial in usable),
        "decision_threshold": ELEVATED_SIGNAL_THRESHOLD,
        "warning": (
            "Research-only acoustic-pattern score trained on a small public telephone-"
            "audio cohort. It is not a diagnostic or population-calibrated probability."
        ),
    }


def create_app(
    database_path: str | Path = DEFAULT_DB,
    model_path: str | Path = DEFAULT_MODEL,
    voice_model_path: str | Path = DEFAULT_VOICE_MODEL,
) -> FastAPI:
    app = FastAPI(title="Parkinson's Spiral Research Tool", version="0.2.0")
    app.state.store = SessionStore(database_path)
    app.state.model_path = Path(model_path)
    app.state.voice_model_path = Path(voice_model_path)
    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_ready": app.state.model_path.exists(),
            "voice_model_ready": app.state.voice_model_path.exists(),
            "capture": "pointer-events",
        }

    @app.get("/api/model")
    def model_information() -> dict:
        metrics_path = app.state.model_path.with_name("metrics.json")
        metrics = None
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")).get("metrics")
        return {
            "ready": app.state.model_path.exists(),
            "name": "UCI trajectory logistic baseline",
            "research_only": True,
            "metrics": metrics,
            "voice_ready": app.state.voice_model_path.exists(),
        }

    @app.post("/api/sessions", status_code=201)
    def create_session(payload: SessionCreate) -> dict:
        if not payload.consent_research:
            raise HTTPException(status_code=422, detail="Research consent is required.")
        return app.state.store.create_session(payload.model_dump())

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        try:
            return app.state.store.get_session(session_id, include_points=False)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found.") from error

    @app.post("/api/sessions/{session_id}/trials", status_code=201)
    def add_trial(session_id: str, payload: TrialCreate) -> dict:
        try:
            session = app.state.store.get_session(session_id, include_points=False)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found.") from error
        dominant = session["handedness"]
        if dominant in {"left", "right"} and payload.hand != dominant:
            raise HTTPException(
                status_code=422,
                detail="Only the participant's dominant hand is used in this protocol.",
            )
        values = payload.model_dump()
        quality = assess_quality(values)
        try:
            trial = app.state.store.add_trial(session_id, values, quality)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found.") from error
        return trial

    @app.post("/api/sessions/{session_id}/voice", status_code=201)
    async def add_voice_trial(
        session_id: str,
        request: Request,
        repetition: Annotated[int, Field(ge=1, le=3)] = 1,
    ) -> dict:
        if not app.state.store.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")
        if not app.state.voice_model_path.exists():
            raise HTTPException(status_code=503, detail="Voice model is not trained.")
        data = await request.body()
        if not data:
            raise HTTPException(status_code=422, detail="An audio recording is required.")
        if len(data) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio recording exceeds 15 MB.")
        try:
            samples, sample_rate = decode_audio(data)
            features, quality = extract_voice_features(samples, sample_rate)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        result = None
        if quality["valid"]:
            result = score_voice_features(features, app.state.voice_model_path)
            if not result["domain_match"]:
                quality["warnings"].append(
                    "Several acoustic measurements fall outside the training range; "
                    "the voice score was suppressed rather than extrapolated."
                )
        return app.state.store.add_voice_trial(
            session_id, repetition, features, quality, result
        )

    @app.post("/api/sessions/{session_id}/voice-score")
    def score_voice_session(session_id: str) -> dict:
        try:
            session = app.state.store.get_session(session_id, include_points=False)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found.") from error
        return _score_voice_session(session)

    @app.post("/api/sessions/{session_id}/score")
    def score_session(session_id: str) -> dict:
        try:
            session = app.state.store.get_session(session_id, include_points=True)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found.") from error
        return _score_session(session, app.state.model_path)

    @app.get("/api/sessions/{session_id}/export")
    def export_session(session_id: str) -> JSONResponse:
        try:
            session = app.state.store.get_session(session_id, include_points=True)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found.") from error
        return JSONResponse(
            session,
            headers={
                "Content-Disposition": f'attachment; filename="spiral-session-{session_id}.json"'
            },
        )

    @app.get("/api/sessions/{session_id}/report.pdf")
    def session_report(session_id: str) -> Response:
        try:
            session = app.state.store.get_session(session_id, include_points=True)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found.") from error
        motor_result = _score_session(session, app.state.model_path)
        voice_result = _score_voice_session(session)
        pdf = build_session_report(
            session,
            motor_result,
            voice_result,
            app.state.model_path,
            app.state.voice_model_path,
        )
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="parkinsons-research-report-{session_id[:12]}.pdf"'
                )
            },
        )

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> Response:
        if not app.state.store.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")
        return Response(status_code=204)

    return app


app = create_app()

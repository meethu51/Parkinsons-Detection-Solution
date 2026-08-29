"""Small local SQLite store for pseudonymous capture sessions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Persist sessions and raw trials without collecting names or contact details."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    participant_code TEXT NOT NULL,
                    handedness TEXT NOT NULL,
                    age_band TEXT NOT NULL,
                    medication_state TEXT NOT NULL,
                    consent_research INTEGER NOT NULL CHECK (consent_research IN (0, 1))
                );

                CREATE TABLE IF NOT EXISTS trials (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    hand TEXT NOT NULL,
                    repetition INTEGER NOT NULL,
                    canvas_width REAL NOT NULL,
                    canvas_height REAL NOT NULL,
                    points_json TEXT NOT NULL,
                    quality_json TEXT NOT NULL,
                    UNIQUE(session_id, mode, hand, repetition)
                );

                CREATE TABLE IF NOT EXISTS voice_trials (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    repetition INTEGER NOT NULL,
                    features_json TEXT NOT NULL,
                    quality_json TEXT NOT NULL,
                    result_json TEXT,
                    UNIQUE(session_id, repetition)
                );
                """
            )

    def create_session(self, values: dict[str, Any]) -> dict[str, Any]:
        session_id = uuid4().hex
        created_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    id, created_at, participant_code, handedness,
                    age_band, medication_state, consent_research
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    created_at,
                    values["participant_code"],
                    values["handedness"],
                    values["age_band"],
                    values["medication_state"],
                    int(values["consent_research"]),
                ),
            )
        return self.get_session(session_id, include_points=False)

    def add_trial(
        self, session_id: str, values: dict[str, Any], quality: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.session_exists(session_id):
            raise KeyError(session_id)
        trial_id = uuid4().hex
        created_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO trials (
                    id, session_id, created_at, mode, hand, repetition,
                    canvas_width, canvas_height, points_json, quality_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, mode, hand, repetition) DO UPDATE SET
                    id = excluded.id,
                    created_at = excluded.created_at,
                    canvas_width = excluded.canvas_width,
                    canvas_height = excluded.canvas_height,
                    points_json = excluded.points_json,
                    quality_json = excluded.quality_json
                """,
                (
                    trial_id,
                    session_id,
                    created_at,
                    values["mode"],
                    values["hand"],
                    values["repetition"],
                    values["canvas_width"],
                    values["canvas_height"],
                    json.dumps(values["points"], separators=(",", ":")),
                    json.dumps(quality, separators=(",", ":")),
                ),
            )
        return self.get_trial(trial_id, include_points=False)

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def get_trial(self, trial_id: str, include_points: bool = True) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM trials WHERE id = ?", (trial_id,)
            ).fetchone()
        if row is None:
            raise KeyError(trial_id)
        return self._trial_from_row(row, include_points)

    def add_voice_trial(
        self,
        session_id: str,
        repetition: int,
        features: dict[str, Any],
        quality: dict[str, Any],
        result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Store derived acoustic measurements; raw microphone audio is not retained."""
        if not self.session_exists(session_id):
            raise KeyError(session_id)
        trial_id = uuid4().hex
        created_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO voice_trials (
                    id, session_id, created_at, repetition,
                    features_json, quality_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, repetition) DO UPDATE SET
                    id = excluded.id,
                    created_at = excluded.created_at,
                    features_json = excluded.features_json,
                    quality_json = excluded.quality_json,
                    result_json = excluded.result_json
                """,
                (
                    trial_id,
                    session_id,
                    created_at,
                    repetition,
                    json.dumps(features, separators=(",", ":")),
                    json.dumps(quality, separators=(",", ":")),
                    None if result is None else json.dumps(result, separators=(",", ":")),
                ),
            )
        return self.get_voice_trial(trial_id)

    def get_voice_trial(self, trial_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM voice_trials WHERE id = ?", (trial_id,)
            ).fetchone()
        if row is None:
            raise KeyError(trial_id)
        return self._voice_trial_from_row(row)

    def get_session(
        self, session_id: str, include_points: bool = True
    ) -> dict[str, Any]:
        with self._connect() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise KeyError(session_id)
            trials = connection.execute(
                """
                SELECT * FROM trials WHERE session_id = ?
                ORDER BY hand, mode, repetition
                """,
                (session_id,),
            ).fetchall()
            voice_trials = connection.execute(
                """
                SELECT * FROM voice_trials WHERE session_id = ?
                ORDER BY repetition
                """,
                (session_id,),
            ).fetchall()
        result = dict(session)
        result["consent_research"] = bool(result["consent_research"])
        result["trials"] = [
            self._trial_from_row(row, include_points) for row in trials
        ]
        result["voice_trials"] = [self._voice_trial_from_row(row) for row in voice_trials]
        return result

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    @staticmethod
    def _trial_from_row(row: sqlite3.Row, include_points: bool) -> dict[str, Any]:
        trial = dict(row)
        trial["quality"] = json.loads(trial.pop("quality_json"))
        raw_points = trial.pop("points_json")
        if include_points:
            trial["points"] = json.loads(raw_points)
        return trial

    @staticmethod
    def _voice_trial_from_row(row: sqlite3.Row) -> dict[str, Any]:
        trial = dict(row)
        trial["features"] = json.loads(trial.pop("features_json"))
        trial["quality"] = json.loads(trial.pop("quality_json"))
        raw_result = trial.pop("result_json")
        trial["result"] = None if raw_result is None else json.loads(raw_result)
        return trial

"""Training and evaluation for an explicitly exploratory screening model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .settings import ELEVATED_SIGNAL_THRESHOLD


RANDOM_SEED = 20260829


def make_pipeline() -> Pipeline:
    """Return a small, regularized, class-balanced and inspectable model."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=5_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _point_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    predictions = (scores >= ELEVATED_SIGNAL_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "decision_threshold": ELEVATED_SIGNAL_THRESHOLD,
        "balanced_accuracy_at_threshold": float(balanced_accuracy_score(y_true, predictions)),
        "sensitivity_at_threshold": float(tp / max(tp + fn, 1)),
        "specificity_at_threshold": float(tn / max(tn + fp, 1)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _bootstrap_intervals(
    y_true: np.ndarray, scores: np.ndarray, iterations: int = 2_000
) -> dict[str, list[float]]:
    generator = np.random.default_rng(RANDOM_SEED)
    collected: dict[str, list[float]] = {
        "roc_auc": [],
        "sensitivity_at_threshold": [],
        "specificity_at_threshold": [],
    }
    for _ in range(iterations):
        indices = generator.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size < 2:
            continue
        sampled = _point_metrics(sampled_y, scores[indices])
        for key in collected:
            collected[key].append(float(sampled[key]))
    return {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in collected.items()
    }


def train_and_evaluate(table: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    """Evaluate with participant-level folds, then fit a final research model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = [
        column for column in table.columns if column not in {"participant_id", "label"}
    ]
    x = table[feature_names]
    y = table["label"].to_numpy(dtype=int)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    out_of_fold_scores = cross_val_predict(
        make_pipeline(), x, y, cv=folds, method="predict_proba"
    )[:, 1]

    metrics: dict[str, object] = {
        "warning": (
            "Research-only screening baseline. Scores are not diagnostic or calibrated "
            "to population Parkinson's prevalence."
        ),
        "evaluation": "5-fold stratified participant-level cross-validation",
        "participants": int(len(table)),
        "parkinsons": int(np.sum(y == 1)),
        "controls": int(np.sum(y == 0)),
        **_point_metrics(y, out_of_fold_scores),
        "bootstrap_95_percent_intervals": _bootstrap_intervals(y, out_of_fold_scores),
    }

    pipeline = make_pipeline()
    pipeline.fit(x, y)
    joblib.dump(
        {"pipeline": pipeline, "feature_names": feature_names, "version": 1},
        output_dir / "model.joblib",
    )
    table[["participant_id", "label"]].assign(
        out_of_fold_screening_score=out_of_fold_scores
    ).to_csv(output_dir / "cross_validation_predictions.csv", index=False)

    coefficients = pipeline.named_steps["classifier"].coef_[0]
    pd.DataFrame(
        {"feature": feature_names, "standardized_coefficient": coefficients}
    ).assign(absolute_coefficient=lambda data: data["standardized_coefficient"].abs()).sort_values(
        "absolute_coefficient", ascending=False
    ).to_csv(output_dir / "feature_coefficients.csv", index=False)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "UCI Machine Learning Repository dataset 395",
        "dataset_doi": "10.24432/C5Q01S",
        "feature_names": feature_names,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metrics


def predict_file(model_path: str | Path, trajectory_path: str | Path) -> float:
    """Generate an experimental score for one UCI-style participant file."""
    from .features import extract_participant_features, read_trajectory

    bundle = joblib.load(model_path)
    features = extract_participant_features(read_trajectory(trajectory_path))
    row = pd.DataFrame([features]).reindex(columns=bundle["feature_names"])
    return float(bundle["pipeline"].predict_proba(row)[0, 1])

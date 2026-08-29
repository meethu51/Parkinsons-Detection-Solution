"""Acquisition and benchmarking for additional online research datasets."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model import RANDOM_SEED, _bootstrap_intervals, _point_metrics


NEWHANDPD_BASE = "http://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/"
NEWHANDPD_FILES = {
    "healthy_spirals": "NewHealthy/HealthySpiral.zip",
    "parkinsons_spirals": "NewPatients/PatientSpiral.zip",
    "metadata": "NewSpiral.csv",
}

DATASET_CATALOG = [
    {
        "id": "uci395",
        "participants": 77,
        "classes": "62 Parkinson's / 15 controls",
        "modality": "x/y/time/pressure tablet trajectories",
        "access": "immediate",
        "license": "CC BY 4.0",
        "integrated": True,
        "url": "https://archive.ics.uci.edu/dataset/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet",
    },
    {
        "id": "newhandpd",
        "participants": 66,
        "classes": "31 Parkinson's / 35 controls",
        "modality": "four spiral images per participant plus smart-pen signals",
        "access": "immediate research download",
        "license": "No explicit license posted; verify intended use with dataset authors",
        "integrated": True,
        "url": "http://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/",
    },
    {
        "id": "handpd",
        "participants": 92,
        "classes": "74 Parkinson's / 18 controls",
        "modality": "four spiral and four meander images per participant",
        "access": "immediate research download",
        "license": "No explicit license posted; verify intended use with dataset authors",
        "integrated": False,
        "url": "http://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/",
    },
    {
        "id": "pahaw",
        "participants": 75,
        "classes": "37 Parkinson's / 38 controls",
        "modality": "dynamic tablet trajectories across eight handwriting tasks",
        "access": "signed institutional license required",
        "license": "research-only, noncommercial, two-year agreement",
        "integrated": False,
        "url": "https://bdalab.utko.fekt.vut.cz/wp-content/uploads/2016/05/PaHaW_licence_agreement.pdf",
    },
    {
        "id": "cc-phd",
        "participants": None,
        "classes": "Parkinson's / essential tremor / healthy controls",
        "modality": "spiral, meander, handwriting, and dynamic pen channels",
        "access": "academic application required",
        "license": "restricted academic research",
        "integrated": False,
        "url": "https://github.com/dreamhcy/MLforPD_DataSet",
    },
    {
        "id": "mendeley-spiral-images",
        "participants": None,
        "classes": "Parkinson's / healthy controls",
        "modality": "spiral and wave images republished from the common drawings corpus",
        "access": "immediate",
        "license": "CC BY 4.0",
        "integrated": False,
        "url": "https://data.mendeley.com/datasets/fd5wd6wmdj/1",
    },
    {
        "id": "figshare-sustained-ah-raw-audio",
        "participants": 81,
        "classes": "40 Parkinson's / 41 healthy controls",
        "modality": "telephone-recorded prolonged /a/ WAV files",
        "access": "immediate",
        "license": "CC BY 4.0",
        "integrated": True,
        "url": "https://doi.org/10.6084/m9.figshare.23849127.v1",
    },
    {
        "id": "uci-parkinsons-voice",
        "participants": 31,
        "classes": "23 Parkinson's / 8 controls",
        "modality": "195 voice recordings with 22 acoustic measures",
        "access": "immediate",
        "license": "CC BY 4.0",
        "integrated": False,
        "url": "https://archive.ics.uci.edu/dataset/174/parkinsons",
    },
    {
        "id": "physionet-gaitpdb",
        "participants": 166,
        "classes": "93 Parkinson's / 73 controls",
        "modality": "foot-force sensor gait recordings",
        "access": "immediate",
        "license": "Open Data Commons Attribution 1.0",
        "integrated": False,
        "url": "https://physionet.org/content/gaitpdb/1.0.0/",
    },
    {
        "id": "mpower",
        "participants": 8320,
        "classes": "Parkinson's / controls with self-reported and survey metadata",
        "modality": "smartphone tapping, walking, standing, voice, and cognition",
        "access": "Synapse account and data-use conditions required",
        "license": "controlled research access",
        "integrated": False,
        "url": "https://www.synapse.org/Synapse:syn4993293/datasets/",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_paths(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.jpg")
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    ]


def _is_complete(root: Path) -> bool:
    return (root / "metadata.csv").exists() and len(_image_paths(root)) == 264


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def _write_manifest(root: Path, checksums: dict[str, str] | None = None) -> None:
    manifest = {
        "dataset": "NewHandPD",
        "source": NEWHANDPD_BASE,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "participants": 66,
        "spiral_images": len(_image_paths(root)),
        "usage_note": (
            "The source page provides public research downloads but does not state an "
            "explicit data license. Confirm reuse terms with the authors before "
            "redistribution, commercial use, or public deployment."
        ),
        "download_sha256": checksums or {},
    }
    (root / "SOURCE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def download_newhandpd(destination: str | Path, force: bool = False) -> Path:
    """Download the balanced NewHandPD spiral-image subset from its official page."""
    destination = Path(destination).resolve()
    if _is_complete(destination) and not force:
        if not (destination / "SOURCE.json").exists():
            _write_manifest(destination)
        return destination
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="newhandpd-") as temporary:
        temporary_root = Path(temporary)
        downloaded: dict[str, Path] = {}
        for name, relative_url in NEWHANDPD_FILES.items():
            suffix = ".csv" if relative_url.endswith(".csv") else ".zip"
            target = temporary_root / f"{name}{suffix}"
            urllib.request.urlretrieve(NEWHANDPD_BASE + relative_url, target)
            downloaded[name] = target

        _safe_extract(downloaded["healthy_spirals"], destination / "healthy")
        _safe_extract(downloaded["parkinsons_spirals"], destination / "parkinsons")
        (destination / "metadata.csv").write_bytes(downloaded["metadata"].read_bytes())
        checksums = {name: _sha256(path) for name, path in downloaded.items()}

    if not _is_complete(destination):
        raise ValueError(
            f"NewHandPD verification failed: expected 264 spiral images under {destination}"
        )
    _write_manifest(destination, checksums)
    return destination


def read_newhandpd_metadata(root: str | Path) -> pd.DataFrame:
    """Read provided features while retaining a participant grouping identifier."""
    metadata_path = Path(root) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"NewHandPD metadata not found at {metadata_path}. Download the dataset first."
        )
    frame = pd.read_csv(metadata_path)
    frame = frame.rename(columns={frame.columns[0]: "participant_id"})
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["label"] = (pd.to_numeric(frame["CLASS_TYPE"]) == 2).astype(int)
    if frame["participant_id"].nunique() != 66:
        raise ValueError("Expected 66 unique NewHandPD participants.")
    return frame


def _newhandpd_pipeline() -> Pipeline:
    return Pipeline(
        [
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


def benchmark_newhandpd(root: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Evaluate NewHandPD features with all four images grouped by participant."""
    table = read_newhandpd_metadata(root)
    metadata_columns = {
        "participant_id",
        "IMAGE_NAME",
        "ID_PATIENT",
        "CLASS_TYPE",
        "GENDER",
        "RIGH/LEFT-HANDED",
        "AGE",
        "label",
    }
    feature_names = [column for column in table.columns if column not in metadata_columns]
    x = table[feature_names].apply(pd.to_numeric, errors="coerce")
    y = table["label"].to_numpy(dtype=int)
    groups = table["participant_id"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_SEED
    )
    out_of_fold = np.full(len(table), np.nan, dtype=float)
    for train_indices, test_indices in splitter.split(x, y, groups):
        pipeline = _newhandpd_pipeline()
        pipeline.fit(x.iloc[train_indices], y[train_indices])
        out_of_fold[test_indices] = pipeline.predict_proba(x.iloc[test_indices])[:, 1]

    image_predictions = table[["participant_id", "label", "IMAGE_NAME"]].copy()
    image_predictions["out_of_fold_screening_score"] = out_of_fold
    participant_predictions = (
        image_predictions.groupby("participant_id", as_index=False)
        .agg(label=("label", "first"), out_of_fold_screening_score=("out_of_fold_screening_score", "mean"))
        .sort_values("participant_id")
    )
    participant_y = participant_predictions["label"].to_numpy(dtype=int)
    participant_scores = participant_predictions["out_of_fold_screening_score"].to_numpy()
    metrics: dict[str, object] = {
        "warning": (
            "Research-only subject-grouped benchmark using features supplied with "
            "NewHandPD. This model is not deployed in the tablet app."
        ),
        "evaluation": "5-fold stratified participant-grouped cross-validation",
        "participants": int(len(participant_predictions)),
        "parkinsons": int(np.sum(participant_y == 1)),
        "controls": int(np.sum(participant_y == 0)),
        "images": int(len(table)),
        **_point_metrics(participant_y, participant_scores),
        "bootstrap_95_percent_intervals": _bootstrap_intervals(
            participant_y, participant_scores
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_pipeline = _newhandpd_pipeline()
    final_pipeline.fit(x, y)
    joblib.dump(
        {
            "pipeline": final_pipeline,
            "feature_names": feature_names,
            "dataset": "NewHandPD provided features",
            "version": 1,
        },
        output_dir / "model.joblib",
    )
    image_predictions.to_csv(output_dir / "image_predictions.csv", index=False)
    participant_predictions.to_csv(
        output_dir / "participant_predictions.csv", index=False
    )
    coefficients = final_pipeline.named_steps["classifier"].coef_[0]
    pd.DataFrame(
        {"feature": feature_names, "standardized_coefficient": coefficients}
    ).assign(absolute_coefficient=lambda data: data["standardized_coefficient"].abs()).sort_values(
        "absolute_coefficient", ascending=False
    ).to_csv(output_dir / "feature_coefficients.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics

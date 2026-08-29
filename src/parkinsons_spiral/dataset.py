"""Dataset acquisition and participant-level table construction."""

from __future__ import annotations

import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .features import extract_participant_features, read_trajectory


UCI_URL = (
    "https://archive.ics.uci.edu/static/public/395/"
    "parkinson%2Bdisease%2Bspiral%2Bdrawings%2Busing%2Bdigitized%2Bgraphics%2Btablet.zip"
)


def download_uci(destination: str | Path, force: bool = False) -> Path:
    """Download and safely extract UCI dataset 395."""
    destination = Path(destination).resolve()
    marker = destination / "hw_dataset" / "readme.txt"
    if marker.exists() and not force:
        return destination
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spiral-pd-") as temporary:
        archive = Path(temporary) / "dataset.zip"
        urllib.request.urlretrieve(UCI_URL, archive)
        with zipfile.ZipFile(archive) as bundle:
            root = destination.resolve()
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if root != target and root not in target.parents:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            bundle.extractall(destination)
    return destination


def _participant_files(root: Path) -> list[tuple[Path, int]]:
    groups = (
        (root / "hw_dataset" / "control", 0),
        (root / "hw_dataset" / "parkinson", 1),
        (root / "new_dataset" / "parkinson", 1),
    )
    files: list[tuple[Path, int]] = []
    for directory, label in groups:
        files.extend((path, label) for path in sorted(directory.glob("*.txt")))
    return files


def build_participant_table(root: str | Path) -> pd.DataFrame:
    """Build exactly one model row for every participant file."""
    root = Path(root)
    rows: list[dict[str, object]] = []
    for path, label in _participant_files(root):
        features = extract_participant_features(read_trajectory(path))
        rows.append({"participant_id": path.stem, "label": label, **features})
    if not rows:
        raise FileNotFoundError(
            f"No UCI participant files found under {root}. Run the download command first."
        )
    return pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)

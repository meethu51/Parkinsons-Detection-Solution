from __future__ import annotations

from pathlib import Path

import pandas as pd

from parkinsons_spiral.online_datasets import (
    benchmark_newhandpd,
    read_newhandpd_metadata,
)


def _metadata(path: Path) -> None:
    rows = []
    for label, prefix, participants in ((0, "H", 35), (1, "P", 31)):
        for participant in range(1, participants + 1):
            for image in range(1, 5):
                rows.append(
                    {
                        "_ID_EXAM": f"{prefix}{participant}",
                        "IMAGE_NAME": f"sp{image}-{prefix}{participant}.jpg",
                        "ID_PATIENT": participant,
                        "CLASS_TYPE": label + 1,
                        "GENDER": "F" if participant % 2 else "M",
                        "RIGH/LEFT-HANDED": "R",
                        "AGE": 55 + label * 5,
                        "RMS": participant * 0.1 + label * 2 + image * 0.01,
                        "MRT": participant * 0.01 + label + image * 0.001,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_newhandpd_benchmark_groups_four_images_per_participant(tmp_path: Path) -> None:
    data_dir = tmp_path / "newhandpd"
    data_dir.mkdir()
    _metadata(data_dir / "metadata.csv")

    table = read_newhandpd_metadata(data_dir)
    assert len(table) == 264
    assert table["participant_id"].nunique() == 66

    metrics = benchmark_newhandpd(data_dir, tmp_path / "artifacts")
    assert metrics["participants"] == 66
    assert metrics["images"] == 264
    assert metrics["parkinsons"] == 31
    assert metrics["controls"] == 35
    assert (tmp_path / "artifacts" / "participant_predictions.csv").exists()

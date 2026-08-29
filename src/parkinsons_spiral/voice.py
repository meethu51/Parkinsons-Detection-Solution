"""Research voice baseline and local sustained-vowel feature extraction.

The deployed model is trained on labeled raw sustained-/a/ telephone recordings
using this exact extractor. A distribution check fails closed when a live recording
does not resemble the training domain.
"""

from __future__ import annotations

import io
import json
import subprocess
import urllib.request
import wave
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import find_peaks, resample_poly
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .settings import ELEVATED_SIGNAL_THRESHOLD


VOICE_DATA_URL = (
    "https://archive.ics.uci.edu/static/public/489/"
    "parkinson%2Bdataset%2Bwith%2Breplicated%2Bacoustic%2Bfeatures.zip"
)
VOICE_DATA_DOI = "10.24432/C5701F"
UCI_VOICE_FEATURES = (
    "Jitter_rel",
    "Jitter_RAP",
    "Jitter_PPQ",
    "Shim_loc",
    "Shim_dB",
    "Shim_APQ3",
    "Shim_APQ5",
    "Shi_APQ11",
)
FIGSHARE_ARTICLE_ID = 23849127
FIGSHARE_DATA_DOI = "10.6084/m9.figshare.23849127.v1"
FIGSHARE_FILES = {
    "PD_AH.zip": "https://ndownloader.figshare.com/files/41836710",
    "HC_AH.zip": "https://ndownloader.figshare.com/files/41836713",
}
VOICE_FEATURES = (
    *UCI_VOICE_FEATURES,
    "pitch_median_hz",
    "pitch_cv",
    "hnr_proxy_db",
    "energy_cv",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_flatness",
    *(f"mfcc_{index}_mean" for index in range(13)),
    *(f"mfcc_{index}_std" for index in range(13)),
)
RANDOM_SEED = 20260829
TARGET_VOICE_SAMPLE_RATE = 8_000


def download_uci489(destination: str | Path, force: bool = False) -> Path:
    """Download and extract the official UCI 489 CSV."""
    destination = Path(destination)
    csv_path = destination / "ReplicatedAcousticFeatures-ParkinsonDatabase.csv"
    if csv_path.exists() and not force:
        return csv_path
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "uci489.zip"
    urllib.request.urlretrieve(VOICE_DATA_URL, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Expected exactly one CSV in the UCI 489 archive.")
        with archive.open(members[0]) as source, csv_path.open("wb") as target:
            target.write(source.read())
    return csv_path


def read_uci489(csv_path: str | Path) -> pd.DataFrame:
    """Aggregate the three dependent recordings into one row per participant."""
    frame = pd.read_csv(csv_path)
    required = {"ID", "Status", *UCI_VOICE_FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"UCI 489 CSV is missing columns: {sorted(missing)}")
    numeric = frame[["Status", *UCI_VOICE_FEATURES]].apply(pd.to_numeric, errors="coerce")
    frame[["Status", *UCI_VOICE_FEATURES]] = numeric
    if frame[list(required)].isna().any().any():
        raise ValueError("UCI 489 CSV contains invalid required values.")
    grouped = frame.groupby("ID", sort=True)
    table = grouped[list(UCI_VOICE_FEATURES)].median().reset_index()
    table["label"] = grouped["Status"].first().to_numpy(dtype=int)
    table = table.rename(columns={"ID": "participant_id"})
    if len(table) != 80 or table["label"].value_counts().to_dict() != {0: 40, 1: 40}:
        raise ValueError("Expected 80 UCI 489 participants split 40 control / 40 PD.")
    return table


def download_figshare_voice(destination: str | Path, force: bool = False) -> Path:
    """Download the CC BY 4.0 labeled sustained-/a/ WAV collection."""
    destination = Path(destination)
    expected = destination / "hc" / "HC_AH"
    if expected.exists() and len(list(destination.rglob("*.wav"))) == 81 and not force:
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    for filename, url in FIGSHARE_FILES.items():
        archive_path = destination / filename
        urllib.request.urlretrieve(url, archive_path)
        target = destination / ("pd" if filename.startswith("PD") else "hc")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                resolved = (target / member.filename).resolve()
                if target.resolve() not in resolved.parents and resolved != target.resolve():
                    raise ValueError("Unsafe path in Figshare voice archive.")
            archive.extractall(target)
    if len(list(destination.rglob("*.wav"))) != 81:
        raise ValueError("Expected 81 WAV files in the Figshare voice dataset.")
    return destination


def read_figshare_voice(root: str | Path) -> pd.DataFrame:
    """Extract the same feature vector used for live microphone inference."""
    root = Path(root)
    rows: list[dict] = []
    for label, folder in ((0, root / "hc"), (1, root / "pd")):
        for path in sorted(folder.rglob("*.wav")):
            samples, sample_rate = decode_audio(path.read_bytes())
            features, _ = extract_voice_features(samples, sample_rate)
            if all(np.isfinite(features[name]) for name in VOICE_FEATURES):
                rows.append(
                    {"participant_id": path.stem, "label": label, **features}
                )
    table = pd.DataFrame(rows)
    counts = table["label"].value_counts().to_dict() if not table.empty else {}
    if len(table) != 81 or counts != {0: 41, 1: 40}:
        raise ValueError(
            f"Expected 81 usable participants split 41 control / 40 PD; got {counts}."
        )
    return table


def _voice_pipeline(feature_count: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=min(12, feature_count))),
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


def train_voice_model(
    table: pd.DataFrame,
    output_dir: str | Path,
    *,
    dataset_name: str = "Figshare labeled sustained-/a/ raw audio",
    dataset_doi: str = FIGSHARE_DATA_DOI,
) -> dict:
    """Evaluate at participant level and fit the reduced live-audio baseline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = [
        column for column in table.columns if column not in {"participant_id", "label"}
    ]
    x = table[feature_names]
    y = table["label"].to_numpy(dtype=int)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_predict(
        _voice_pipeline(len(feature_names)), x, y, cv=folds, method="predict_proba"
    )[:, 1]
    predictions = (scores >= ELEVATED_SIGNAL_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    metrics = {
        "warning": (
            "Research-only acoustic-pattern baseline trained from a small telephone-audio "
            "cohort. Scores are not diagnostic or population-calibrated probabilities."
        ),
        "evaluation": "5-fold stratified participant-level cross-validation",
        "participants": int(len(table)),
        "parkinsons": int(np.sum(y == 1)),
        "controls": int(np.sum(y == 0)),
        "recordings_per_participant": 1,
        "features": feature_names,
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "decision_threshold": ELEVATED_SIGNAL_THRESHOLD,
        "balanced_accuracy_at_threshold": float(balanced_accuracy_score(y, predictions)),
        "sensitivity_at_threshold": float(tp / max(tp + fn, 1)),
        "specificity_at_threshold": float(tn / max(tn + fp, 1)),
    }
    pipeline = _voice_pipeline(len(feature_names)).fit(x, y)
    selected_mask = pipeline.named_steps["selector"].get_support()
    domain_feature_names = [
        name for name, selected in zip(feature_names, selected_mask) if selected
    ]
    quantiles = {
        feature: {
            "low": float(table[feature].quantile(0.005)),
            "high": float(table[feature].quantile(0.995)),
        }
        for feature in feature_names
    }
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": feature_names,
            "domain_feature_names": domain_feature_names,
            "training_quantiles": quantiles,
            "version": 1,
        },
        output_dir / "model.joblib",
    )
    pd.DataFrame(
        {
            "participant_id": table["participant_id"],
            "label": y,
            "out_of_fold_screening_score": scores,
        }
    ).to_csv(output_dir / "cross_validation_predictions.csv", index=False)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "dataset_doi": dataset_doi,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


def _decode_wave(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float32), int(sample_rate)


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV directly or use local FFmpeg for browser WebM/Opus audio."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        try:
            return _decode_wave(data)
        except (ValueError, wave.Error):
            # Some research WAV files use IEEE-float or WAVE_FORMAT_EXTENSIBLE,
            # which Python's standard-library reader does not decode.
            pass
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                str(TARGET_VOICE_SAMPLE_RATE),
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=data,
            capture_output=True,
            check=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("Audio could not be decoded; FFmpeg is required for browser recordings.") from error
    return (
        np.frombuffer(completed.stdout, dtype="<i2").astype(np.float32) / 32768.0,
        TARGET_VOICE_SAMPLE_RATE,
    )


def _apq(values: np.ndarray, window: int) -> float:
    if values.size < window:
        return float("nan")
    local = np.convolve(values, np.ones(window) / window, mode="valid")
    offset = window // 2
    aligned = values[offset : offset + local.size]
    return float(np.mean(np.abs(aligned - local)) / max(np.mean(values), 1e-9))


def _spectral_summary(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Return compact frame-level spectral and cepstral summaries."""
    frame_length = int(round(0.025 * sample_rate))
    hop = int(round(0.010 * sample_rate))
    if samples.size < frame_length:
        return {
            name: float("nan")
            for name in (
                "energy_cv",
                "zero_crossing_rate",
                "spectral_centroid_hz",
                "spectral_bandwidth_hz",
                "spectral_flatness",
                *(f"mfcc_{index}_mean" for index in range(13)),
                *(f"mfcc_{index}_std" for index in range(13)),
            )
        }
    frame_count = 1 + (samples.size - frame_length) // hop
    indices = np.arange(frame_length)[None, :] + hop * np.arange(frame_count)[:, None]
    frames = samples[indices]
    frame_energy = np.mean(frames**2, axis=1)
    windowed = frames * np.hanning(frame_length)[None, :]
    power = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)
    total_power = np.sum(power, axis=1) + 1e-12
    centroid = np.sum(power * frequencies[None, :], axis=1) / total_power
    bandwidth = np.sqrt(
        np.sum(power * (frequencies[None, :] - centroid[:, None]) ** 2, axis=1)
        / total_power
    )
    flatness = np.exp(np.mean(np.log(power + 1e-12), axis=1)) / (
        np.mean(power + 1e-12, axis=1)
    )
    zero_crossings = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)

    mel_low = 2595.0 * np.log10(1.0 + 50.0 / 700.0)
    mel_high = 2595.0 * np.log10(1.0 + (sample_rate / 2) / 700.0)
    mel_points = np.linspace(mel_low, mel_high, 28)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.clip(
        np.floor((frame_length + 1) * hz_points / sample_rate).astype(int),
        0,
        power.shape[1] - 1,
    )
    filters = np.zeros((26, power.shape[1]), dtype=float)
    for index in range(26):
        left, center, right = bins[index : index + 3]
        if center > left:
            filters[index, left:center] = np.arange(center - left) / (center - left)
        if right > center:
            filters[index, center:right] = np.arange(right - center, 0, -1) / (
                right - center
            )
    log_mel = np.log(np.maximum(power @ filters.T, 1e-12))
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :13]
    result = {
        "energy_cv": float(np.std(frame_energy) / max(np.mean(frame_energy), 1e-12)),
        "zero_crossing_rate": float(np.mean(zero_crossings)),
        "spectral_centroid_hz": float(np.mean(centroid)),
        "spectral_bandwidth_hz": float(np.mean(bandwidth)),
        "spectral_flatness": float(np.mean(flatness)),
    }
    result.update({f"mfcc_{i}_mean": float(np.mean(mfcc[:, i])) for i in range(13)})
    result.update({f"mfcc_{i}_std": float(np.std(mfcc[:, i])) for i in range(13)})
    return result


def extract_voice_features(samples: np.ndarray, sample_rate: int) -> tuple[dict, dict]:
    """Extract reproducible perturbation proxies and recording-quality measures."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or sample_rate < 8_000:
        raise ValueError("A non-empty mono recording sampled at 8 kHz or faster is required.")
    if sample_rate != TARGET_VOICE_SAMPLE_RATE:
        divisor = int(np.gcd(sample_rate, TARGET_VOICE_SAMPLE_RATE))
        samples = resample_poly(
            samples,
            TARGET_VOICE_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
        sample_rate = TARGET_VOICE_SAMPLE_RATE
    samples = samples - float(np.mean(samples))
    duration_s = float(samples.size / sample_rate)
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples**2)))
    clipping_fraction = float(np.mean(np.abs(samples) >= 0.98))

    frame_length = int(0.03 * sample_rate)
    hop = frame_length
    usable = samples[: (samples.size // hop) * hop]
    frame_rms = (
        np.sqrt(np.mean(usable.reshape(-1, hop) ** 2, axis=1))
        if usable.size
        else np.array([], dtype=float)
    )
    threshold = max(rms * 0.2, 0.003)
    voiced_flags = frame_rms >= threshold
    voiced_ratio = float(np.mean(voiced_flags)) if voiced_flags.size else 0.0
    if np.any(voiced_flags):
        active = np.flatnonzero(voiced_flags)
        start = max(0, int(active[0] * hop - 0.05 * sample_rate))
        end = min(samples.size, int((active[-1] + 1) * hop + 0.05 * sample_rate))
        voiced = samples[start:end]
    else:
        voiced = samples

    smoothed = np.convolve(voiced, np.ones(5) / 5.0, mode="same")
    pitch_probe = smoothed[: min(smoothed.size, sample_rate * 3)]
    if pitch_probe.size:
        fft_size = 1 << max(1, int(pitch_probe.size * 2 - 1).bit_length())
        spectrum = np.fft.rfft(pitch_probe, n=fft_size)
        autocorrelation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)
        minimum_lag = int(sample_rate / 350)
        maximum_lag = min(int(sample_rate / 70), pitch_probe.size - 1)
        if maximum_lag > minimum_lag:
            pitch_lag = minimum_lag + int(
                np.argmax(autocorrelation[minimum_lag : maximum_lag + 1])
            )
        else:
            pitch_lag = minimum_lag
    else:
        pitch_lag = int(sample_rate / 200)
    positive_prominence = max(float(np.std(smoothed)) * 0.25, 1e-4)
    peaks, _ = find_peaks(
        smoothed,
        distance=max(1, int(pitch_lag * 0.65)),
        prominence=positive_prominence,
    )
    periods = np.diff(peaks) / sample_rate
    valid = (periods >= 1 / 350) & (periods <= 1 / 70)
    periods = periods[valid]
    if periods.size:
        median_period = float(np.median(periods))
        periods = periods[np.abs(periods - median_period) <= median_period * 0.35]
    amplitudes = np.abs(smoothed[peaks]).astype(float)
    if amplitudes.size:
        median_amplitude = float(np.median(amplitudes))
        amplitudes = amplitudes[
            (amplitudes >= median_amplitude * 0.25) & (amplitudes <= median_amplitude * 4.0)
        ]

    errors: list[str] = []
    warnings: list[str] = []
    if duration_s < 3.0:
        errors.append("Recording is too short; sustain 'ah' for at least 3 seconds.")
    if duration_s > 12.0:
        errors.append("Recording is longer than the supported 12-second window.")
    if rms < 0.005 or voiced_ratio < 0.35:
        errors.append("Too little sustained voice was detected; move closer and try again.")
    if clipping_fraction > 0.02:
        errors.append("The recording is clipping; move farther from the microphone.")
    if periods.size < 80 or amplitudes.size < 80:
        errors.append("A stable sustained vowel could not be measured; hold one steady 'ah'.")

    if periods.size >= 5:
        mean_period = max(float(np.mean(periods)), 1e-9)
        jitter_rel = float(100.0 * np.mean(np.abs(np.diff(periods))) / mean_period)
        jitter_rap = _apq(periods, 3)
        jitter_ppq = _apq(periods, 5)
        pitch_median_hz = float(1.0 / np.median(periods))
        pitch_cv = float(np.std(1.0 / periods) / max(np.mean(1.0 / periods), 1e-9))
    else:
        jitter_rel = jitter_rap = jitter_ppq = float("nan")
        pitch_median_hz = pitch_cv = float("nan")
    if amplitudes.size >= 11:
        mean_amplitude = max(float(np.mean(amplitudes)), 1e-9)
        shimmer_local = float(np.mean(np.abs(np.diff(amplitudes))) / mean_amplitude)
        ratios = np.maximum(amplitudes[1:], 1e-9) / np.maximum(amplitudes[:-1], 1e-9)
        shimmer_db = float(np.mean(np.abs(20.0 * np.log10(ratios))))
        shimmer_apq3 = _apq(amplitudes, 3)
        shimmer_apq5 = _apq(amplitudes, 5)
        shimmer_apq11 = _apq(amplitudes, 11)
    else:
        shimmer_local = shimmer_db = shimmer_apq3 = shimmer_apq5 = shimmer_apq11 = float("nan")

    if pitch_probe.size and pitch_lag < autocorrelation.size:
        correlation = float(
            np.clip(
                autocorrelation[pitch_lag] / max(autocorrelation[0], 1e-12),
                0.0,
                0.999999,
            )
        )
        hnr_proxy_db = float(10.0 * np.log10((correlation + 1e-9) / (1.0 - correlation + 1e-9)))
    else:
        hnr_proxy_db = float("nan")

    features = {
        "Jitter_rel": jitter_rel,
        "Jitter_RAP": jitter_rap,
        "Jitter_PPQ": jitter_ppq,
        "Shim_loc": shimmer_local,
        "Shim_dB": shimmer_db,
        "Shim_APQ3": shimmer_apq3,
        "Shim_APQ5": shimmer_apq5,
        "Shi_APQ11": shimmer_apq11,
        "pitch_median_hz": pitch_median_hz,
        "pitch_cv": pitch_cv,
        "hnr_proxy_db": hnr_proxy_db,
    }
    features.update(_spectral_summary(voiced, sample_rate))
    if not all(np.isfinite(value) for value in features.values()):
        errors.append("Acoustic features could not be calculated from this recording.")
    quality = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "duration_s": round(duration_s, 3),
        "sample_rate_hz": sample_rate,
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "clipping_fraction": round(clipping_fraction, 5),
        "voiced_ratio": round(voiced_ratio, 3),
        "period_count": int(periods.size),
    }
    return features, quality


def score_voice_features(features: dict, model_path: str | Path) -> dict:
    bundle = joblib.load(model_path)
    row = pd.DataFrame([features]).reindex(columns=bundle["feature_names"])
    score = float(bundle["pipeline"].predict_proba(row)[0, 1])
    outside = []
    domain_features = bundle.get("domain_feature_names", bundle["feature_names"])
    quantiles = bundle.get("training_quantiles", {})
    for name in domain_features:
        limits = quantiles.get(name)
        if limits is None:
            continue
        value = float(features[name])
        if value < limits["low"] or value > limits["high"]:
            outside.append(name)
    domain_match = len(outside) <= max(2, int(np.ceil(len(domain_features) * 0.2)))
    return {
        "screening_score": score if domain_match else None,
        "pattern_signal": (
            "elevated"
            if domain_match and score >= ELEVATED_SIGNAL_THRESHOLD
            else "lower" if domain_match else "unscorable"
        ),
        "out_of_distribution_features": outside,
        "domain_match": domain_match,
        "decision_threshold": ELEVATED_SIGNAL_THRESHOLD,
    }

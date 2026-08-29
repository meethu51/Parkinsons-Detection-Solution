"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_participant_table, download_uci
from .model import predict_file, train_and_evaluate
from .online_datasets import DATASET_CATALOG, benchmark_newhandpd, download_newhandpd
from .voice import download_figshare_voice, read_figshare_voice, train_voice_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research-only Parkinson's spiral screening baseline"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    download = subcommands.add_parser("download", help="Download UCI dataset 395")
    download.add_argument("--data-dir", type=Path, default=Path("data/raw/uci395"))
    download.add_argument("--force", action="store_true")

    train = subcommands.add_parser("train", help="Train and cross-validate the baseline")
    train.add_argument("--data-dir", type=Path, default=Path("data/raw/uci395"))
    train.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train.add_argument(
        "--download", action="store_true", help="Download the dataset if it is absent"
    )

    predict = subcommands.add_parser(
        "predict", help="Score one UCI-format participant trajectory"
    )
    predict.add_argument("trajectory", type=Path)
    predict.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))

    serve = subcommands.add_parser("serve", help="Run the local tablet-capture app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--database", type=Path, default=Path("data/local/sessions.sqlite3"))
    serve.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    serve.add_argument(
        "--voice-model", type=Path, default=Path("artifacts/voice/model.joblib")
    )

    subcommands.add_parser("datasets", help="List online datasets and access conditions")

    newhandpd_download = subcommands.add_parser(
        "download-newhandpd", help="Download the official NewHandPD spiral-image subset"
    )
    newhandpd_download.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/newhandpd")
    )
    newhandpd_download.add_argument("--force", action="store_true")

    newhandpd_benchmark = subcommands.add_parser(
        "benchmark-newhandpd", help="Run subject-grouped NewHandPD feature evaluation"
    )
    newhandpd_benchmark.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/newhandpd")
    )
    newhandpd_benchmark.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/newhandpd")
    )
    newhandpd_benchmark.add_argument("--download", action="store_true")

    voice_download = subcommands.add_parser(
        "download-voice", help="Download labeled Figshare sustained-vowel WAV files"
    )
    voice_download.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/figshare_voice")
    )
    voice_download.add_argument("--force", action="store_true")

    voice_train = subcommands.add_parser(
        "train-voice", help="Train the participant-level raw-audio voice baseline"
    )
    voice_train.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/figshare_voice")
    )
    voice_train.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/voice")
    )
    voice_train.add_argument("--download", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "download":
        location = download_uci(args.data_dir, force=args.force)
        print(f"Dataset available at {location}")
    elif args.command == "train":
        if args.download:
            download_uci(args.data_dir)
        table = build_participant_table(args.data_dir)
        metrics = train_and_evaluate(table, args.output_dir)
        print(json.dumps(metrics, indent=2))
    elif args.command == "predict":
        score = predict_file(args.model, args.trajectory)
        print(
            json.dumps(
                {
                    "experimental_screening_score": score,
                    "warning": (
                        "Not a diagnosis or population-calibrated probability. "
                        "A clinician must assess concerning symptoms."
                    ),
                },
                indent=2,
            )
        )
    elif args.command == "serve":
        import uvicorn

        from .app import create_app

        uvicorn.run(
            create_app(
                database_path=args.database,
                model_path=args.model,
                voice_model_path=args.voice_model,
            ),
            host=args.host,
            port=args.port,
        )
    elif args.command == "datasets":
        print(json.dumps(DATASET_CATALOG, indent=2))
    elif args.command == "download-newhandpd":
        location = download_newhandpd(args.data_dir, force=args.force)
        print(f"NewHandPD available at {location}")
    elif args.command == "benchmark-newhandpd":
        if args.download:
            download_newhandpd(args.data_dir)
        metrics = benchmark_newhandpd(args.data_dir, args.output_dir)
        print(json.dumps(metrics, indent=2))
    elif args.command == "download-voice":
        location = download_figshare_voice(args.data_dir, force=args.force)
        print(f"Figshare voice dataset available at {location}")
    elif args.command == "train-voice":
        if args.download or len(list(args.data_dir.rglob("*.wav"))) != 81:
            download_figshare_voice(args.data_dir)
        metrics = train_voice_model(read_figshare_voice(args.data_dir), args.output_dir)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

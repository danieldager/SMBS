"""SMBS CLI — Speech Model Benchmarking Suite.

Unified command-line interface for all SMBS workflows.

Usage:
    smbs encode   --encoder spidr_base --dataset chunks30
    smbs train    --encoder spidr_base --arch gpt2
    smbs evaluate --encoder spidr_base --model gpt2_e768_l12_h12_feb12
    smbs prepare-swuggy --encoder spidr_base --parquet-pattern '/path/*.parquet'
    smbs plots
    smbs grid     --encoder spidr_base
    smbs scan     /path/to/audio/dataset
    smbs vad-pyannote --manifest chunks5
    smbs vad-ten     --manifest chunks5
"""

import argparse
import sys


def add_encode_parser(subparsers):
    p = subparsers.add_parser("encode", help="Encode audio files to discrete tokens")
    p.add_argument("--encoder", required=True, help="Encoder name (e.g. spidr_base, mhubert, hubert-500)")
    p.add_argument("--dataset", required=True, help="Dataset name or manifest path (auto-resolves from manifests/)")
    p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    p.add_argument("--task-id", type=int, default=0, help="SLURM array task ID")
    p.add_argument("--num-tasks", type=int, default=1, help="Total SLURM array tasks")


def add_train_parser(subparsers):
    p = subparsers.add_parser("train", help="Train a language model on encoded tokens")
    p.add_argument("--encoder", required=True, help="Encoder name")
    p.add_argument("--arch", default="gpt2", choices=["gpt2", "lstm"], help="Architecture (default: gpt2)")
    p.add_argument("--train-dataset", default=None, help="Training data name (default: chunks30)")
    p.add_argument("--eval-dataset", default=None, help="Eval data name (default: chunks-eval)")


def add_evaluate_parser(subparsers):
    p = subparsers.add_parser("evaluate", help="Evaluate a model on sWuggy benchmark")
    p.add_argument("--encoder", required=True, help="Encoder name")
    p.add_argument("--model", required=True, help="Model directory name under weights/<encoder>/")
    p.add_argument("--dataset", default="swuggy", help="Evaluation dataset name (default: swuggy)")
    p.add_argument("--force", action="store_true", help="Re-score even if results exist")


def add_prepare_swuggy_parser(subparsers):
    p = subparsers.add_parser("prepare-swuggy", help="Prepare and encode sWuggy benchmark data")
    p.add_argument("--encoder", required=True, help="Encoder name")
    p.add_argument("--parquet-pattern", required=True, help="Glob pattern for raw sWuggy parquet files")
    p.add_argument("--device", default="cuda", help="Device (default: cuda)")


def add_plots_parser(subparsers):
    p = subparsers.add_parser("plots", help="Generate evaluation result plots")
    p.add_argument("--raw", action="store_true", help="Use raw accuracy instead of per-voice")


def add_grid_parser(subparsers):
    p = subparsers.add_parser("grid", help="Run LSTM hyperparameter grid search")
    p.add_argument("--encoder", default="spidr_base", help="Encoder name (default: spidr_base)")
    p.add_argument("--max-steps", type=int, default=None, help="Max training steps per config")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds")
    p.add_argument("--train-dataset", default="chunks30", help="Training data name")
    p.add_argument("--eval-dataset", default="chunks-eval", help="Eval data name")


def add_scan_parser(subparsers):
    p = subparsers.add_parser("scan", help="Scan a directory for audio files and create a manifest")
    p.add_argument("directory", help="Root directory to scan")
    p.add_argument("--workers", type=int, default=None, help="Parallel workers (default: auto)")


def add_vad_pyannote_parser(subparsers):
    p = subparsers.add_parser("vad-pyannote", help="Run PyAnnote GPU voice activity detection")
    p.add_argument("--manifest", required=True, help="Manifest name or path")
    p.add_argument("--array-id", type=int, default=0, help="SLURM array task ID")
    p.add_argument("--array-count", type=int, default=1, help="Total SLURM array tasks")


def add_vad_ten_parser(subparsers):
    p = subparsers.add_parser("vad-ten", help="Run TenVAD CPU voice activity detection")
    p.add_argument("--manifest", required=True, help="Manifest name or path")
    p.add_argument("--hop-size", type=int, default=256, help="Frame hop size (default: 256)")
    p.add_argument("--threshold", type=float, default=0.5, help="VAD threshold (default: 0.5)")
    p.add_argument("--workers", type=int, default=None, help="Parallel workers (default: all CPUs)")


def main():
    parser = argparse.ArgumentParser(
        prog="smbs",
        description="Speech Model Benchmarking Suite",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_encode_parser(subparsers)
    add_train_parser(subparsers)
    add_evaluate_parser(subparsers)
    add_prepare_swuggy_parser(subparsers)
    add_plots_parser(subparsers)
    add_grid_parser(subparsers)
    add_scan_parser(subparsers)
    add_vad_pyannote_parser(subparsers)
    add_vad_ten_parser(subparsers)

    args = parser.parse_args()

    if args.command == "encode":
        from smbs.utils.manifest import resolve_manifest
        from smbs.encode.run import run_encode

        manifest_path = str(resolve_manifest(args.dataset))
        run_encode(
            encoder_name=args.encoder,
            dataset=args.dataset,
            manifest_path=manifest_path,
            device=args.device,
            task_id=args.task_id,
            num_tasks=args.num_tasks,
        )

    elif args.command == "train":
        from smbs.train.run import run_train

        run_train(
            encoder=args.encoder,
            arch=args.arch,
            train_dataset=args.train_dataset,
            eval_dataset=args.eval_dataset,
        )

    elif args.command == "evaluate":
        from smbs.evaluate.swuggy import run_evaluate

        run_evaluate(
            encoder=args.encoder,
            model=args.model,
            dataset=args.dataset,
            force=args.force,
        )

    elif args.command == "prepare-swuggy":
        from smbs.evaluate.swuggy import prepare_swuggy

        prepare_swuggy(
            raw_parquet_pattern=args.parquet_pattern,
            encoder_name=args.encoder,
            device=args.device,
        )

    elif args.command == "plots":
        from smbs.evaluate.plots import create_unified_plot

        create_unified_plot(use_raw=args.raw)

    elif args.command == "grid":
        from smbs.train.grid import run_grid

        kwargs = {
            "encoder": args.encoder,
            "train_dataset": args.train_dataset,
            "eval_dataset": args.eval_dataset,
        }
        if args.max_steps is not None:
            kwargs["max_steps"] = args.max_steps
        if args.seeds is not None:
            kwargs["seeds"] = args.seeds
        run_grid(**kwargs)

    elif args.command == "scan":
        from smbs.scan import run_scan

        run_scan(dataset_dir=args.directory, workers=args.workers)

    elif args.command == "vad-pyannote":
        from smbs.utils.manifest import resolve_manifest
        from smbs.vad.pyannote import run_pyannote

        manifest_path = str(resolve_manifest(args.manifest))
        run_pyannote(
            manifest=manifest_path,
            array_id=args.array_id,
            array_count=args.array_count,
        )

    elif args.command == "vad-ten":
        from smbs.utils.manifest import resolve_manifest
        from smbs.vad.tenvad import run_tenvad

        manifest_path = str(resolve_manifest(args.manifest))
        run_tenvad(
            manifest=manifest_path,
            hop_size=args.hop_size,
            threshold=args.threshold,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()

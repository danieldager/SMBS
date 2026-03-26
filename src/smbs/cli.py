"""SMBS CLI — Speech Model Benchmarking Suite.

SLURM job launcher. Each command submits the appropriate SLURM script via sbatch.
Use --local on any GPU/CPU command to bypass SLURM and run directly (e.g. on an
interactive node).

Usage:
    smbs encode   --encoder spidr_base --dataset chunks30
    smbs train    --encoder spidr_base --arch gpt2
    smbs evaluate --encoder spidr_base --model gpt2_e768_l12_h12_feb12
    smbs plots
    ...

All commands except 'plots' submit SLURM jobs by default.
Override SLURM defaults with -p/--partition and --time.
"""

import argparse
import subprocess
import sys
from pathlib import Path

SLURM_DIR = Path(__file__).resolve().parent.parent.parent / "slurm"


# ── SLURM helpers ────────────────────────────────────────────────────────


def _sbatch(script: str, script_args: list[str], flags: list[str] | None = None) -> None:
    """Submit a SLURM job and exit with its return code."""
    path = SLURM_DIR / script
    if not path.exists():
        print(f"Error: SLURM script not found: {path}", file=sys.stderr)
        sys.exit(1)

    cmd = ["sbatch", *(flags or []), str(path), *script_args]
    print(f"$ {' '.join(cmd)}")
    sys.exit(subprocess.run(cmd).returncode)


def _slurm_flags(args: argparse.Namespace) -> list[str]:
    """Collect common sbatch overrides from parsed args."""
    flags: list[str] = []
    if getattr(args, "partition", None):
        flags.append(f"--partition={args.partition}")
    if getattr(args, "time", None):
        flags.append(f"--time={args.time}")
    return flags


def _add_slurm_args(
    p: argparse.ArgumentParser,
    *,
    has_array: bool = False,
    has_gpus: bool = False,
) -> None:
    """Add --local and common SLURM overrides to a subparser."""
    p.add_argument("--local", action="store_true",
                    help="Run directly instead of submitting a SLURM job")
    p.add_argument("-p", "--partition", default=None,
                    help="Override SLURM partition")
    p.add_argument("--time", default=None,
                    help="Override SLURM time limit (e.g. 24:00:00)")
    if has_array:
        p.add_argument("--array", default=None,
                        help="Override SLURM array spec (e.g. 0-4)")
    if has_gpus:
        p.add_argument("--gpus", type=int, default=None,
                        help="Override number of GPUs")


# ── Subcommand handlers ─────────────────────────────────────────────────


def cmd_encode(args: argparse.Namespace) -> None:
    if args.local:
        from smbs.encode.run import run_encode
        from smbs.utils.manifest import resolve_manifest

        manifest_path = str(resolve_manifest(args.dataset))
        run_encode(args.encoder, args.dataset, manifest_path,
                   args.device, args.task_id, args.num_tasks)
    else:
        flags = _slurm_flags(args)
        if args.array:
            flags.append(f"--array={args.array}")
        _sbatch("encode.slurm", [args.encoder, args.dataset], flags)


def cmd_train(args: argparse.Namespace) -> None:
    if args.local:
        from smbs.train.run import run_train

        run_train(args.encoder, args.arch, args.train_dataset, args.eval_dataset)
    else:
        flags = _slurm_flags(args)
        if args.gpus:
            flags.append(f"--gres=gpu:{args.gpus}")
        _sbatch("train.slurm", [args.encoder, args.arch], flags)


def cmd_evaluate(args: argparse.Namespace) -> None:
    if args.local:
        from smbs.evaluate.swuggy import run_evaluate

        run_evaluate(args.encoder, args.model, args.dataset, args.force)
    else:
        flags = _slurm_flags(args)
        _sbatch("evaluate.slurm", ["evaluate", args.encoder, args.model], flags)


def cmd_prepare_swuggy(args: argparse.Namespace) -> None:
    if args.local:
        from smbs.evaluate.swuggy import prepare_swuggy

        prepare_swuggy(args.parquet_pattern, args.encoder, args.device)
    else:
        flags = _slurm_flags(args)
        _sbatch("evaluate.slurm", ["prepare", args.encoder, args.parquet_pattern], flags)


def cmd_plots(args: argparse.Namespace) -> None:
    # Always local — lightweight CPU-only task
    from smbs.evaluate.plots import create_unified_plot

    create_unified_plot(use_raw=args.raw)


def cmd_grid(args: argparse.Namespace) -> None:
    if args.local:
        from smbs.train.grid import run_grid

        kwargs: dict = {
            "encoder": args.encoder,
            "train_dataset": args.train_dataset,
            "eval_dataset": args.eval_dataset,
        }
        if args.max_steps is not None:
            kwargs["max_steps"] = args.max_steps
        if args.seeds is not None:
            kwargs["seeds"] = args.seeds
        run_grid(**kwargs)
    else:
        flags = _slurm_flags(args)
        _sbatch("grid.slurm", [args.encoder], flags)


def cmd_scan(args: argparse.Namespace) -> None:
    if args.local:
        from smbs.scan import run_scan

        run_scan(args.directory, args.workers)
    else:
        flags = _slurm_flags(args)
        _sbatch("scan.slurm", [args.directory], flags)



def cmd_vad_ten(args: argparse.Namespace) -> None:
    if args.local:
        from smbs.utils.manifest import resolve_manifest
        from smbs.vad.tenvad import run_tenvad

        manifest_path = str(resolve_manifest(args.manifest))
        run_tenvad(manifest_path, args.hop_size, args.threshold, args.workers)
    else:
        flags = _slurm_flags(args)
        _sbatch("vad_ten.slurm", [args.manifest], flags)


# ── CLI entry point ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="smbs",
        description="Speech Model Benchmarking Suite — SLURM job launcher",
        epilog="Use --local on any command to run directly on the current node.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # encode
    p = sub.add_parser("encode", help="Encode audio → tokens [GPU]")
    p.add_argument("--encoder", required=True)
    p.add_argument("--dataset", required=True, help="Dataset name (→ manifests/<name>)")
    p.add_argument("--device", default="cuda", help="Device for --local mode")
    p.add_argument("--task-id", type=int, default=0, help="Array task ID for --local mode")
    p.add_argument("--num-tasks", type=int, default=1, help="Array task count for --local mode")
    _add_slurm_args(p, has_array=True)
    p.set_defaults(func=cmd_encode)

    # train
    p = sub.add_parser("train", help="Train language model [multi-GPU]")
    p.add_argument("--encoder", required=True)
    p.add_argument("--arch", default="gpt2", choices=["gpt2", "lstm"])
    p.add_argument("--train-dataset", default=None)
    p.add_argument("--eval-dataset", default=None)
    _add_slurm_args(p, has_gpus=True)
    p.set_defaults(func=cmd_train)

    # evaluate
    p = sub.add_parser("evaluate", help="Evaluate on sWuggy [GPU]")
    p.add_argument("--encoder", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="swuggy")
    p.add_argument("--force", action="store_true")
    _add_slurm_args(p)
    p.set_defaults(func=cmd_evaluate)

    # prepare-swuggy
    p = sub.add_parser("prepare-swuggy", help="Encode sWuggy benchmark audio [GPU]")
    p.add_argument("--encoder", required=True)
    p.add_argument("--parquet-pattern", required=True)
    p.add_argument("--device", default="cuda")
    _add_slurm_args(p)
    p.set_defaults(func=cmd_prepare_swuggy)

    # plots (always local)
    p = sub.add_parser("plots", help="Generate result plots [local, no SLURM]")
    p.add_argument("--raw", action="store_true", help="Use raw accuracy")
    p.set_defaults(func=cmd_plots)

    # grid
    p = sub.add_parser("grid", help="LSTM hyperparameter grid search [GPU]")
    p.add_argument("--encoder", default="spidr_base")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--train-dataset", default="chunks30")
    p.add_argument("--eval-dataset", default="chunks-eval")
    _add_slurm_args(p)
    p.set_defaults(func=cmd_grid)

    # scan
    p = sub.add_parser("scan", help="Scan directory for audio files [CPU]")
    p.add_argument("directory")
    p.add_argument("--workers", type=int, default=None)
    _add_slurm_args(p)
    p.set_defaults(func=cmd_scan)

    # vad
    p = sub.add_parser("vad", help="TenVAD voice activity detection [CPU]")
    p.add_argument("--manifest", required=True)
    p.add_argument("--hop-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=None)
    _add_slurm_args(p)
    p.set_defaults(func=cmd_vad_ten)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table


def run_step(name: str, cmd: list[str], cwd: Path) -> None:
    console = Console()
    console.rule(f"[bold cyan]{name}")
    console.print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}")


def newest_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files match {pattern!r} under {root}")
    return matches[0]


def base_train_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "train.py",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--feature-set",
        args.sequence_feature_set,
        "--val-target-from-end",
        str(args.val_target_from_end),
        "--action-loss-weight",
        str(args.action_loss_weight),
        "--point-loss-weight",
        str(args.point_loss_weight),
        "--rally-loss-weight",
        str(args.rally_loss_weight),
        "--selection-metric",
        args.selection_metric,
        "--class-weight-mode",
        args.class_weight_mode,
        "--label-smoothing",
        str(args.label_smoothing),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
    ]
    if args.no_plots:
        cmd.append("--no-plots")
    if args.limit_rallies > 0:
        cmd.extend(["--limit-rallies", str(args.limit_rallies)])
    if args.cv_folds > 1:
        cmd.extend(["--cv-folds", str(args.cv_folds)])
    return cmd


def transformer_cmd(args: argparse.Namespace, out_dir: Path) -> list[str]:
    cmd = base_train_args(args)
    cmd.extend(
        [
            "--model-type",
            "transformer",
            "--d-model",
            str(args.transformer_d_model),
            "--layers",
            str(args.transformer_layers),
            "--ffn",
            str(args.transformer_ffn),
            "--out-dir",
            str(out_dir),
        ]
    )
    return cmd


def lstm_cmd(args: argparse.Namespace, out_dir: Path) -> list[str]:
    cmd = base_train_args(args)
    cmd.extend(
        [
            "--model-type",
            "lstm",
            "--d-model",
            str(args.lstm_d_model),
            "--layers",
            str(args.lstm_layers),
            "--out-dir",
            str(out_dir),
        ]
    )
    return cmd


def lightgbm_cmd(args: argparse.Namespace, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "train_tabular.py",
        "--backend",
        "lightgbm",
        "--feature-set",
        args.tabular_feature_set,
        "--window",
        str(args.tabular_window),
        "--max-iter",
        str(args.tabular_max_iter),
        "--val-target-from-end",
        str(args.val_target_from_end),
        "--out-dir",
        str(out_dir),
    ]
    if args.limit_rallies > 0:
        cmd.extend(["--limit-rallies", str(args.limit_rallies)])
    if args.cv_folds > 1:
        cmd.extend(["--cv-folds", str(args.cv_folds)])
    return cmd


def collect_artifacts(transformer_dir: Path, lstm_dir: Path, tabular_dir: Path) -> list[Path]:
    artifacts: list[Path] = []
    artifacts.extend(sorted(transformer_dir.rglob("best_transformer_*.pt")))
    artifacts.extend(sorted(lstm_dir.rglob("best_transformer_*.pt")))
    artifacts.extend(sorted(tabular_dir.rglob("tabular_lightgbm_*.pkl")))
    if not artifacts:
        raise FileNotFoundError("No artifacts were produced by the pipeline.")
    return artifacts


def validate_cmd(args: argparse.Namespace, artifacts: list[Path]) -> list[str]:
    cmd = [
        sys.executable,
        "validate_ensemble.py",
        "--device",
        args.device,
        "--val-target-from-end",
        str(args.val_target_from_end),
        "--checkpoint",
        *[str(p) for p in artifacts],
    ]
    if args.limit_rallies > 0:
        cmd.extend(["--limit-rallies", str(args.limit_rallies)])
    return cmd


def inference_cmd(args: argparse.Namespace, artifacts: list[Path], out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "inference.py",
        "--device",
        args.device,
        "--checkpoint",
        *[str(p) for p in artifacts],
        "--out-dir",
        str(out_dir),
    ]
    return cmd


def write_manifest(path: Path, args: argparse.Namespace, artifacts: list[Path], run_dir: Path) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "artifacts": [str(p) for p in artifacts],
        "args": vars(args),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    cwd = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else cwd / "runs" / f"full_pipeline_{stamp}"
    transformer_dir = run_dir / "checkpoints_transformer"
    lstm_dir = run_dir / "checkpoints_lstm"
    tabular_dir = run_dir / "checkpoints_lightgbm"
    submission_dir = run_dir / "submissions"
    run_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    table = Table(title="Full pipeline")
    table.add_column("item")
    table.add_column("value")
    for k, v in {
        "run dir": run_dir,
        "device": args.device,
        "epochs": args.epochs,
        "batch": args.batch,
        "cv folds": args.cv_folds,
        "sequence features": args.sequence_feature_set,
        "tabular features": args.tabular_feature_set,
        "val target from end": args.val_target_from_end,
    }.items():
        table.add_row(str(k), str(v))
    console.print(table)

    if not args.skip_transformer:
        run_step("Train Transformer", transformer_cmd(args, transformer_dir), cwd)
    if not args.skip_lstm:
        run_step("Train LSTM", lstm_cmd(args, lstm_dir), cwd)
    if not args.skip_lightgbm:
        run_step("Train LightGBM", lightgbm_cmd(args, tabular_dir), cwd)

    artifacts = collect_artifacts(transformer_dir, lstm_dir, tabular_dir)
    write_manifest(run_dir / "manifest.json", args, artifacts, run_dir)

    console.print("[bold green]Artifacts selected for ensemble:[/]")
    for artifact in artifacts:
        console.print(f"  {artifact}")

    if not args.skip_validation:
        run_step("Validate Mixed Ensemble", validate_cmd(args, artifacts), cwd)
    if not args.skip_inference:
        run_step("Infer Test Submission", inference_cmd(args, artifacts, submission_dir), cwd)
        submission = newest_file(submission_dir, "submission_transformer_*.csv")
        console.print(f"[bold green]Submission:[/] {submission}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer, LSTM, LightGBM, validate ensemble, and infer test submission.")
    parser.add_argument("--run-dir", default="", help="Output run directory. Defaults to runs/full_pipeline_<timestamp>.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--cv-folds", type=int, default=1)
    parser.add_argument("--limit-rallies", type=int, default=0)
    parser.add_argument("--val-target-from-end", type=int, default=2)
    parser.add_argument("--sequence-feature-set", choices=("base", "score", "enhanced"), default="base")
    parser.add_argument("--tabular-feature-set", choices=("base", "score", "enhanced"), default="enhanced")
    parser.add_argument("--action-loss-weight", type=float, default=0.35)
    parser.add_argument("--point-loss-weight", type=float, default=0.55)
    parser.add_argument("--rally-loss-weight", type=float, default=0.10)
    parser.add_argument("--selection-metric", choices=("score", "action_point_score", "min_f1"), default="action_point_score")
    parser.add_argument("--class-weight-mode", choices=("none", "inverse", "sqrt", "effective"), default="sqrt")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--transformer-d-model", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-ffn", type=int, default=512)
    parser.add_argument("--lstm-d-model", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--tabular-window", type=int, default=8)
    parser.add_argument("--tabular-max-iter", type=int, default=1000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-lightgbm", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    main(parser.parse_args())

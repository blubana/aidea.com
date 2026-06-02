"""Interactive one-stop runner for the training and submission pipeline.

The runner is GPU-first for components that support GPU acceleration:

- PyTorch LSTM: CUDA profile defaults to `candidate` (BF16 AMP + TF32 + cuDNN benchmark).
- CatBoost: defaults to `--task-type GPU`.

Some current ensemble steps still use scikit-learn estimators
(`train_tabular_baseline.py`, phase models, server/point stacking). They run on
CPU by design today; keep them enabled for the full final blend, or disable CPU
extras to run only the GPU-backed core stages.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ["actionId", "pointId", "serverGetPoint"]


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    outputs: tuple[Path, ...] = ()
    gpu_backed: bool = False
    cpu_sklearn: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full aidea.com modeling pipeline.")
    parser.add_argument("--yes", action="store_true", help="Run non-interactively with defaults.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip steps whose expected outputs already exist.")
    parser.add_argument("--gpu-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-cpu-extras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-prefix-len", type=int, default=12)
    parser.add_argument("--max-len", type=int, default=12)
    parser.add_argument("--lstm-profile", default="candidate")
    parser.add_argument("--catboost-task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--catboost-iterations", type=int, default=800)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, choices=DEFAULT_TARGETS)
    parser.add_argument("--sample", type=int, default=0, help="Optional debug sample size for train scripts; 0 means full data.")
    parser.add_argument("--server-output", default="float", choices=["float", "bool"])
    parser.add_argument("--submission-path", default="submissions/submission_final_blend_brandnew.csv")
    parser.add_argument("--use-cross-target-stacking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-action-phase", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def prompt_bool(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    answer = input(f"{label} [{suffix}]: ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes", "1", "true", "t"}


def prompt_text(label: str, default: str) -> str:
    answer = input(f"{label} [{default}]: ").strip()
    return answer or default


def maybe_interactive(args: argparse.Namespace) -> argparse.Namespace:
    if args.yes:
        return args
    print("GPU-first pipeline runner")
    print("PyTorch LSTM and CatBoost can run on GPU; sklearn ensemble extras are CPU-based.")
    args.gpu_first = prompt_bool("Prefer GPU-backed settings", args.gpu_first)
    args.include_cpu_extras = prompt_bool("Include CPU sklearn extras required for final blend", args.include_cpu_extras)
    args.skip_existing = prompt_bool("Skip steps with existing outputs", args.skip_existing)
    args.lstm_profile = prompt_text("LSTM CUDA profile", args.lstm_profile)
    args.catboost_task_type = prompt_text("CatBoost task type (GPU/CPU)", args.catboost_task_type).upper()
    args.submission_path = prompt_text("Submission output path", args.submission_path)
    return args


def py(script: str, *extra: str) -> list[str]:
    return [sys.executable, script, *extra]


def sample_args(args: argparse.Namespace) -> list[str]:
    return ["--sample", str(args.sample)] if args.sample > 0 else []


def build_steps(args: argparse.Namespace) -> list[Step]:
    steps: list[Step] = [
        Step(
            "build_prefix_dataset",
            py("src/build_prefix_dataset.py", "--max-prefix-len", str(args.max_prefix_len)),
            (Path("data/processed/prefix_train_features.csv"), Path("data/processed/prefix_test_features.csv")),
        ),
        Step(
            "build_lstm_dataset",
            py("src/lstm_dataset.py", "--max-len", str(args.max_len)),
            (Path("data/processed/lstm_dataset.npz"),),
        ),
    ]

    for target in args.targets:
        steps.append(
            Step(
                f"train_lstm_{target}",
                py("src/train_lstm.py", "--target", target, "--cuda-opt-profile", args.lstm_profile, *sample_args(args)),
                (Path(f"reports/lstm/{target}_metrics.json"), Path(f"reports/lstm/{target}_oof_proba.npy")),
                gpu_backed=True,
            )
        )

    if args.include_cpu_extras:
        steps.append(
            Step(
                "train_tabular_baseline_cpu",
                py("src/train_tabular_baseline.py", "--targets", *args.targets, *sample_args(args)),
                tuple(Path(f"reports/tabular_baseline/{target}_oof_proba.npy") for target in args.targets),
                cpu_sklearn=True,
            )
        )

    steps.append(
        Step(
            "train_catboost_gpu" if args.catboost_task_type == "GPU" else "train_catboost_cpu",
            py(
                "src/train_catboost_baseline.py",
                "--targets",
                *args.targets,
                "--task-type",
                args.catboost_task_type,
                "--iterations",
                str(args.catboost_iterations),
                *sample_args(args),
            ),
            tuple(Path(f"reports/catboost/{target}_test_proba.npy") for target in args.targets),
            gpu_backed=args.catboost_task_type == "GPU",
        )
    )

    if args.include_cpu_extras:
        steps.extend(
            [
                Step(
                    "train_action_phase_cpu",
                    py("src/train_action_phase_models.py", *sample_args(args)),
                    (Path("reports/action_phase/actionId_oof_proba.npy"),),
                    cpu_sklearn=True,
                ),
                Step(
                    "train_point_phase_cpu",
                    py("src/train_point_phase_models.py", *sample_args(args)),
                    (Path("reports/point_phase/pointId_oof_proba.npy"),),
                    cpu_sklearn=True,
                ),
                Step(
                    "train_server_stacking_cpu",
                    py("src/train_server_stacking.py", *sample_args(args)),
                    (Path("models/server_stacking/serverGetPoint_extratrees.joblib"),),
                    cpu_sklearn=True,
                ),
                Step(
                    "train_point_stacking_cpu",
                    py("src/train_point_stacking.py", *sample_args(args)),
                    (Path("models/point_stacking/pointId_extratrees.joblib"),),
                    cpu_sklearn=True,
                ),
                Step(
                    "predict_final_blend",
                    py(
                        "src/predict_final_blend.py",
                        "--submission-path",
                        args.submission_path,
                        "--server-output",
                        args.server_output,
                        "--use-cross-target-stacking" if args.use_cross_target_stacking else "--no-use-cross-target-stacking",
                        "--use-action-phase" if args.use_action_phase else "--no-use-action-phase",
                    ),
                    (Path(args.submission_path),),
                ),
            ]
        )
    return steps


def should_skip(step: Step, args: argparse.Namespace) -> bool:
    return args.skip_existing and bool(step.outputs) and all((ROOT / path).exists() for path in step.outputs)


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_step(step: Step, args: argparse.Namespace) -> None:
    tag = "GPU" if step.gpu_backed else "CPU/sklearn" if step.cpu_sklearn else "pipeline"
    print(f"\n=== {step.name} ({tag}) ===")
    print(format_command(step.command))
    if should_skip(step, args):
        print("skip: expected outputs already exist")
        return
    if args.dry_run:
        return
    subprocess.run(step.command, cwd=ROOT, check=True)


def main() -> None:
    args = maybe_interactive(parse_args())
    if args.gpu_first:
        args.lstm_profile = args.lstm_profile or "candidate"
        args.catboost_task_type = args.catboost_task_type or "GPU"
    steps = build_steps(args)
    if not args.include_cpu_extras:
        print("CPU sklearn extras disabled: final blend prediction will not run in this mode.")
    for step in steps:
        run_step(step, args)


if __name__ == "__main__":
    main()

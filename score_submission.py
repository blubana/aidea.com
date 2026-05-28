from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import f1_score, roc_auc_score


REQUIRED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
WEIGHTS = {
    "f1_action": 0.4,
    "f1_point": 0.4,
    "auc_server_get_point": 0.2,
}


def _read_required_csv(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    return df[REQUIRED_COLUMNS].copy()


def score_submission(prediction: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    merged = truth.merge(
        prediction,
        on="rally_uid",
        how="inner",
        suffixes=("_true", "_pred"),
        validate="one_to_one",
    )
    if len(merged) != len(truth):
        missing = sorted(set(truth["rally_uid"]) - set(prediction["rally_uid"]))
        preview = missing[:10]
        raise ValueError(
            f"Prediction covers {len(merged)}/{len(truth)} truth rows; missing rally_uid examples: {preview}"
        )

    f1_action = f1_score(
        merged["actionId_true"],
        merged["actionId_pred"],
        average="macro",
        zero_division=0,
    )
    f1_point = f1_score(
        merged["pointId_true"],
        merged["pointId_pred"],
        average="macro",
        zero_division=0,
    )
    y_true = merged["serverGetPoint_true"]
    y_score = merged["serverGetPoint_pred"].clip(0.0, 1.0)
    auc = roc_auc_score(y_true, y_score) if y_true.nunique() > 1 else 0.5
    overall = (
        WEIGHTS["f1_action"] * f1_action
        + WEIGHTS["f1_point"] * f1_point
        + WEIGHTS["auc_server_get_point"] * auc
    )
    return {
        "rows": float(len(merged)),
        "f1_action": float(f1_action),
        "f1_point": float(f1_point),
        "auc_server_get_point": float(auc),
        "overall_score": float(overall),
    }


def print_scores(scores: dict[str, float]) -> None:
    console = Console()
    table = Table(title="Submission score")
    table.add_column("metric")
    table.add_column("weight")
    table.add_column("score", justify="right")
    table.add_row("ActionId Macro F1", "0.4", f"{scores['f1_action']:.6f}")
    table.add_row("PointId Macro F1", "0.4", f"{scores['f1_point']:.6f}")
    table.add_row("ServerGetPoint AUC", "0.2", f"{scores['auc_server_get_point']:.6f}")
    table.add_row("Overall Score", "", f"[bold]{scores['overall_score']:.6f}[/]")
    console.print(table)
    console.print(f"Rows scored: {int(scores['rows'])}")


def main(args: argparse.Namespace) -> None:
    prediction = _read_required_csv(Path(args.pred), "prediction")
    truth = _read_required_csv(Path(args.truth), "truth")
    scores = score_submission(prediction, truth)
    print_scores(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score AI CUP rally submission CSV against a labeled truth CSV.")
    parser.add_argument("--pred", required=True, help="Path to submission/prediction CSV.")
    parser.add_argument("--truth", required=True, help="Path to labeled ground-truth CSV with the same required columns.")
    main(parser.parse_args())

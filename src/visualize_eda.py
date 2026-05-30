"""Create EDA plots from reports/*.csv outputs.

Run after `src/eda_dataset.py`:

    uv run python src/visualize_eda.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, rotation: int = 0) -> None:
    plt.figure(figsize=(11, 5))
    sns.barplot(data=df, x=x, y=y, color="#4C72B0")
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--output-dir", default="reports/figures")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    sns.set_theme(style="whitegrid")

    plots = [
        ("train_rally_length_distribution.csv", "rally_len", "count", "Train rally length distribution", "train_rally_length.png"),
        ("test_prefix_length_distribution.csv", "prefix_len", "count", "Test prefix length distribution", "test_prefix_length.png"),
        ("train_actionId_distribution.csv", "actionId", "count", "Train actionId distribution", "train_actionId.png"),
        ("train_pointId_distribution.csv", "pointId", "count", "Train pointId distribution", "train_pointId.png"),
        (
            "train_serverGetPoint_distribution.csv",
            "serverGetPoint",
            "count",
            "Train serverGetPoint distribution",
            "train_serverGetPoint.png",
        ),
    ]

    for csv_name, x, y, title, png_name in plots:
        path = report_dir / csv_name
        if path.exists():
            df = pd.read_csv(path)
            save_bar(df, x, y, title, output_dir / png_name)

    for csv_name, title, png_name in [
        ("action_by_next_strikeNumber.csv", "Top actionId ratio by next strike number", "action_by_next_strike.png"),
        ("point_by_next_strikeNumber.csv", "Top pointId ratio by next strike number", "point_by_next_strike.png"),
    ]:
        path = report_dir / csv_name
        if path.exists():
            df = pd.read_csv(path)
            plt.figure(figsize=(11, 5))
            sns.lineplot(data=df, x="next_strikeNumber", y="top_label_ratio", marker="o")
            plt.title(title)
            plt.ylim(0, 1)
            plt.tight_layout()
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / png_name, dpi=150)
            plt.close()

    print(f"Wrote EDA figures to {output_dir}")


if __name__ == "__main__":
    main()

"""Dataset EDA for the rally prediction project.

Writes human-readable summaries and small CSV distribution tables to reports/.
Requires only Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from features import group_by_rally, read_csv_rows, to_int, write_csv_rows


def percentile(values, q):
    if not values:
        return 0
    xs = sorted(values)
    idx = round((len(xs) - 1) * q)
    return xs[idx]


def counter_rows(counter: Counter, key_name: str, value_name: str = "count"):
    def sort_key(item):
        key, _ = item
        try:
            return int(key)
        except (TypeError, ValueError):
            return key

    return [{key_name: k, value_name: v} for k, v in sorted(counter.items(), key=sort_key)]


def rally_length_summary(groups):
    lengths = [len(rows) for rows in groups.values()]
    return {
        "rallies": len(lengths),
        "min": min(lengths) if lengths else 0,
        "median": median(lengths) if lengths else 0,
        "mean": round(mean(lengths), 3) if lengths else 0,
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "max": max(lengths) if lengths else 0,
    }


def label_by_next_strike(train_groups, label_col):
    table = defaultdict(Counter)
    for rows in train_groups.values():
        for row in rows[1:]:
            table[to_int(row["strikeNumber"])][row[label_col]] += 1
    out = []
    for strike in sorted(table):
        total = sum(table[strike].values())
        top_label, top_count = table[strike].most_common(1)[0]
        out.append(
            {
                "next_strikeNumber": strike,
                "samples": total,
                "top_label": top_label,
                "top_label_count": top_count,
                "top_label_ratio": round(top_count / total, 6),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    train = read_csv_rows(dataset_dir / "train.csv")
    test = read_csv_rows(dataset_dir / "test_new.csv")
    train_groups = group_by_rally(train)
    test_groups = group_by_rally(test)

    train_len_counts = Counter(len(rows) for rows in train_groups.values())
    test_len_counts = Counter(len(rows) for rows in test_groups.values())

    write_csv_rows(report_dir / "train_rally_length_distribution.csv", counter_rows(train_len_counts, "rally_len"))
    write_csv_rows(report_dir / "test_prefix_length_distribution.csv", counter_rows(test_len_counts, "prefix_len"))

    for col in ["actionId", "pointId", "serverGetPoint"]:
        if col in train[0]:
            write_csv_rows(report_dir / f"train_{col}_distribution.csv", counter_rows(Counter(row[col] for row in train), col))

    write_csv_rows(report_dir / "action_by_next_strikeNumber.csv", label_by_next_strike(train_groups, "actionId"))
    write_csv_rows(report_dir / "point_by_next_strikeNumber.csv", label_by_next_strike(train_groups, "pointId"))

    train_matches = {row["match"] for row in train}
    test_matches = {row["match"] for row in test}
    train_players = {row["gamePlayerId"] for row in train} | {row["gamePlayerOtherId"] for row in train}
    test_players = {row["gamePlayerId"] for row in test} | {row["gamePlayerOtherId"] for row in test}

    match_rallies = Counter(rows[0]["match"] for rows in train_groups.values())
    write_csv_rows(report_dir / "train_match_rally_counts.csv", counter_rows(match_rallies, "match", "rallies"))

    summary_lines = []
    summary_lines.append("Dataset EDA summary")
    summary_lines.append("===================")
    summary_lines.append("")
    summary_lines.append(f"train rows: {len(train):,}")
    summary_lines.append(f"test rows: {len(test):,}")
    summary_lines.append(f"train rally length: {rally_length_summary(train_groups)}")
    summary_lines.append(f"test prefix length: {rally_length_summary(test_groups)}")
    summary_lines.append(f"train/test match overlap count: {len(train_matches & test_matches)}")
    summary_lines.append(f"train/test player overlap count: {len(train_players & test_players)}")
    summary_lines.append("")
    summary_lines.append("Notes:")
    summary_lines.append("- Use rally or match grouped validation; never random row split.")
    summary_lines.append("- Build train samples as prefix 1..k -> target stroke k+1 to match test_new.csv.")
    summary_lines.append("- test_new prefixes are short, so short-prefix sampling/weights matter.")

    (report_dir / "eda_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()

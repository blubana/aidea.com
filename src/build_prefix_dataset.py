"""Build train/test prefix feature datasets.

Train construction:
  observed strokes 1..k -> labels from stroke k+1 plus rally outcome.

Test construction:
  all rows for each rally_uid are the observed prefix -> one feature row.

Outputs CSV files under data/processed/ by default.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from features import build_prefix_features, feature_fieldnames, group_by_rally, read_csv_rows, to_int, write_csv_rows


def build_test_length_weights(test_groups):
    counts = Counter(len(rows) for rows in test_groups.values())
    total = sum(counts.values()) or 1
    max_ratio = max((count / total for count in counts.values()), default=1.0)
    # Normalize so the most common prefix length gets weight 1.0; unseen lengths
    # still get a small positive weight.
    return {length: round((count / total) / max_ratio, 6) for length, count in counts.items()}


def build_length_distribution(groups, max_prefix_len: int, is_train: bool) -> dict[int, float]:
    counts = Counter()
    if is_train:
        for rows in groups.values():
            rally_len = len(rows)
            if rally_len < 2:
                continue
            for k in range(1, min(rally_len - 1, max_prefix_len) + 1):
                counts[k] += 1
    else:
        for rows in groups.values():
            counts[min(len(rows), max_prefix_len)] += 1
    total = sum(counts.values()) or 1
    return {length: count / total for length, count in counts.items()}


def build_sample_weight_lookup(train_groups, test_groups, max_prefix_len: int, weight_mode: str, max_sample_weight: float) -> dict[int, float]:
    if weight_mode == "uniform":
        return {}
    if weight_mode == "test_freq_norm":
        return build_test_length_weights(test_groups)

    train_dist = build_length_distribution(train_groups, max_prefix_len, is_train=True)
    test_dist = build_length_distribution(test_groups, max_prefix_len, is_train=False)
    weights = {}
    all_lengths = sorted(set(train_dist) | set(test_dist))
    for length in all_lengths:
        p_train = train_dist.get(length, 0.0)
        p_test = test_dist.get(length, 0.0)
        if weight_mode == "test_over_train":
            raw = p_test / max(p_train, 1e-9)
        elif weight_mode == "rally_balanced_server":
            raw = 1.0 / max(p_train, 1e-9)
        else:
            raise ValueError(f"Unknown weight mode: {weight_mode}")
        weights[length] = min(raw, max_sample_weight)
    mean_weight = sum(weights.values()) / max(len(weights), 1)
    if mean_weight > 0:
        weights = {k: round(v / mean_weight, 6) for k, v in weights.items()}
    return weights


def build_train_rows(train_groups, sample_weight_lookup, max_prefix_len, weight_mode):
    out = []
    for rally_uid, rows in sorted(train_groups.items(), key=lambda kv: to_int(kv[0], 0)):
        rally_len = len(rows)
        if rally_len < 2:
            continue
        max_k = min(rally_len - 1, max_prefix_len)
        server_label = rows[-1].get("serverGetPoint", rows[0].get("serverGetPoint", ""))
        for k in range(1, max_k + 1):
            target = rows[k]
            features = build_prefix_features(rows[:k])
            features.update(
                {
                    "sample_id": f"{rally_uid}_{k}",
                    "source_rally_len": rally_len,
                    "target_strikeNumber": to_int(target.get("strikeNumber"), k + 1),
                    "sample_weight": 1.0 if weight_mode == "uniform" else sample_weight_lookup.get(k, 0.1),
                    "label_actionId": to_int(target.get("actionId")),
                    "label_pointId": to_int(target.get("pointId")),
                    "label_serverGetPoint": to_int(server_label),
                }
            )
            out.append(features)
    return out


def build_test_rows(test_groups):
    out = []
    for rally_uid, rows in sorted(test_groups.items(), key=lambda kv: to_int(kv[0], 0)):
        features = build_prefix_features(rows)
        features.update(
            {
                "sample_id": str(rally_uid),
                "source_rally_len": len(rows),
                "target_strikeNumber": len(rows) + 1,
                "sample_weight": 1.0,
            }
        )
        out.append(features)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--max-prefix-len", type=int, default=12)
    parser.add_argument("--weight-mode", default="test_freq_norm", choices=["test_freq_norm", "test_over_train", "uniform", "rally_balanced_server"])
    parser.add_argument("--max-sample-weight", type=float, default=5.0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    train = read_csv_rows(dataset_dir / "train.csv")
    test = read_csv_rows(dataset_dir / "test_new.csv")
    train_groups = group_by_rally(train)
    test_groups = group_by_rally(test)

    sample_weight_lookup = build_sample_weight_lookup(train_groups, test_groups, args.max_prefix_len, args.weight_mode, args.max_sample_weight)
    train_rows = build_train_rows(train_groups, sample_weight_lookup, args.max_prefix_len, args.weight_mode)
    test_rows = build_test_rows(test_groups)

    fieldnames = feature_fieldnames(train_rows[0] if train_rows else test_rows[0])
    test_fieldnames = [name for name in fieldnames if not name.startswith("label_")]
    write_csv_rows(output_dir / "prefix_train_features.csv", train_rows, fieldnames)
    write_csv_rows(output_dir / "prefix_test_features.csv", test_rows, test_fieldnames)

    print(f"Wrote {len(train_rows):,} train prefix samples to {output_dir / 'prefix_train_features.csv'}")
    print(f"Wrote {len(test_rows):,} test prefix samples to {output_dir / 'prefix_test_features.csv'}")
    metadata = {
        "weight_mode": args.weight_mode,
        "max_sample_weight": args.max_sample_weight,
        "sample_weight_lookup": dict(sorted(sample_weight_lookup.items())),
    }
    (output_dir / "prefix_dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Sample weights: {dict(sorted(sample_weight_lookup.items()))}")


if __name__ == "__main__":
    main()

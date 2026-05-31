"""Feature generation utilities for table-tennis rally prefix prediction.

The project environment intentionally avoids third-party dependencies here.
All functions use only Python's standard library so the same pipeline can be
used before installing modeling packages.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ID_COLUMNS = {
    "rally_uid",
    "sex",
    "match",
    "numberGame",
    "rally_id",
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "serverGetPoint",
    "gamePlayerId",
    "gamePlayerOtherId",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
}

STROKE_FEATURE_COLUMNS = [
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
    "gamePlayerId",
    "gamePlayerOtherId",
]

COUNT_SPECS = {
    "actionId": range(0, 19),
    "pointId": range(0, 10),
    "strikeId": [1, 2, 4, 8, 16],
    "handId": range(0, 3),
    "strengthId": range(0, 4),
    "spinId": range(0, 6),
    "positionId": range(0, 4),
}

ACTION_GROUPS = {
    "zero": {0},
    "attack": {1, 2, 3, 4, 5, 6, 7},
    "control": {8, 9, 10, 11},
    "defensive": {12, 13, 14},
    "serve": {15, 16, 17, 18},
}


def to_int(value: object, default: int = -1) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def group_by_rally(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["rally_uid"]].append(row)
    for rally_rows in groups.values():
        rally_rows.sort(key=lambda r: to_int(r.get("strikeNumber"), 0))
    return dict(groups)


def action_group(action_id: int) -> str:
    for name, ids in ACTION_GROUPS.items():
        if action_id in ids:
            return name
    return "unknown"


def point_depth(point_id: int) -> int:
    if point_id in (1, 2, 3):
        return 1  # short
    if point_id in (4, 5, 6):
        return 2  # half-long
    if point_id in (7, 8, 9):
        return 3  # long
    return 0


def point_side(point_id: int) -> int:
    if point_id in (1, 4, 7):
        return 1  # forehand
    if point_id in (2, 5, 8):
        return 2  # middle
    if point_id in (3, 6, 9):
        return 3  # backhand
    return 0


def _safe_ratio(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def _transition_token(left: object, right: object) -> str:
    return f"{left}->{right}"


def build_prefix_features(prefix_rows: Sequence[Dict[str, str]], last_n: int = 5) -> Dict[str, object]:
    """Build one tabular feature row from a known rally prefix.

    prefix_rows must contain only observed strokes. The target stroke must not be
    included; this prevents future leakage for next action/point prediction.
    """

    if not prefix_rows:
        raise ValueError("prefix_rows must contain at least one stroke")

    rows = sorted(prefix_rows, key=lambda r: to_int(r.get("strikeNumber"), 0))
    first = rows[0]
    last = rows[-1]
    prefix_len = len(rows)
    next_strike = prefix_len + 1
    score_self = to_int(last.get("scoreSelf"), 0)
    score_other = to_int(last.get("scoreOther"), 0)

    features: Dict[str, object] = {
        "rally_uid": first.get("rally_uid", ""),
        "sex": to_int(first.get("sex")),
        "match": to_int(first.get("match")),
        "numberGame": to_int(first.get("numberGame")),
        "rally_id": to_int(first.get("rally_id")),
        "prefix_len": prefix_len,
        "log_prefix_len": round(math.log1p(prefix_len), 6),
        "next_strikeNumber": next_strike,
        "scoreSelf": score_self,
        "scoreOther": score_other,
        "score_diff": score_self - score_other,
        "score_sum": score_self + score_other,
        "abs_score_diff": abs(score_self - score_other),
        "is_deuce": int(score_self >= 10 and score_other >= 10),
        "is_close_score": int(abs(score_self - score_other) <= 1),
        "is_self_leading": int(score_self > score_other),
        "is_other_leading": int(score_other > score_self),
        "is_short_prefix": int(prefix_len <= 2),
        "is_medium_prefix": int(3 <= prefix_len <= 5),
        "is_long_prefix": int(prefix_len >= 6),
        "next_is_receive": int(next_strike == 2),
        "next_is_third_ball": int(next_strike == 3),
        "next_is_rally_phase": int(next_strike >= 4),
        "next_is_odd": next_strike % 2,
        "last_hitter_id": to_int(last.get("gamePlayerId")),
        "last_opponent_id": to_int(last.get("gamePlayerOtherId")),
        "next_hitter_is_initial_server_side": int(next_strike % 2 == 1),
    }

    # Last-N explicit features. Missing history is encoded as -1.
    for lag in range(1, last_n + 1):
        source = rows[-lag] if len(rows) >= lag else None
        for col in STROKE_FEATURE_COLUMNS:
            features[f"last{lag}_{col}"] = to_int(source.get(col)) if source else -1
        if source:
            aid = to_int(source.get("actionId"), 0)
            pid = to_int(source.get("pointId"), 0)
            features[f"last{lag}_action_group"] = action_group(aid)
            features[f"last{lag}_point_depth"] = point_depth(pid)
            features[f"last{lag}_point_side"] = point_side(pid)
        else:
            features[f"last{lag}_action_group"] = "missing"
            features[f"last{lag}_point_depth"] = -1
            features[f"last{lag}_point_side"] = -1

    # Counts and ratios over the observed prefix.
    for col, values in COUNT_SPECS.items():
        counter = Counter(to_int(row.get(col), -999) for row in rows)
        for value in values:
            count = counter[value]
            features[f"cnt_{col}_{value}"] = count
            features[f"ratio_{col}_{value}"] = round(_safe_ratio(count, prefix_len), 6)

    group_counter = Counter(action_group(to_int(row.get("actionId"), 0)) for row in rows)
    for name in ["zero", "attack", "control", "defensive", "serve", "unknown"]:
        count = group_counter[name]
        features[f"cnt_action_group_{name}"] = count
        features[f"ratio_action_group_{name}"] = round(_safe_ratio(count, prefix_len), 6)

    depth_counter = Counter(point_depth(to_int(row.get("pointId"), 0)) for row in rows)
    side_counter = Counter(point_side(to_int(row.get("pointId"), 0)) for row in rows)
    for value in range(0, 4):
        features[f"cnt_point_depth_{value}"] = depth_counter[value]
        features[f"ratio_point_depth_{value}"] = round(_safe_ratio(depth_counter[value], prefix_len), 6)
        features[f"cnt_point_side_{value}"] = side_counter[value]
        features[f"ratio_point_side_{value}"] = round(_safe_ratio(side_counter[value], prefix_len), 6)

    features["last_action_is_zero"] = int(to_int(last.get("actionId"), 0) == 0)
    features["last_point_is_zero"] = int(to_int(last.get("pointId"), 0) == 0)
    features["has_zero_action"] = int(any(to_int(row.get("actionId"), 0) == 0 for row in rows))
    features["has_zero_point"] = int(any(to_int(row.get("pointId"), 0) == 0 for row in rows))

    prev = rows[-2] if len(rows) >= 2 else None
    if prev is not None:
        prev_action = to_int(prev.get("actionId"), 0)
        last_action = to_int(last.get("actionId"), 0)
        prev_point = to_int(prev.get("pointId"), 0)
        last_point = to_int(last.get("pointId"), 0)
        prev_depth = point_depth(prev_point)
        last_depth = point_depth(last_point)
        prev_side = point_side(prev_point)
        last_side = point_side(last_point)
        prev_group = action_group(prev_action)
        last_group = action_group(last_action)
        features["last2_action_transition"] = _transition_token(prev_action, last_action)
        features["last2_point_transition"] = _transition_token(prev_point, last_point)
        features["last2_depth_transition"] = _transition_token(prev_depth, last_depth)
        features["last2_side_transition"] = _transition_token(prev_side, last_side)
        features["last2_action_group_transition"] = _transition_token(prev_group, last_group)
        same_hitter = to_int(prev.get("gamePlayerId"), -1) == to_int(last.get("gamePlayerId"), -1)
        features["last2_same_hitter"] = int(same_hitter)
    else:
        features["last2_action_transition"] = "missing"
        features["last2_point_transition"] = "missing"
        features["last2_depth_transition"] = "missing"
        features["last2_side_transition"] = "missing"
        features["last2_action_group_transition"] = "missing"
        features["last2_same_hitter"] = -1

    last_hitter = to_int(last.get("gamePlayerId"), -1)
    last_action = to_int(last.get("actionId"), -1)
    recent_window = rows[-3:]
    features["recent_same_hitter_count_3"] = sum(to_int(r.get("gamePlayerId"), -999) == last_hitter for r in recent_window)
    features["recent_same_action_count_3"] = sum(to_int(r.get("actionId"), -999) == last_action for r in recent_window)
    features["recent_attack_group_count_3"] = sum(action_group(to_int(r.get("actionId"), 0)) == "attack" for r in recent_window)

    return features


def feature_fieldnames(example_row: Dict[str, object]) -> List[str]:
    labels = ["label_actionId", "label_pointId", "label_serverGetPoint"]
    meta = ["sample_id", "source_rally_len", "target_strikeNumber", "sample_weight"]
    ordered = []
    for key in ["sample_id", "rally_uid", "source_rally_len", "target_strikeNumber", "sample_weight"]:
        if key in example_row:
            ordered.append(key)
    ordered.extend(k for k in example_row if k not in set(ordered + labels + meta))
    ordered.extend(k for k in labels if k in example_row)
    return ordered

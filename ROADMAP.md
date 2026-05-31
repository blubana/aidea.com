# Project roadmap

Goal: predict one row per `rally_uid` for three targets:

- `actionId`: maximize Macro F1
- `pointId`: maximize Macro F1
- `serverGetPoint`: maximize ROC AUC

## Simplified route

```text
EDA
→ EDA visualization
→ prefix builder
→ feature generator
→ feature visualization / sanity checks
→ tabular baseline + validation
→ first submission
→ LSTM dataset
→ independent LSTM models
→ compare with baseline
→ simple ensemble validation
→ Step A diagnostics
→ Step B point-phase optimization
→ Step C server stacking optimization
→ Step D CatBoost OOF baselines and blending
→ stacking / auxiliary tasks / tuning
```

## Current modeling decision

Start with one independent model per task:

```text
model_actionId
model_pointId
model_serverGetPoint
```

Reason:

- The three targets have different behavior.
- `actionId` and `pointId` are next-stroke predictions.
- `serverGetPoint` is rally outcome prediction.
- Independent models are easier to validate, debug, tune, and ensemble.

Later stages may add stacking, where action/point out-of-fold probabilities are
used as additional features for the `serverGetPoint` model.

Current Step C command placeholder:

```powershell
uv run python src\train_server_stacking.py --model extratrees
```

## Status

- [x] EDA
- [x] EDA visualization
- [x] prefix builder
- [x] feature generator
- [ ] feature visualization / sanity checks
- [x] tabular baseline + validation
- [x] first tabular submission script
- [x] LSTM dataset
- [x] independent LSTM models smoke test
- [x] independent LSTM models full validation
- [x] compare with baseline
- [x] ensemble validation
- [x] Step A diagnostics review
- [x] Step B point-phase optimization review
- [x] Step C server stacking optimization
- [x] Step D CatBoost OOF baselines and blending
- [x] final blended submission
- [ ] stacking / auxiliary tasks / tuning

Current final blend command:

```powershell
uv run python src\predict_final_blend.py
```

Current final blend weights:

- `actionId`: `0.85 * action_base + 0.15 * CatBoost`, where `action_base = 0.70 tabular + 0.30 LSTM`, then serve-class mask.
- `pointId`: `0.70 * point_without_catboost + 0.30 * CatBoost`, where `point_without_catboost = 0.60 * point_base + 0.40 * point_phase`, and `point_base = 0.35 tabular + 0.65 LSTM`.
- `serverGetPoint`: `0.45 * server_without_catboost + 0.55 * CatBoost`, where `server_without_catboost = 0.35 * server_base + 0.65 * server_stacking`, and `server_base = 0.80 tabular + 0.20 LSTM`.

Current Step D command placeholders:

```powershell
uv run python src\train_catboost_baseline.py --targets actionId pointId serverGetPoint
uv run python src\blend_catboost.py
```

Final blended submission generated and sanity-checked:

```text
submissions/submission_final_blend.csv
rows: 1,845
columns: rally_uid, actionId, pointId, serverGetPoint
no NaNs: true
unique rally_uid: 1,845
valid class ranges: true
```

## Current best validation direction

Primary validation remains `GroupKFold` by `match` because train/test match
overlap is zero. Current metric-based OOF ensemble directions after Step D:

| Target | Objective | Key extra blend | Metric |
|---|---|---:|---:|
| `actionId` | Macro F1 | CatBoost 0.15 | 0.34459 |
| `pointId` | Macro F1 | CatBoost 0.25 | 0.21245 |
| `serverGetPoint` | ROC AUC | CatBoost 0.60 | 0.60946 |

The tabular-first submission applies an action serve-class mask for test targets
with `target_strikeNumber >= 2`.

## Step A diagnostics summary

Step A checks whether current validation is inflated by test-prefix mismatch or
leakage.

Key diagnostic results:

| Target | Source | Main Metric | All OOF | Test-weighted | Prefix <= 3 | Prefix <= 4 |
|---|---|---:|---:|---:|---:|---:|
| `pointId` | ensemble | Macro F1 | 0.20690 | 0.19177 | 0.18452 | 0.18829 |
| `serverGetPoint` | ensemble | ROC AUC | 0.60167 | 0.58407 | 0.57532 | 0.58045 |

Leakage checks:

- `source_rally_len` parity is an oracle for `serverGetPoint` in train
  (`AUC=0.99846`), so it must never be used as a feature.
- Saved tabular model features pass forbidden-feature checks.
- LSTM manual features pass forbidden-feature checks.
- Prefix length alone is not a useful server proxy (`prefix_len` raw AUC=0.48743).
- Prefix-length ablation showed minimal effect:
  - `pointId` Macro F1: 0.19461 with prefix features vs. 0.19409 without.
  - `serverGetPoint` AUC: 0.59821 with prefix features vs. 0.59819 without.

## Step B point-phase optimization summary

Step B trains pointId-only phase-specific ExtraTrees models by
`next_strikeNumber` bucket:

- `receive`: `next_strikeNumber == 2`
- `third_ball`: `next_strikeNumber == 3`
- `early_rally`: `next_strikeNumber in {4, 5}`
- `rally`: `next_strikeNumber >= 6`

Results:

| Point model | Macro F1 | Test-weighted Macro F1 | Prefix <= 3 | Prefix <= 4 |
|---|---:|---:|---:|---:|
| previous point ensemble | 0.20690 | 0.19177 | 0.18452 | 0.18829 |
| point phase only | 0.19764 | 0.18609 | 0.18467 | 0.18942 |
| phase blend, best all Macro F1 | 0.20932 | 0.19438 | 0.19144 | 0.19816 |
| phase blend, best weighted Macro F1 | 0.20909 | 0.19462 | 0.19189 | 0.19836 |

Interpretation: phase-specific point models are not strong enough alone, but they
add useful diversity when blended with the previous point ensemble. The best
test-like setting currently uses `weight_point_phase=0.40`.

## Step C server stacking optimization summary

Step C stacks safe base features with action/point OOF probability features to
improve `serverGetPoint` ROC AUC. It uses no `source_rally_len`, labels,
`target_strikeNumber`, IDs, or `sample_weight` as model features.

Results:

| Server model | ROC AUC | Test-weighted ROC AUC | Prefix <= 3 AUC | Prefix <= 4 AUC |
|---|---:|---:|---:|---:|
| previous server ensemble | 0.60167 | 0.58407 | 0.58247 | 0.59198 |
| server stacking only | 0.60238 | 0.58640 | 0.58592 | 0.59411 |
| stack blend, best all AUC | 0.60473 | 0.58762 | 0.58653 | 0.59535 |
| stack blend, best weighted AUC | 0.60465 | 0.58776 | 0.58680 | 0.59547 |

Interpretation: server stacking gives a small but consistent honest improvement.
The best test-like setting currently uses `weight_server_stacking=0.65`.

## Step D CatBoost summary

Step D trains CatBoost OOF baselines for all three tasks and blends their OOF
probabilities with the previous best validation blends. CatBoost is trained with
the same safe tabular feature policy: no `source_rally_len`, labels, IDs,
`target_strikeNumber`, or `sample_weight` as features.

CatBoost-only results:

| Target | Main metric | Test-weighted metric | Prefix <= 3 | Prefix <= 4 |
|---|---:|---:|---:|---:|
| `actionId` Macro F1 | 0.27815 | 0.26101 | 0.26959 | 0.27345 |
| `pointId` Macro F1 | 0.17603 | 0.16893 | 0.16714 | 0.16993 |
| `serverGetPoint` ROC AUC | 0.60593 | 0.58566 | 0.58403 | 0.59329 |

Best CatBoost blend-search results:

| Target | CatBoost Weight | Main metric | Test-weighted metric |
|---|---:|---:|---:|
| `actionId` Macro F1 | 0.15 | 0.34459 | — |
| `pointId` weighted Macro F1 | 0.30 | 0.21196 | 0.20162 |
| `serverGetPoint` weighted ROC AUC | 0.55 | 0.60941 | 0.59080 |

Interpretation: CatBoost is mainly useful as ensemble diversity. The updated
final submission generator now includes CatBoost test probabilities using the
test-like weighted blend choices for point and server.

## Visualization support

Use `matplotlib` and `seaborn` for:

- train/test rally length distribution
- test prefix length distribution
- label distributions
- action/point distribution by next stroke number
- validation fold metric plots
- feature importance plots for tabular models

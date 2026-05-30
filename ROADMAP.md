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
- [ ] final blended submission
- [ ] stacking / auxiliary tasks / tuning

## Current best validation direction

Primary validation remains `GroupKFold` by `match` because train/test match
overlap is zero. Current metric-based OOF ensemble directions:

| Target | Objective | Best LSTM Weight | Metric |
|---|---|---:|---:|
| `actionId` | Macro F1 | 0.30 | 0.33891 |
| `pointId` | Macro F1 | 0.65 | 0.20690 |
| `serverGetPoint` | ROC AUC | 0.20 | 0.60167 |

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

## Visualization support

Use `matplotlib` and `seaborn` for:

- train/test rally length distribution
- test prefix length distribution
- label distributions
- action/point distribution by next stroke number
- validation fold metric plots
- feature importance plots for tabular models

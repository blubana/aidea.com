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

## Visualization support

Use `matplotlib` and `seaborn` for:

- train/test rally length distribution
- test prefix length distribution
- label distributions
- action/point distribution by next stroke number
- validation fold metric plots
- feature importance plots for tabular models

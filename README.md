# AIDEA Table-Tennis Rally Prediction

This repository builds a reproducible machine-learning pipeline for predicting
three targets per `rally_uid`:

- `actionId`: next-stroke action class, 19 classes `0..18`; objective Macro F1
- `pointId`: next-stroke landing-point class, 10 classes `0..9`; objective Macro F1
- `serverGetPoint`: rally outcome from the server perspective, binary `0/1`; objective ROC AUC

The current strategy is to train one independent model per target first. This is
intentional because `actionId` and `pointId` are next-stroke prediction tasks,
while `serverGetPoint` is a rally-outcome task. Independent models are easier to
debug, validate, tune, and later ensemble.

## Current roadmap

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

## Environment

The recommended environment manager is `uv`.

```powershell
uv python install 3.12
uv venv --python 3.12
uv sync
```

The project is configured to install PyTorch CUDA wheels from the `cu130` index.
The local machine has an NVIDIA RTX 4070 Laptop GPU, and CUDA is available with:

```text
torch 2.12.0+cu130
cuda available: True
```

## Reproduction

### 1. Generate EDA reports

```powershell
uv run python src\eda_dataset.py
```

Outputs:

```text
reports/eda_summary.txt
reports/train_rally_length_distribution.csv
reports/test_prefix_length_distribution.csv
reports/train_actionId_distribution.csv
reports/train_pointId_distribution.csv
reports/train_serverGetPoint_distribution.csv
reports/action_by_next_strikeNumber.csv
reports/point_by_next_strikeNumber.csv
reports/train_match_rally_counts.csv
```

### 2. Generate EDA visualizations

```powershell
uv run python src\visualize_eda.py
```

Outputs figures under:

```text
reports/figures/
```

### 3. Build prefix feature datasets

```powershell
uv run python src\build_prefix_dataset.py --max-prefix-len 12
```

Outputs:

```text
data/processed/prefix_train_features.csv
data/processed/prefix_test_features.csv
```

Current generated dataset sizes:

```text
prefix_train_features.csv: 65,790 rows, 228 columns
prefix_test_features.csv: 1,845 rows, 225 columns
```

The train file has three additional label columns:

```text
label_actionId
label_pointId
label_serverGetPoint
```

### 4. Train tabular baselines with grouped validation

```powershell
uv run python src\train_tabular_baseline.py --folds 5 --n-estimators 300 --max-depth 18 --min-samples-leaf 2 --group-col match
```

Outputs:

```text
models/tabular_baseline/actionId_extratrees.joblib
models/tabular_baseline/pointId_extratrees.joblib
models/tabular_baseline/serverGetPoint_extratrees.joblib

reports/tabular_baseline/summary.json
reports/tabular_baseline/*_fold_metrics.csv
reports/tabular_baseline/*_oof_predictions.csv
reports/tabular_baseline/*_oof_proba.npy
```

### 5. Build the LSTM dataset

```powershell
uv run python src\lstm_dataset.py --max-len 12
```

Output:

```text
data/processed/lstm_dataset.npz
```

The NPZ contains:

```text
X_cat_seq:      (65,790, 12, 10), int64
X_num_seq:      (65,790, 12, 10), float32
X_manual:       (65,790, 214), float32
lengths:        (65,790,), int64

test_X_cat_seq: (1,845, 12, 10), int64
test_X_num_seq: (1,845, 12, 10), float32
test_X_manual:  (1,845, 214), float32
test_lengths:   (1,845,), int64
```

Sequence construction details:

- Each train sample uses `prefix strokes 1..k` from its `sample_id = rally_uid_k`.
- Each test sample uses all known rows for that `rally_uid`.
- If a prefix is longer than `max_len`, the most recent `max_len` strokes are kept.
- If a prefix is longer than `max_len`, the most recent `max_len` strokes are kept.
- Valid timesteps are placed first and right-padded with zeros, so PyTorch
  `pack_padded_sequence` reads real strokes before padding.

Step-level categorical features:

```text
sex, gamePlayerId, gamePlayerOtherId, strikeId, handId, strengthId,
spinId, pointId, actionId, positionId
```

Step-level numeric features:

```text
strikeNumber, scoreSelf, scoreOther, score_diff, score_sum,
is_odd_stroke, is_serve_stroke, is_receive_stroke,
is_third_ball, is_rally_phase
```

### 6. Train independent LSTM models

Smoke-test commands:

```powershell
uv run python src\train_lstm.py --target actionId --epochs 1 --sample 1024 --max-folds 1 --batch-size 128 --hidden-size 32
uv run python src\train_lstm.py --target pointId --epochs 1 --sample 1024 --max-folds 1 --batch-size 128 --hidden-size 32
uv run python src\train_lstm.py --target serverGetPoint --epochs 1 --sample 1024 --max-folds 1 --batch-size 128 --hidden-size 32
```

Objective-aligned baseline commands:

```powershell
uv run python src\train_lstm.py --target actionId --preset action
uv run python src\train_lstm.py --target pointId --preset point
uv run python src\train_lstm.py --target serverGetPoint --preset server --no-use-sample-weight
```

Outputs:

```text
models/lstm/<target>_fold*.pt
reports/lstm/<target>_metrics.json
reports/lstm/<target>_oof_proba.npy
reports/lstm/<target>_oof_predictions.csv
```

The current LSTM trainer uses:

- CUDA if available
- `GroupKFold` by match
- `AdamW`
- gradient clipping
- objective-aligned early stopping by default:
  - `actionId`: Macro F1
  - `pointId`: Macro F1
  - `serverGetPoint`: ROC AUC
- class-weighted cross entropy for `actionId` and `pointId`
- binary cross entropy for `serverGetPoint`
- optional per-sample weights from test prefix-length distribution
- train-fold standardization for `X_manual` and valid timesteps in `X_num_seq`

Preset examples:

```powershell
uv run python src\train_lstm.py --target actionId --preset action
uv run python src\train_lstm.py --target pointId --preset point
uv run python src\train_lstm.py --target serverGetPoint --preset server --no-use-sample-weight
```

### 7. Train Step C serverGetPoint stacking

```powershell
uv run python src\train_server_stacking.py --model extratrees
```

Outputs:

```text
models/server_stacking/serverGetPoint_<model>.joblib
reports/server_stacking/serverGetPoint_oof_proba.npy
reports/server_stacking/serverGetPoint_oof_predictions.csv
reports/server_stacking/blend_summary.csv
reports/server_stacking/summary.json
```

### 8. Create final blended submission

```powershell
uv run python src\predict_final_blend.py
uv run python src\predict_final_blend.py --server-output bool
```

Current final blend weights include CatBoost probabilities from `reports/catboost/*_test_proba.npy`:

- `actionId`: `0.85 * (0.70 * tabular + 0.30 * LSTM) + 0.15 * CatBoost`, then serve-class mask for `target_strikeNumber >= 2`
- `pointId`: `0.70 * (0.60 * (0.35 * tabular + 0.65 * LSTM) + 0.40 * point_phase) + 0.30 * CatBoost`
- `serverGetPoint`: `0.45 * (0.35 * (0.80 * tabular + 0.20 * LSTM) + 0.65 * server_stacking) + 0.55 * CatBoost`

Notes:

- `src/predict_final_blend.py` now supports `--server-output {float,bool}`. `float` writes the positive-class probability, while `bool` writes the argmax class.
- `--use-cross-target-stacking/--no-use-cross-target-stacking` is reserved for future bounded experiments and is currently stored in the run summary only.
- `src/train_tabular_baseline.py` now explicitly drops `sample_weight` from model features while still using it as training weight.

Outputs:

```text
submissions/submission_final_blend.csv
reports/final_blend/*.npy
reports/final_blend/summary.json
```

### Step D. Train CatBoost baselines and OOF blends

```powershell
uv run python src\train_catboost_baseline.py --targets actionId pointId serverGetPoint --task-type GPU
uv run python src\blend_catboost.py
```

Outputs:

```text
models/catboost/<target>_fold*.cbm
models/catboost/<target>_final.cbm
reports/catboost/<target>_oof_proba.npy
reports/catboost/<target>_test_proba.npy
reports/catboost/<target>_oof_predictions.csv
reports/catboost/summary.json
reports/catboost_blend/summary.csv
reports/catboost_blend/summary.json
```

Current CatBoost-only OOF results:

| Target | Main metric | Test-weighted metric | Prefix <= 3 | Prefix <= 4 |
|---|---:|---:|---:|---:|
| `actionId` Macro F1 | 0.27815 | 0.26101 | 0.26959 | 0.27345 |
| `pointId` Macro F1 | 0.17603 | 0.16893 | 0.16714 | 0.16993 |
| `serverGetPoint` ROC AUC | 0.60593 | 0.58566 | 0.58403 | 0.59329 |

CatBoost adds useful ensemble diversity even though standalone multiclass F1 is
weaker than the current blends. Best OOF blend-search results:

| Target | CatBoost Weight | Main metric | Test-weighted metric |
|---|---:|---:|---:|
| `actionId` Macro F1 | 0.15 | 0.34459 | — |
| `pointId` Macro F1 | 0.25 | 0.21245 | 0.20149 |
| `pointId` weighted Macro F1 | 0.30 | 0.21196 | 0.20162 |
| `serverGetPoint` ROC AUC | 0.60 | 0.60946 | 0.59072 |
| `serverGetPoint` weighted ROC AUC | 0.55 | 0.60941 | 0.59080 |

The final submission generator currently uses the test-like weighted choices for
`pointId` (`weight_catboost=0.30`) and `serverGetPoint` (`weight_catboost=0.55`).

### 9. Create tabular-baseline submission

```powershell
uv run python src\predict_tabular_submission.py
```

Output:

```text
submissions/submission_tabular_baseline.csv
```

### 10. Validate a simple tabular + LSTM ensemble

```powershell
uv run python src\ensemble_validation.py
```

Outputs:

```text
reports/ensemble/summary.csv
reports/ensemble/summary.json
```

### 11. Run Step A validation diagnostics

```powershell
uv run python src\validation_diagnostics.py
uv run python src\validation_diagnostics.py --run-prefix-ablation
```

Outputs:

```text
reports/diagnostics/oof_slices.csv
reports/diagnostics/summary.json
reports/diagnostics/leakage_checks.json
reports/diagnostics/prefix_ablation.csv  # optional
```

### 10. Run Step B pointId phase-specific optimization

```powershell
uv run python src\train_point_phase_models.py
```

Outputs:

```text
models/point_phase/global.joblib
models/point_phase/<bucket>.joblib
reports/point_phase/summary.json
reports/point_phase/per_bucket_metrics.csv
reports/point_phase/pointId_oof_proba.npy
reports/point_phase/pointId_oof_predictions.csv
reports/point_phase/pointId_blend_summary.csv  # optional
```

## Random seed and reproducibility

The tabular and LSTM scripts use a default seed:

```text
--random-state 42
```

This seed is applied to:

- global NumPy random state
- optional debug row sampling
- `ExtraTreesClassifier(random_state=42)`
- PyTorch CPU seed
- PyTorch CUDA seed when CUDA is available

`GroupKFold` itself is deterministic and does not shuffle groups.

For an exact rerun, use the same command-line arguments, the same generated
prefix feature files, and the same dependency versions from `uv.lock`.

## Current baseline results

Validation split:

```text
GroupKFold by match, 5 folds
```

Model:

```text
ExtraTreesClassifier
n_estimators = 300
max_depth = 18
min_samples_leaf = 2
class_weight = balanced
random_state = 42
```

Out-of-fold results:

| Target | Accuracy | Macro F1 | Log Loss |
|---|---:|---:|---:|
| `actionId` | 0.46349 | 0.31743 | 1.59867 |
| `pointId` | 0.27066 | 0.19617 | 1.88638 |
| `serverGetPoint` | 0.56825 | 0.56736 | 0.67629 |

## Current LSTM status

Implemented files:

```text
src/lstm_dataset.py
src/models_lstm.py
src/train_lstm.py
```

Smoke tests have passed on CUDA for all three targets with:

```text
--epochs 1 --sample 1024 --max-folds 1 --batch-size 128 --hidden-size 32
```

Full validation has also been run with:

```powershell
uv run python src\train_lstm.py --target actionId --epochs 30 --batch-size 256 --hidden-size 128 --num-layers 1 --dropout 0.2
uv run python src\train_lstm.py --target pointId --epochs 30 --batch-size 256 --hidden-size 128 --num-layers 1 --dropout 0.2
uv run python src\train_lstm.py --target serverGetPoint --epochs 30 --batch-size 256 --hidden-size 128 --num-layers 1 --dropout 0.2
```

Validation split:

```text
GroupKFold by match, 5 folds
```

Full LSTM out-of-fold fold-average results:

| Target | Accuracy | Macro F1 | Log Loss | ROC AUC |
|---|---:|---:|---:|---:|
| `actionId` | 0.39398 | 0.29964 | 2.30651 | — |
| `pointId` | 0.23118 | 0.17471 | 2.40484 | — |
| `serverGetPoint` | 0.52576 | 0.52338 | 4.61164 | 0.52903 |

Current conclusion:

- The first LSTM baseline is functional but weaker than the tabular baseline.
- Do not use this raw LSTM as the primary model yet.
- The next LSTM work should focus on calibration, early stopping, better manual
  feature normalization, target-specific losses, and ensemble testing.

## Generated feature groups

The feature generator currently creates the following groups.

### Rally prefix length features

- `prefix_len`
- `log_prefix_len`
- `is_short_prefix`
- `is_medium_prefix`
- `is_long_prefix`

### Next-stroke position features

- `next_strikeNumber`
- `next_is_receive`
- `next_is_third_ball`
- `next_is_rally_phase`
- `next_is_odd`

### Last-N stroke features

For `last1` through `last5`:

- `strikeId`
- `handId`
- `strengthId`
- `spinId`
- `pointId`
- `actionId`
- `positionId`
- `gamePlayerId`
- `gamePlayerOtherId`
- derived `action_group`
- derived `point_depth`
- derived `point_side`

### Action group features

`actionId` is mapped into:

- `zero`
- `attack`
- `control`
- `defensive`
- `serve`
- `unknown`

The pipeline creates count and ratio features for each group.

### Point decomposition features

`pointId` is decomposed into:

- depth: none / short / half-long / long
- side: none / forehand / middle / backhand

The pipeline creates count and ratio features for depth and side.

### Score state features

- `scoreSelf`
- `scoreOther`
- `score_diff`
- `score_sum`
- `abs_score_diff`
- `is_deuce`
- `is_close_score`
- `is_self_leading`
- `is_other_leading`

### Player turn features

- `last_hitter_id`
- `last_opponent_id`
- `next_hitter_is_initial_server_side`

### Error / terminal proxy features

- `last_action_is_zero`
- `last_point_is_zero`
- `has_zero_action`
- `has_zero_point`

### Prefix count and ratio features

For the observed prefix, the generator also creates class counts and ratios for:

- `actionId`
- `pointId`
- `strikeId`
- `handId`
- `strengthId`
- `spinId`
- `positionId`

## Notes

- Do not use random row split. Use grouped validation by `match` or `rally_uid`.
- The stricter default is `GroupKFold` by `match`, because train and test have no
  match overlap.
- The reference-only old test data should not be used for official validation or
  training because it has leakage risk.

## Objective-aligned optimization update

The highest-level validation objectives are now:

```text
actionId: maximize Macro F1
pointId: maximize Macro F1
serverGetPoint: maximize ROC AUC
```

The LSTM trainer now supports objective-aligned early stopping:

```powershell
uv run python src\train_lstm.py --target actionId --preset action
uv run python src\train_lstm.py --target pointId --preset point
uv run python src\train_lstm.py --target serverGetPoint --preset server --no-use-sample-weight
```

Implementation details:

- `actionId` and `pointId` checkpoints are selected by validation Macro F1.
- `serverGetPoint` checkpoints are selected by validation ROC AUC.
- Manual features and sequence numeric features are standardized per fold.
- The LSTM manual-feature BatchNorm layer was removed to avoid double
  normalization and unstable early validation.
- LSTM OOF probabilities are exported for ensemble validation.

Metric-based ensemble validation:

```powershell
uv run python src\ensemble_validation.py
```

Current OOF ensemble results:

| Target | Objective | LSTM Weight | Accuracy | Macro F1 | Log Loss | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| `actionId` | Macro F1 | 0.30 | 0.46913 | 0.33891 | 1.54291 | — |
| `pointId` | Macro F1 | 0.65 | 0.26762 | 0.20690 | 1.91872 | — |
| `serverGetPoint` | ROC AUC | 0.20 | 0.57010 | 0.56908 | 0.67540 | 0.60167 |

Tabular-first submission generation:

```powershell
uv run python src\predict_tabular_submission.py
```

Output:

```text
submissions/submission_tabular_baseline.csv
```

The tabular submission applies an `actionId` structural mask: for test targets
with `target_strikeNumber >= 2`, serve classes `15..18` are zeroed before argmax.

## Step A validation diagnostics

Run objective-focused diagnostics:

```powershell
uv run python src\validation_diagnostics.py
```

Run diagnostics plus prefix-length ablation:

```powershell
uv run python src\validation_diagnostics.py --run-prefix-ablation --n-estimators 120
```

Outputs:

```text
reports/diagnostics/oof_slices.csv
reports/diagnostics/summary.json
reports/diagnostics/leakage_checks.json
reports/diagnostics/prefix_ablation.csv
```

Current Step A findings:

| Target | Source | Main Metric | All OOF | Test-weighted | Prefix <= 3 | Prefix <= 4 |
|---|---|---:|---:|---:|---:|---:|
| `actionId` | ensemble | Macro F1 | 0.33891 | 0.31735 | 0.31952 | 0.32651 |
| `pointId` | ensemble | Macro F1 | 0.20690 | 0.19177 | 0.18452 | 0.18829 |
| `serverGetPoint` | ensemble | ROC AUC | 0.60167 | 0.58407 | 0.57532 | 0.58045 |

Leakage findings:

- `source_rally_len` parity is nearly an oracle for train `serverGetPoint`
  (`AUC=0.99846`) and must remain excluded from all features.
- Saved tabular model features pass forbidden-feature checks.
- LSTM manual features pass forbidden-feature checks.
- Prefix length alone is not an effective server proxy (`prefix_len` raw
  `AUC=0.48743`).
- Prefix-length ablation shows minimal objective impact:
  - `pointId`: 0.19461 with prefix features vs. 0.19409 without.
  - `serverGetPoint`: 0.59821 with prefix features vs. 0.59819 without.

Interpretation:

- Test-weighted and short-prefix metrics are lower than all-OOF metrics, so the
  honest target estimate should use these diagnostics, not only all-OOF scores.
- The current validation gap supports a realistic improvement goal around
  `pointId` Macro F1 0.24–0.28 and `serverGetPoint` ROC AUC 0.63–0.68 without
  leakage, rather than the stretch targets 0.50 / 0.85.

## Step B pointId phase-specific optimization

Run the pointId phase-specific model experiment:

```powershell
uv run python src\train_point_phase_models.py --n-estimators 400 --blend-with-existing reports\ensemble\summary.csv
```

Outputs:

```text
models/point_phase/global.joblib
models/point_phase/receive.joblib
models/point_phase/third_ball.joblib
models/point_phase/early_rally.joblib
models/point_phase/rally.joblib

reports/point_phase/summary.json
reports/point_phase/per_bucket_metrics.csv
reports/point_phase/pointId_oof_proba.npy
reports/point_phase/pointId_oof_predictions.csv
reports/point_phase/pointId_blend_summary.csv
```

Buckets:

| Bucket | Definition |
|---|---|
| `receive` | `next_strikeNumber == 2` |
| `third_ball` | `next_strikeNumber == 3` |
| `early_rally` | `next_strikeNumber in {4, 5}` |
| `rally` | `next_strikeNumber >= 6` |

Step B pointId results:

| Point model | Macro F1 | Test-weighted Macro F1 | Prefix <= 3 | Prefix <= 4 |
|---|---:|---:|---:|---:|
| previous point ensemble | 0.20690 | 0.19177 | 0.18452 | 0.18829 |
| point phase only | 0.19764 | 0.18609 | 0.18467 | 0.18942 |
| phase blend, best all Macro F1 | 0.20932 | 0.19438 | 0.19144 | 0.19816 |
| phase blend, best weighted Macro F1 | 0.20909 | 0.19462 | 0.19189 | 0.19836 |

Interpretation:

- Phase-specific point models are weaker than the current point ensemble by
  themselves.
- They add useful diversity when blended.
- The best test-like blend currently uses `weight_point_phase=0.40`, improving
  weighted Macro F1 from `0.19177` to `0.19462` and short-prefix Macro F1 from
  `0.18452` to `0.19189`.

## Step C serverGetPoint stacking optimization

Run the server stacking experiment:

```powershell
uv run python src\train_server_stacking.py --model extratrees --n-estimators 400
```

Outputs:

```text
models/server_stacking/serverGetPoint_extratrees.joblib

reports/server_stacking/summary.json
reports/server_stacking/blend_summary.csv
reports/server_stacking/serverGetPoint_oof_proba.npy
reports/server_stacking/serverGetPoint_oof_predictions.csv
```

Stacking features:

- Safe tabular prefix features.
- `actionId` OOF probabilities from tabular, LSTM, and ensemble models.
- `pointId` OOF probabilities from tabular, LSTM, ensemble, point-phase, and
  point-phase blend models.
- Compact probability meta-features: argmax, max probability, entropy, and top-2
  margin.

Forbidden features excluded:

```text
sample_id
rally_uid
source_rally_len
sample_weight
target_strikeNumber
label_actionId
label_pointId
label_serverGetPoint
```

Step C server results:

| Server model | ROC AUC | Test-weighted ROC AUC | Prefix <= 3 AUC | Prefix <= 4 AUC |
|---|---:|---:|---:|---:|
| previous server ensemble | 0.60167 | 0.58407 | 0.58247 | 0.59198 |
| server stacking only | 0.60238 | 0.58640 | 0.58592 | 0.59411 |
| stack blend, best all AUC | 0.60473 | 0.58762 | 0.58653 | 0.59535 |
| stack blend, best weighted AUC | 0.60465 | 0.58776 | 0.58680 | 0.59547 |

Interpretation:

- Server stacking gives a small but consistent AUC improvement.
- The best all-OOF AUC uses `weight_server_stacking=0.55`.
- The best test-weighted AUC uses `weight_server_stacking=0.65`.

## Final blended submission

Generate the final blended submission:

```powershell
uv run python src\predict_final_blend.py
```

Output:

```text
submissions/submission_final_blend.csv
```

Final blend weights:

```text
actionId:
  0.70 * tabular_baseline
+ 0.30 * LSTM
  then mask serve classes 15..18 for target_strikeNumber >= 2

pointId:
  point_base = 0.35 * tabular_baseline + 0.65 * LSTM
  final_point = 0.60 * point_base + 0.40 * point_phase

serverGetPoint:
  server_base = 0.80 * tabular_baseline + 0.20 * LSTM
  final_server = 0.35 * server_base + 0.65 * server_stacking
```

Generated artifacts:

```text
reports/final_blend/summary.json
reports/final_blend/*_proba.npy
```

Sanity checks from the generated file:

```text
rows: 1,845
unique rally_uid: 1,845
columns: rally_uid, actionId, pointId, serverGetPoint
no NaNs: true
actionId range: 0..14 after serve-class mask
pointId range: 0..9
serverGetPoint range: 0..1
```

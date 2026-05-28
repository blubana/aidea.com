# AI CUP Rally Prediction

This project trains sequence and tabular models for AI CUP table-tennis rally prediction.

The three submission targets are:

- `actionId`: next-strike action class, evaluated by Macro F1.
- `pointId`: next-strike landing point class, evaluated by Macro F1.
- `serverGetPoint`: rally outcome probability, evaluated by AUC-ROC.

The overall validation score printed by the project is:

```text
score = 0.4 * F1_action + 0.4 * F1_point + 0.2 * AUC
```

## Project Layout

```text
data/dataset.py        Feature engineering, categorical encoding, sequence dataset builders
model/transformer.py   RoPE Transformer multi-task model
model/rnn.py           GRU/LSTM multi-task baseline
model/pcgrad.py        PCGrad gradient surgery helper
train.py               Transformer / GRU / LSTM training entrypoint
train_tabular.py       LightGBM / CatBoost / sklearn tabular training entrypoint
inference.py           Single-model and mixed ensemble inference
score_submission.py    Local scoring helper when ground truth is available
dataset/AI CUP競賽資料集/
  train.csv
  test_new.csv
  sample_submission.csv
```

## Data Assumptions

Default data directory:

```text
dataset/AI CUP競賽資料集
```

Expected files:

```text
train.csv
test_new.csv
sample_submission.csv
```

The project uses `test_new.csv` for final inference. The old reference test data is not used for submission.

## Validation Design

Use:

```text
--val-target-from-end 2
```

This is the default.

Reason: the terminal strike in training rallies has a degenerate `pointId` distribution. Almost every final strike has `pointId=0`, so validating on the final strike makes `F1_point` misleading. With `--val-target-from-end 2`, validation predicts the pre-terminal strike instead, which has a normal `pointId` distribution.

Do not use `--val-target-from-end 1` for model selection unless you are intentionally studying terminal-strike behavior.

## Feature Sets

Select with:

```text
--feature-set base|score|enhanced
```

Available presets:

```text
base:
  sex, handId, strengthId, spinId, actionId, pointId, positionId,
  strikeId, scoreSelf, scoreOther, strikeNumber

score:
  base features plus scoreDiff, scoreDiffBucket, scoreTotal, scorePhase,
  isTied, isLeading, isDeuceLike, isServeShot, shotParity, rallyLengthBucket

enhanced:
  score features plus actionPointCombo, spinStrengthCombo,
  lastActionId, lastPointId, lastStrikeId, lastPositionId
```

Recommended starting point:

```text
--feature-set base
```

Then try `score`, and only use `enhanced` if validation confirms it helps.

## Install / Environment

This project is intended to run with `uv`.

```powershell
uv sync
```

CUDA PyTorch is configured through `pyproject.toml` using the `pytorch-cu128` index.

## Train Transformer

Recommended stable baseline:

```powershell
uv run python train.py --device cuda --model-type transformer --epochs 30 --d-model 128 --layers 4 --ffn 512 --batch 128 --feature-set base --val-target-from-end 2 --action-loss-weight 0.35 --point-loss-weight 0.55 --rally-loss-weight 0.10 --selection-metric action_point_score --class-weight-mode sqrt --label-smoothing 0.0 --out-dir checkpoints_transformer_base_v2
```

Larger Transformer experiment:

```powershell
uv run python train.py --device cuda --model-type transformer --epochs 40 --d-model 256 --layers 6 --ffn 1024 --batch 96 --feature-set score --val-target-from-end 2 --action-loss-weight 0.35 --point-loss-weight 0.50 --rally-loss-weight 0.15 --selection-metric score --class-weight-mode sqrt --label-smoothing 0.0 --out-dir checkpoints_transformer_score_big
```

If GPU memory is insufficient, reduce:

```text
--batch 96 -> --batch 64
--d-model 256 -> --d-model 128
--layers 6 -> --layers 4
```

## Train GRU / LSTM

GRU:

```powershell
uv run python train.py --device cuda --model-type gru --epochs 30 --d-model 128 --layers 2 --batch 128 --feature-set base --val-target-from-end 2 --action-loss-weight 0.35 --point-loss-weight 0.55 --rally-loss-weight 0.10 --selection-metric action_point_score --class-weight-mode sqrt --label-smoothing 0.0 --out-dir checkpoints_gru_base_v2
```

LSTM:

```powershell
uv run python train.py --device cuda --model-type lstm --epochs 30 --d-model 128 --layers 2 --batch 128 --feature-set base --val-target-from-end 2 --action-loss-weight 0.35 --point-loss-weight 0.55 --rally-loss-weight 0.10 --selection-metric action_point_score --class-weight-mode sqrt --label-smoothing 0.0 --out-dir checkpoints_lstm_base_v2
```

Optional bidirectional RNN:

```text
--bidirectional-rnn
```

Use this only as an experiment; inference still uses the observed sequence, but bidirectional encoders can overfit more easily.

## Train Tabular Model

Tabular models flatten the last `--window` encoded strikes and train three independent classifiers:

- action classifier
- point classifier
- rally outcome classifier

Recommended command:

```powershell
uv run python train_tabular.py --backend auto --feature-set score --window 6 --max-iter 300 --val-target-from-end 2 --out-dir checkpoints_tabular_score_v2
```

Backend behavior:

```text
auto      Try LightGBM, then CatBoost, then sklearn fallback
lightgbm  Require lightgbm
catboost  Require catboost
sklearn   Use HistGradientBoostingClassifier
```

If LightGBM or CatBoost is not installed, use:

```powershell
uv run python train_tabular.py --backend sklearn --feature-set score --window 6 --max-iter 300 --val-target-from-end 2 --out-dir checkpoints_tabular_score_v2
```

## K-Fold Training

For neural models:

```powershell
uv run python train.py --device cuda --model-type transformer --epochs 30 --feature-set base --val-target-from-end 2 --cv-folds 5 --out-dir checkpoints_transformer_cv
```

For tabular models:

```powershell
uv run python train_tabular.py --backend auto --feature-set score --window 6 --cv-folds 5 --out-dir checkpoints_tabular_cv
```

K-fold training creates one best artifact per fold. Inference can ensemble them.

## Inference

Single neural checkpoint:

```powershell
uv run python inference.py --device cuda --checkpoint checkpoints_transformer_base_v2\best_transformer_YYYYMMDD_HHMMSS.pt --out-dir submissions
```

All best neural checkpoints in a directory:

```powershell
uv run python inference.py --device cuda --checkpoint-dir checkpoints_transformer_cv --checkpoint-glob best_transformer_*.pt --recursive --out-dir submissions
```

Tabular artifact:

```powershell
uv run python inference.py --device cuda --checkpoint checkpoints_tabular_score_v2\tabular_sklearn_YYYYMMDD_HHMMSS.pkl --out-dir submissions
```

Mixed ensemble:

```powershell
uv run python inference.py --device cuda --checkpoint-dir checkpoints_transformer_cv --checkpoint-glob best_transformer_*.pt --recursive --checkpoint checkpoints_gru_base_v2\best_transformer_YYYYMMDD_HHMMSS.pt checkpoints_tabular_score_v2\tabular_sklearn_YYYYMMDD_HHMMSS.pkl --out-dir submissions_ensemble
```

The ensemble averages:

```text
actionId: average class probabilities, then argmax
pointId: average class probabilities, then argmax
serverGetPoint: average positive-class probabilities
```

## Important Training Parameters

### Data / Validation

```text
--data-dir
  Directory containing train.csv, test_new.csv, sample_submission.csv.

--train
  Optional explicit training CSV path.

--feature-set
  Feature preset: base, score, enhanced.

--val-size
  Fraction of rallies used for validation when not using k-fold.

--val-target-from-end
  Validation target offset from rally end. Default 2 avoids terminal pointId collapse.

--cv-folds
  Number of stratified rally-level folds. Use >1 for k-fold training.

--limit-rallies
  Debug limit. Use 0 for full training.
```

### Model

```text
--model-type
  transformer, gru, or lstm.

--d-model
  Hidden size.

--layers
  Number of Transformer encoder layers or RNN layers.

--ffn
  Transformer feed-forward hidden size. Ignored by GRU/LSTM.

--nhead
  Transformer attention heads.

--emb-dim
  Embedding size per categorical feature.

--dropout
  Dropout probability.

--rally-pool
  Transformer rally head pooling: last_mean_mlp or mean_linear.
```

### Optimization

```text
--lr
  Base learning rate. Default 3e-4.

--lr-decay
  Layer-wise learning rate decay for neural models.

--weight-decay
  AdamW weight decay.

--grad-clip
  Gradient clipping norm.

--scheduler
  onecycle or none.

--onecycle-pct-start
  Warmup fraction for OneCycleLR.

--early-stopping-patience
  Stop after this many epochs without selection metric improvement. 0 disables.
```

### Multi-Task Loss

```text
--action-loss-weight
  Training loss weight for actionId.

--point-loss-weight
  Training loss weight for pointId.

--rally-loss-weight
  Training loss weight for serverGetPoint.

--pcgrad / --no-pcgrad
  Enable or disable PCGrad.

--pcgrad-rally-mode
  separate: PCGrad only action/point, rally loss backward separately.
  together: PCGrad all three losses together.

--label-smoothing
  Label smoothing for actionId and pointId cross entropy.

--class-weight-mode
  none, inverse, sqrt, effective.

--class-weight-max
  Cap for class weights.
```

### Checkpoint Selection

```text
--selection-metric score
  Select best checkpoint by 0.4 action F1 + 0.4 point F1 + 0.2 AUC.

--selection-metric action_point_score
  Select by 0.5 action F1 + 0.5 point F1.

--selection-metric min_f1
  Select by min(action F1, point F1).
```

Recommended when stabilizing `pointId`:

```text
--selection-metric action_point_score
```

Recommended when optimizing final score after action/point are stable:

```text
--selection-metric score
```

## Log Fields

Training prints:

```text
F1_action
F1_point
AUC
ap_score
score
select
dense_score
point_pred
top_point
```

Meaning:

```text
ap_score
  0.5 * F1_action + 0.5 * F1_point.

score
  0.4 * F1_action + 0.4 * F1_point + 0.2 * AUC.

select
  The metric currently used to save best checkpoint.

dense_score
  Score on dense sliding-window validation samples.

point_pred=x/y
  x = number of pointId classes predicted by the model.
  y = number of true pointId classes in validation.

top_point
  Fraction of predictions assigned to the most frequent predicted point class.
```

Healthy `pointId` behavior usually looks like:

```text
point_pred=8/10 to 10/10
top_point < 0.50
```

Potential collapse:

```text
point_pred=1/10
top_point > 0.70
```

## Suggested Workflow

1. Train a stable Transformer baseline with `feature-set base`.
2. Train GRU and/or LSTM baselines with the same validation setting.
3. Train a tabular model with `feature-set score`.
4. Compare validation logs, especially `F1_point`, `AUC`, and `point_pred`.
5. Use mixed ensemble inference for final submission.

Start conservative, then add complexity:

```text
base feature set -> score feature set -> enhanced feature set
medium model -> larger model
single split -> k-fold ensemble
```


i am gay
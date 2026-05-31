# Metric Improvement Assessment

This note captures the current priority recommendation and oracle-style risk assessment for the next improvement sprint.

## Executive summary

- The highest-confidence gains still come from **safe ensembling and calibration**, not from expanding raw feature space.
- `source_rally_len` remains a **forbidden leakage feature** because train-only parity is nearly an oracle for `serverGetPoint`.
- Short-prefix and test-weighted slices remain the honest guide for iteration priority.

## Ranked recommendation table

| Priority | Workstream | Feasibility | Expected benefit | Leakage / risk | Implementation complexity | Decision |
|---|---|---:|---:|---|---:|---|
| P0 | Submit `serverGetPoint` as probability (`server_final[:, 1]`) with a bool/float switch | Very high | High | Requires confirming the competition accepts floats | Very low | Implement first; default to float while preserving bool mode |
| P0 | Remove `sample_weight` from ExtraTrees input features while preserving fit-time weights | Very high | Medium | Low; fixes train/test covariate mismatch | Very low plus retrain | Implement and retrain/reblend |
| P1 | Add Macro-F1 class probability multipliers / logit bias for `actionId` and `pointId` | High | Medium | OOF tuning can be mildly optimistic | Low-medium | Implement as post-hoc OOF search and opt-in/recorded final transform |
| P1 | Extend phase-aware blending to `actionId`, including CatBoost phase weights | High | Medium | Low | Medium | Implement after P0 validation |
| P1 | Add optional OOF cross-target stacking | Medium-high | Medium, especially for `pointId` | Must verify OOF row/fold alignment | Medium | Gate behind explicit switch |
| P2 | Fix prefix weighting to `P_test(prefix_len=k) / P_train(prefix_len=k)` | Medium | Small-medium | Low | Medium-high full pipeline retrain | Defer until P1 is measured |
| P2 | Rally-balanced `serverGetPoint` training | Medium | Small | May overlap with prefix weighting | Medium | Evaluate after weighting cleanup |
| P2 | Per-target tuning and XGBoost diversity | High | Small-medium | Low | High | Use if cheap fixes plateau |
| P2 | Remove or transform raw IDs such as `match`, `rally_id`, player IDs | Medium | Uncertain | Could remove useful player-overlap signal | Medium | Run ablation/feature importance before applying |
| P3 | Add transition features | High | Small-medium | Low | Medium | Later feature-engineering sprint |
| P3 | Hierarchical `pointId` via depth/side | Medium | Uncertain | More moving parts | Medium-high | Later experiment only |

## Oracle / leakage assessment

### Confirmed constraints

- `source_rally_len` parity is nearly an oracle for train `serverGetPoint` (`AUC ≈ 0.99846`), so it must stay excluded.
- `sample_weight` should be used only as a training weight, not as a feature.
- Test-like slices (`prefix_len <= 3`, `prefix_len <= 4`, weighted metrics) remain below all-OOF scores, so they should drive decisions.

### Practical implication

The best near-term path is to **improve combination logic among already-safe predictors**, not to loosen feature controls.

## Recommended sprint plan

### Sprint 1: fix the current submission path

1. Support `serverGetPoint` submission as either probability (`float`) or hard label (`bool`) to match competition validation needs. Use `float` by default because ROC AUC is ranking-based.
2. Remove `sample_weight` from ExtraTrees feature columns while keeping it as a training weight.
3. Retrain ExtraTrees, re-run blend search, regenerate the final submission, and record all updated metrics.

### Sprint 2: honest-metric blend tuning

1. Add Macro-F1 class multiplier search for `actionId` and `pointId` on OOF probabilities.
2. Re-optimize CatBoost blend weights against weighted / short-prefix slices after the ExtraTrees retrain.
3. Add phase-aware action blending if multiplier gains are not enough.

### Sprint 3: bounded cross-target stacking

1. Add optional cross-target stacking switches behind safe defaults.
2. Test only with already-generated model probabilities.
3. Reject any variant that weakens weighted or short-prefix metrics, even if all-OOF improves.

## Validation protocol

Every change must report all of the following before being promoted to the final submission path:

- Full OOF metric.
- Test-prefix-weighted OOF metric.
- `prefix_len <= 3` slice.
- `prefix_len <= 4` slice.
- Fold-level stability.
- Forbidden-feature checks for IDs, labels, `source_rally_len`, `target_strikeNumber`, and `sample_weight` as applicable.

## Success criteria for the next sprint

- No forbidden features in saved model bundles.
- Submission generation supports both server probability and boolean output modes.
- Weighted / short-prefix metrics do not regress while tuning blend weights.

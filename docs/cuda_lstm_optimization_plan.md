# CUDA/LSTM optimization plan

Scope: `brandnew` branch, LSTM-heavy PyTorch training on an RTX 4070 Laptop GPU
with 8 GB VRAM. The plan is intentionally measurement-first: enable one change
at a time, keep the run reproducible, and retain only settings that improve
throughput or memory without degrading validation metrics.

## 1. Environment confirmation

- Print `torch.__version__`, `torch.version.cuda`, CUDA availability, GPU name,
  CUDA device capability, and cuDNN version at run start.
- Confirm the installed PyTorch build is CUDA-enabled, not CPU-only.
- Confirm the NVIDIA driver is new enough for the CUDA runtime bundled with the
  PyTorch wheel.

## 2. Baseline benchmark

- Run a short controlled baseline before enabling TF32, AMP, or cuDNN benchmark.
- Fix seed, batch size, sequence length policy, dataset slice, warmup steps, and
  measured training steps.
- Record step time, samples/sec, peak VRAM, loss curve, validation metric, and
  device metadata.

## 3. Confirm `nn.LSTM` / cuDNN fast path

- Prefer `torch.nn.LSTM` over a Python loop with `LSTMCell`.
- Keep sequence tensors on CUDA during training.
- Preserve the current `pack_padded_sequence` path for variable-length rallies.
- If a custom recurrent cell becomes necessary, benchmark it separately before
  replacing the cuDNN-backed implementation.

## 4. TF32 isolated test

- Test `torch.backends.cuda.matmul.allow_tf32 = True` and
  `torch.backends.cudnn.allow_tf32 = True` in isolation.
- Expect modest LSTM gains, not Transformer-level acceleration.
- Retain if step time improves by roughly 3-5% and validation does not regress.

## 5. cuDNN benchmark isolated test

- Test `torch.backends.cudnn.benchmark = True` only when input shapes are stable.
- Disable it for highly variable sequence lengths, dynamic padding patterns, or
  workloads that repeatedly retrigger autotuning.
- Retain only if epoch time improves consistently.

## 6. BF16 AMP test

- Prefer BF16 AMP first on Ada GPUs because it has a wider dynamic range than
  FP16 and usually does not require `GradScaler`.
- Monitor NaN/Inf, validation metric drift, peak VRAM, and whether a larger batch
  size becomes possible.

## 7. FP16 AMP fallback

- Test FP16 with `GradScaler` if BF16 is slower or unavailable.
- Watch loss-scale drops, NaN/Inf, and LSTM numerical stability.
- Unscale gradients before gradient clipping.

## 8. Combination matrix

Run the following candidates and compare against baseline:

| Candidate | TF32 | cuDNN benchmark | AMP |
|---|---:|---:|---|
| baseline | off | off | off |
| tf32 | on | off | off |
| benchmark | off | on | off |
| bf16 | off | off | BF16 |
| bf16_tf32 | on | off | BF16 |
| candidate | on | on | BF16 |

Choose the simplest stable configuration with the best validation-safe
throughput/VRAM trade-off.

## 9. Profiling

- Use PyTorch Profiler first, then Nsight Systems if the bottleneck is unclear.
- Check for GPU idle gaps, host-to-device copy time, CPU launch overhead,
  `aten::lstm` / cuDNN RNN kernels, and dataloader stalls.

## 10. Data pipeline optimization

- Tune `DataLoader` with `num_workers`, `pin_memory`, `persistent_workers`, and
  `prefetch_factor` only after profiling shows input stalls.
- Move batches with `non_blocking=True` when pinned memory is enabled.
- Consider DALI only if preprocessing is heavy enough to justify the added
  complexity.

## 11. CUDA Graphs + bucketing

- Consider CUDA Graphs only if profiling shows CPU launch overhead and shapes can
  be made fixed.
- For variable-length LSTM batches, group sequence lengths into buckets and pad
  to fixed bucket sizes before graph capture.

## 12. Deployment and scale-out options

- TensorRT: evaluate after training if low-latency FP16/INT8 inference matters.
- Triton Inference Server: evaluate for production serving with dynamic batching.
- NCCL/DDP: evaluate only for multi-GPU training; single-GPU runs do not need it.
- RAPIDS/cuDF: evaluate only if large tabular/time-series preprocessing becomes a
  bottleneck.

## Initial implementation hooks

`src/train_lstm.py` exposes `--cuda-opt-profile` presets for controlled tests:

```powershell
uv run python src\train_lstm.py --target serverGetPoint --sample 2048 --max-folds 1 --epochs 1 --cuda-opt-profile baseline
uv run python src\train_lstm.py --target serverGetPoint --sample 2048 --max-folds 1 --epochs 1 --cuda-opt-profile tf32
uv run python src\train_lstm.py --target serverGetPoint --sample 2048 --max-folds 1 --epochs 1 --cuda-opt-profile bf16
uv run python src\train_lstm.py --target serverGetPoint --sample 2048 --max-folds 1 --epochs 1 --cuda-opt-profile candidate
```

The generated metrics JSON includes CUDA settings, device metadata, epoch timing,
throughput, and peak VRAM for comparison.

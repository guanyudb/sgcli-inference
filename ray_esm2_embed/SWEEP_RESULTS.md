# H100 Optimization Sweep — 100K UniRef50 sequences

Config base: 1 H100 node (8 GPU), ESM-2 650M FP16, seq_len=1024, 100K records on e2FE.

| # | Change | BATCH | COMPILE | RAY_BLOCK | PRETOK | wall (s) | tput (seq/s) | per-GPU | mean util | sat ≥90% | mem max | vs #0 |
|---|--------|-------|---------|-----------|--------|----------|--------------|---------|-----------|----------|---------|-------|
| 0 | baseline | 32 | 0 | default | 0 | 152.6 | 655.4 | 81.9 | 57.9% | 35% | 6% | 1.00x |
| 1 | larger batch | 64 | 0 | default | 0 | 163.3 | 612.2 | 76.5 | 57.1% | 48% | 9% | 0.93x |
| 2 | larger batch | 128 | 0 | default | 0 | 163.6 | 611.2 | 76.4 | 59.1% | 52% | 15% | 0.93x |
| 3 | + Ray prefetch | 128 | 0 | 256 MB | 0 | 161.5 | 619.2 | 77.4 | 60.3% | 55% | 15% | 0.94x |
| 4 | + torch.compile | 128 | 1 | default | 0 | 227.2 | 440.2 | 55.0 | 27.5% | 25% | 11% | 0.67x |
| **5** | **+ pretokenize** | **128** | 0 | default | **1** | **125.4** | **797.3** ⭐ | **99.7** | **76.4%** | **73%** | 15% | **1.22x** |
| 6 | + bigger batch | 256 | 0 | default | 1 | 126.4 | 791.3 | 98.9 | 77.1% | 74% | 27% | 1.21x |

## Bottleneck verdict

**Tokenization in the actor's hot path was the throughput limiter**, not GPU compute.

Evidence:
- Bumping batch size (32 → 256) without pretokenization changed nothing (~611 seq/s, ~58% util)
- Pretokenizing alone (with B=128) jumped throughput **+30%** (611 → 797 seq/s) and **util from 59% → 76%**
- B=256 + pretok plateaued at B=128's level — compute now caught up to data feed
- Remaining 23% util gap is likely Ray write back-pressure + model-load amortization (15s × 8 actors / 125s ≈ 12% of wall)

## Recommendation for the customer

| Optimization | Impact | Effort |
|---|---|---|
| **Pre-tokenize upstream** (offline batch job → tokenized parquet) | **+22% throughput, +18 pts util** | One-time Spark/sgcli job per dataset |
| Larger batch alone | ~0 improvement | None — don't bother below pretok |
| `torch.compile` (default mode) | **-33% throughput** | Hurts on dynamic-shape inputs; only useful with fixed-shape padding + `dynamic=True` |
| Ray block size tuning | +1% (margin of noise) | Negligible |

## Optimal config we found

```yaml
env_variables:
  NUM_WORKERS: "8"
  BATCH_SIZE: "128"     # 256 doesn't add value but doesn't hurt either
  PRETOKENIZED: "1"     # the key knob
  INPUT_STAGE_DIR: "/Volumes/<cat>/<schema>/sgc/<dataset>_tokenized"
compute:
  gpus: 8
  gpu_type: h100
```

With pretokenized input pipeline expected throughput at production scale (500K+):
- Per-GPU: ~110 seq/s (after model-load amortization)
- Aggregate (8×H100): **~880 seq/s**, ~84% util
- 500K wall: ~10 min embed

For Teddy 400M (longer sequences but fewer params), expect roughly the same recipe to apply — the **CPU tokenization bottleneck is even more pronounced** because Teddy's tokenizer has heavier per-sample work (bio annotations + binning). Pretokenization will be an even bigger lever there.

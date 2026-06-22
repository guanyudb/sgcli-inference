# sgcli-inference

Inference-focused test suites for Databricks Serverless GPU (SGC). Three angles:

| Suite | What it tests | Status |
|-------|---------------|--------|
| [`dcs_teddy/`](dcs_teddy/) | Custom Docker image (DCS) inference — runs `run_inference.py` baked into a teddy-inference image on SGC | ✅ passing on SGCCLI, e2FE |
| [`ray_smoke/`](ray_smoke/) | Minimal Ray + Ray Data + parquet I/O on the orchestrator pod, no model | not yet run |
| [`ray_esm2_embed/`](ray_esm2_embed/) | ESM-2 batch embedding via Ray Data — Srijit's notebook ported to a sgcli-runnable script | ✅ passing on **e2FE only** (1 A10, 1000 UniRef50 seqs, ~75s embed wall, 13.4 seq/s) |

## Why three suites

- **DCS** — does my custom-built container work on SGC end-to-end?
- **Ray smoke** — does Ray init + Ray Data + parquet I/O work without notebook context?
- **Ray ESM-2** — does the production embedding pattern work end-to-end?

If a customer's job fails, running these three tells you whether the issue is in (a) their image, (b) their Ray usage, or (c) the platform.

## Setup once

```bash
# Authenticate your Databricks profile
databricks auth login --host https://<workspace>.cloud.databricks.com --profile <YOUR_PROFILE>

# Install a DCS-capable sgcli wheel (0.1.0+)
uv tool install --python 3.12 /path/to/databricks_serverless_gpu_cli-0.1.0-py3-none-any.whl

# Create a UC Volume for staging
databricks api post /api/2.1/unity-catalog/volumes --profile <YOUR_PROFILE> \
  --json '{"catalog_name":"<catalog>","schema_name":"<schema>","name":"sgc","volume_type":"MANAGED"}'
```

For each `train_workload.yaml`, replace the `<UPDATE_THIS_to_local_path_of_sgcli-inference>` placeholder with your local clone path (once per machine).

## Running each suite

```bash
# 1. DCS teddy inference (~3 min — image already pushed by Srijit)
sgcli run -f dcs_teddy/inference_workload.yaml -p <PROFILE> --watch

# 2. Ray smoke (~5 min — uses env v5 base, 2 A10 workers)
sgcli run -f ray_smoke/train_workload.yaml -p <PROFILE> --watch

# 3. Ray ESM-2 (~5 min for 1000 seqs)
#    Step A: stage real UniRef50 protein sequences (HuggingFace mirror of UniProt)
pip install datasets   # one-time
python ray_esm2_embed/prep_data.py \
  --output /Volumes/<cat>/<schema>/sgc/ray_input_stage \
  --n 1000 --profile <PROFILE>

#    Step B: submit the job (also update INPUT_STAGE_DIR / OUTPUT_STAGE_PARQUET
#    in train_workload.yaml to match your volume)
sgcli run -f ray_esm2_embed/train_workload.yaml -p <PROFILE> --watch
```

## Benchmark sweep (10K UniRef50 sequences, e2FE)

Run the three preset benchmarks back-to-back to validate scaling:

```bash
# Stage 10K seqs once
python ray_esm2_embed/prep_data.py \
  --output /Volumes/<cat>/<schema>/sgc/ray_input_stage_10k \
  --n 10000 --profile <PROFILE>

# Then submit each in turn
sgcli run -f ray_esm2_embed/bench_1a10.yaml  -p <PROFILE> --watch
sgcli run -f ray_esm2_embed/bench_2a10.yaml  -p <PROFILE> --watch
sgcli run -f ray_esm2_embed/bench_1h100.yaml -p <PROFILE> --watch
```

Measured results on e2FE:

| Config | Layout | Embed wall | Aggregate tput | Per-GPU tput | vs 1-A10 |
|--------|--------|-----------|----------------|--------------|----------|
| 1× A10 | 1 node × 1 GPU | 511 s | 19.6 seq/s | 19.6 | 1.00x |
| 2× A10 (sharded) | 2 nodes × 1 GPU | 270 s | 37.1 seq/s | 18.6 | 1.89x |
| 1 H100 node | 1 node × 8 GPU | 38.7 s | **258.2 seq/s** | 32.3 | **13.2x** |

Notes:
- 2× A10 scaling is ~95% efficient (1.89x of 2.0x ideal)
- H100 per-GPU is ~1.65x A10 (less than typical 3-4x for transformers — ESM-2 650M with bs=32, seq_len=1024 in FP16 is memory-bandwidth bound, not compute bound)
- Earlier 1 A10 / 1K seq result was 13.4 seq/s — lower because per-batch overhead wasn't amortized

## Known platform gotchas (the ones we hit)

### Ray / `serverless_gpu` constraints

1. **`@ray_launch(...).distributed()` is notebook-only.** It calls `set_databricks_credentials_from_dbutils()` which needs `cluster_id` from a notebook context. From an `sgcli run` script, this fails with `cluster_id is required in the configuration`. **Workaround:** bypass `@ray_launch` — just call `ray.init()` directly. See `ray_esm2_embed/embed.py` for the pattern. True multi-A10 Ray scaling currently requires running from a Databricks notebook.

2. **Even `remote=False` triggers the dbutils path.** `remote=True/False` only controls remote pod spawning — the `.distributed()` method itself always tries the dbutils credential lookup. Don't use `@ray_launch.distributed()` from sgcli at all.

### Dependency constraints

3. **Don't upgrade `torch` via `dependencies.yaml`** unless you also upgrade `torchvision` to a matching version. Base env ships torch 2.7.1 + torchvision 0.22.1 paired. Pinning just `torch` upgrades it alone, and you get `RuntimeError: operator torchvision::nms does not exist` at runtime. Easiest fix: don't list torch — let pip use the base.

4. **`version: '5'` (AI env preview) silently falls back to `version: '4'`** on workspaces where v5 isn't enabled. The runtime did NOT include torch/ray as the docs claimed for v5 in our tests. Pin what you need explicitly until v5 is GA.

5. **Pinned old torch versions** (e.g. Srijit's `torch==2.3.1`) trigger heavy pip resolution that can cause `WorkloadPodFailure` with no user logs. Reproduced on SGCCLI. Use the base torch instead.

### Compute layout

6. **A10 = 1 GPU per node.** `gpus: N gpu_type: a10` = N separate nodes. With our `ray.init()` local-pod approach, only the orchestrator pod runs the model; the extra A10 pods sit idle. For real multi-A10, you need `@ray_launch(remote=True)` from a notebook.

7. **H100 = 8 GPUs per node.** `gpus: 8 gpu_type: h100` = 1 node, 8 GPUs.

### Workspace differences

8. **SGCCLI workspace (e2-dogfood-staging) was unreliable for `ray-esm2-embed`** at time of testing — 4 consecutive submissions returned `WorkloadPodFailure` with no logs. Single-A10 hello world worked fine on the same workspace. Root cause not identified. e2FE (e2-demo-field-eng) ran the same workload successfully. Use `train_workload_e2fe.yaml` as the known-working reference.

### Other

9. **WORKDIR is NOT honored** at runtime inside DCS containers. Use absolute paths in `command:`.
10. **DCS supports Docker Hub only** in private preview. No ECR/GCR/GHCR. AWS + Azure only.
11. **UC Volume FUSE write from SGC jobs** is unreliable for arbitrary `os.makedirs`. For shared write paths, use the Databricks SDK from the orchestrator side.
12. **`$CODE_SOURCE_PATH`** already includes the repo's last directory. Don't duplicate it in the `command:` block.
13. **`NCCL_NET_PLUGIN: "none"`** silences A10 EFA OFI probe warnings (A10 has no EFA hardware).

## What the suites share

| Convention | Detail |
|------------|--------|
| `train_workload.yaml` | sgcli job spec |
| `dependencies.yaml` | pip pins (where applicable) |
| Snapshot mode | `code_source: snapshot` with `include_paths: [<suite>]` |
| Placeholder | `<UPDATE_THIS_to_local_path_of_sgcli-inference>` — sed once per machine |

## File reference

```
sgcli-inference/
├── README.md                          # this file
├── .gitignore
│
├── dcs_teddy/                         # DCS image inference
│   ├── Dockerfile
│   ├── Dockerfile.inference.sgc       # TEDDY-G image (400M weights baked in)
│   ├── train.py, train_workload.yaml  # GPU sanity inside container
│   ├── inference_workload.yaml        # the working A10x8 inference command
│   └── run_inference.py
│
├── ray_smoke/                         # Ray platform smoke test
│   ├── ray_smoke.py
│   ├── train_workload.yaml
│   └── dependencies.yaml
│
└── ray_esm2_embed/                    # ESM-2 batch embedding (working)
    ├── embed.py                       # bypasses @ray_launch, uses ray.init()
    ├── prep_data.py                   # UniRef50 / SwissProt / synthetic
    ├── train_workload.yaml            # generic — update volume paths
    ├── train_workload_e2fe.yaml       # tested working on e2FE
    ├── dependencies.yaml              # ray[data] + transformers + pyarrow
    └── 03_batch_embed_sequences_ray.py  # Srijit's original notebook
```

# sgcli-inference

Inference-focused test suites for Databricks Serverless GPU (SGC). Three angles:

| Suite | What it tests | Status |
|-------|---------------|--------|
| [`dcs_teddy/`](dcs_teddy/) | Custom Docker image (DCS) inference — runs `run_inference.py` baked into a teddy-inference image on SGC | passing |
| [`ray_smoke/`](ray_smoke/) | `serverless_gpu.ray.ray_launch` platform pieces — Ray cluster bringup + Ray Data + distributed parquet write, no model | not yet run |
| [`ray_esm2_embed/`](ray_esm2_embed/) | Production-shaped Ray Data ESM-2 batch embedding (Srijit's notebook ported to a sgcli-runnable script) | not yet run |

## Why three suites

Each angle answers a different question:

- **DCS** — does my custom-built container work on SGC end-to-end? (snapshot upload, image registration, image pull on the node, run script discovery, multi-process torchrun)
- **Ray smoke** — does `@ray_launch` work? Does Ray Data + parquet I/O work? Without burning real model compute.
- **Ray ESM-2** — does the production pattern work end-to-end? Same model + math as Srijit's customer notebook, but driven by `sgcli run` (not a Databricks notebook).

If a customer's job fails, running these three tells you whether the issue is in (a) their image, (b) their Ray usage, or (c) the platform.

## Setup once

```bash
# Authenticate your Databricks profile
databricks auth login --host https://<workspace>.cloud.databricks.com --profile <YOUR_PROFILE>

# Make sure you have an sgcli wheel installed (DCS-capable: 0.1.0+)
uv tool install --python 3.12 /path/to/databricks_serverless_gpu_cli-0.1.0-py3-none-any.whl

# Make sure you have a UC Volume to use as staging (one-time)
databricks api post /api/2.1/unity-catalog/volumes --profile <YOUR_PROFILE> \
  --json '{"catalog_name":"main","schema_name":"<schema>","name":"sgc","volume_type":"MANAGED"}'
```

## Running each suite

Each suite has its own subfolder. From the repo root:

```bash
# 1. DCS teddy inference (~3 min — image is already pushed by Srijit)
sgcli run -f dcs_teddy/train_workload.yaml -p <PROFILE> --watch

# 2. Ray smoke (~5 min — uses env v5 base image, 2 A10 workers)
sgcli run -f ray_smoke/train_workload.yaml -p <PROFILE> --watch

# 3. Ray ESM-2 (~10-20 min depending on N; requires data prep first)
#    Step A: stage protein sequences to UC Volume.
#    Default pulls REAL data from UniRef50 (HuggingFace mirror of UniProt
#    reference clusters at 50% identity — what ESM-2 was trained on).
pip install datasets   # one-time, for the uniref50 source
python ray_esm2_embed/prep_data.py \
  --output /Volumes/<cat>/<schema>/sgc/ray_input_stage \
  --n 1000 \
  --profile <PROFILE>
#    Other sources: --source swissprot (570K curated proteins from UniProt),
#                    --source synthetic (random sequences, no network)
#
#    Step B: submit the job
sgcli run -f ray_esm2_embed/train_workload.yaml -p <PROFILE> --watch
```

For each `train_workload.yaml`, replace the `<UPDATE_THIS_to_local_path_of_repo>` placeholder with your local clone's absolute path (one-time per machine).

## What the suites share

| Convention | Detail |
|------------|--------|
| `train_workload.yaml` | sgcli job spec; one per suite |
| `dependencies.yaml` | pip pins (where applicable) |
| Snapshot mode | `code_source: snapshot` with `include_paths: [<suite>]` so only the test code is uploaded, not the whole repo |
| Env v5 default | Where no docker image is specified, `version: '5'` selects the AI environment (torch 2.9, transformers 4.57, Ray 2.51, vLLM 0.13). Single knob to override base. |
| A10 noise muffle | `NCCL_NET_PLUGIN: "none"` silences EFA OFI probe warnings on A10 |

## Known platform gotchas

(Carried over from the streaming suite — same SGC environment, same surprises.)

- **A10 is 1 GPU per node.** `gpus: N gpu_type: a10` = N nodes. Use `--nnodes=N --nproc_per_node=1` if you torchrun.
- **H100 is 8 GPU per node.** `gpus: 8 gpu_type: h100` = 1 node.
- **UC Volume FUSE write from SGC jobs** is not reliable for arbitrary `os.makedirs` operations. For shared write paths, use the Databricks SDK from the orchestrator side.
- **WORKDIR is NOT honored** at runtime inside DCS containers. Use absolute paths in `command:`.
- **DCS supports Docker Hub only** in private preview. No ECR/GCR/GHCR. AWS + Azure only, no GCP.
- **MLflow / Ray / Streaming all use distributed barriers** internally. Single-process tests inheriting SGC's `WORLD_SIZE` env will hang. Override for single-proc work.

## File reference

```
sgcli-inference/
├── README.md                    # this file
├── .gitignore
│
├── dcs_teddy/                   # DCS image inference (existing working test)
│   ├── README.md                # build/push/register/run details
│   ├── Dockerfile               # base reference (databricksruntime/air)
│   ├── Dockerfile.inference.sgc # the actual TEDDY-G image (bakes in 400M weights)
│   ├── train.py                 # GPU sanity inside container
│   ├── train_workload.yaml      # uses srijitnair254/teddy-inference image
│   ├── inference_workload.yaml  # the working A10x8 inference command
│   └── run_inference.py         # TEDDY-G embedding pipeline (CPU+CUDA)
│
├── ray_smoke/                   # Ray platform smoke test
│   ├── ray_smoke.py
│   ├── train_workload.yaml
│   └── dependencies.yaml        # env v5, no extras
│
└── ray_esm2_embed/              # Full ESM-2 batch embedding via Ray Data
    ├── embed.py                 # ported from Srijit's notebook
    ├── prep_data.py             # generates + uploads synthetic input data
    ├── train_workload.yaml
    ├── dependencies.yaml        # env v5 + pyarrow
    └── 03_batch_embed_sequences_ray.py  # original notebook for reference
```

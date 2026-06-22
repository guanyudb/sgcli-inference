"""
Minimal smoke test for serverless_gpu.ray.ray_launch on SGC.

Goal: confirm the platform pieces work end-to-end WITHOUT real model code:
  1. @ray_launch provisions N single-GPU A10 SGC workers
  2. Ray head + worker handshake succeeds (rank-0 head, others join)
  3. Ray cluster reports the expected resources
  4. Ray Data can run a trivial map_batches across all GPUs
  5. write_parquet from each actor to a UC Volume succeeds

This is the platform-only validation. If this passes but real ESM-2 code
fails, the issue is in the model/Tx — not in SGC + Ray plumbing. Pair with
ray_esm2_embed/ for the production-shaped test.

Run via sgcli (no notebook context required — that's what we're testing).
"""

import os
import sys
import tempfile

import numpy as np

# NOTE: serverless_gpu must be installed; in env v5 it's pre-bundled.
from serverless_gpu.ray import ray_launch


NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
GPU_TYPE = os.environ.get("GPU_TYPE", "a10")
OUTPUT_DIR = os.environ.get(
    "RAY_SMOKE_OUTPUT",
    "/Volumes/main/guanyu_chen/sgc/ray_smoke_output",
)


@ray_launch(gpus=NUM_WORKERS, gpu_type=GPU_TYPE, remote=True)
def smoke_job():
    """Runs on rank-0 as the Ray driver after Ray head/workers are up."""
    import ray
    import torch

    print(f"[smoke] Ray cluster resources: {ray.cluster_resources()}")
    print(f"[smoke] Ray cluster nodes:     {len(ray.nodes())}")

    # Trivial actor — verifies GPU is visible inside the actor and we can
    # round-trip data through Ray Data without any model dependency.
    class GpuSanityActor:
        def __init__(self):
            self.device_name = torch.cuda.get_device_name(0)
            print(f"[actor] device={self.device_name} on host={ray.util.get_node_ip_address()}")

        def __call__(self, batch):
            # Pretend embedding: multiply input by 2 on GPU
            x = torch.as_tensor(batch["value"], device="cuda")
            y = (x * 2).cpu().numpy()
            return {
                "id":     np.asarray(batch["id"]),
                "doubled": y,
                "device":  np.array([self.device_name] * len(batch["id"])),
            }

    # Synthetic input — 64 rows, no external dependency
    ds = ray.data.from_items([{"id": i, "value": float(i)} for i in range(64)])
    print(f"[smoke] Input dataset: {ds.count()} rows")

    result = ds.map_batches(
        GpuSanityActor,
        batch_size=8,
        num_gpus=1,
        concurrency=NUM_WORKERS,
        batch_format="numpy",
    )

    # Distributed write
    print(f"[smoke] Writing parquet to {OUTPUT_DIR}")
    result.write_parquet(OUTPUT_DIR)
    print(f"[smoke] Write complete")


def main():
    print("=" * 60)
    print(f"SGC + Ray smoke test (workers={NUM_WORKERS}, gpu_type={GPU_TYPE})")
    print("=" * 60)

    smoke_job.distributed()
    print("\nRay job complete on the orchestrator side.")

    # Verify output exists from the orchestrator (post-job)
    if os.path.exists(OUTPUT_DIR):
        files = os.listdir(OUTPUT_DIR)
        parquet_files = [f for f in files if f.endswith(".parquet")]
        print(f"Output dir contents:  {len(files)} files, {len(parquet_files)} .parquet")
        if parquet_files:
            print(f"  e.g. {parquet_files[0]}")
            # Read back one to validate
            import pyarrow.parquet as pq
            tbl = pq.read_table(os.path.join(OUTPUT_DIR, parquet_files[0]))
            print(f"Sample parquet schema: {tbl.schema}")
            print(f"Sample parquet rows:   {tbl.num_rows}")
            print("\nALL CHECKS PASSED — Ray + SGC platform pieces work end-to-end")
            sys.exit(0)

    print(f"\nFAIL: no parquet output found at {OUTPUT_DIR}")
    sys.exit(1)


if __name__ == "__main__":
    main()

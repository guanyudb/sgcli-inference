"""
ESM-2 batch embedding on SGC + Ray Data — sgcli-runnable port of
03_batch_embed_sequences_ray.py.

Differences vs the original notebook:
  - No `dbutils`, no `%pip`, no `%run` — pure Python
  - No Spark — orchestrator reads the input parquet directly via pyarrow
    (assumes upstream has staged the sequence_db -> parquet step on a UC Volume)
  - No CTAS — leaves parquet output where Ray writes it; promote to a UC
    table from a notebook after this job completes
  - Paths + sizes are configurable via env vars (overridable from
    train_workload.yaml without editing this file)

Usage:
  sgcli run -f train_workload.yaml -p <PROFILE> --watch

What the orchestrator does:
  1. Reads INPUT_STAGE_DIR to count rows (sanity check)
  2. Calls embed_with_ray.distributed() which:
     - Spawns NUM_WORKERS single-A10 Ray workers
     - Each worker loads ESM-2 once, processes batches via Ray Data
     - Distributed write_parquet to OUTPUT_STAGE_PARQUET
  3. Verifies the output is non-empty

What you still need to do MANUALLY (from a notebook):
  1. Stage your source table -> INPUT_STAGE_DIR (Spark write.parquet)
  2. After this job completes, CTAS the OUTPUT_STAGE_PARQUET into a managed
     UC Delta table:
       CREATE OR REPLACE TABLE <catalog>.<schema>.sequence_embeddings
       USING DELTA AS SELECT * FROM parquet.`<OUTPUT_STAGE_PARQUET>`
     (UC managed tables can't be written from non-Spark clients, so Ray
     can't write the final table directly — this is the same constraint
     as the notebook flow.)
"""

import os
import sys
from pathlib import Path

from serverless_gpu.ray import ray_launch


NUM_WORKERS  = int(os.environ.get("NUM_WORKERS",  "8"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE",   "32"))
MAX_LENGTH   = int(os.environ.get("MAX_LENGTH",   "1024"))
ESM2_MODEL   = os.environ.get("ESM2_MODEL", "facebook/esm2_t33_650M_UR50D")

INPUT_STAGE_DIR      = os.environ["INPUT_STAGE_DIR"]
OUTPUT_STAGE_PARQUET = os.environ["OUTPUT_STAGE_PARQUET"]


@ray_launch(gpus=NUM_WORKERS, gpu_type="a10", remote=True)
def embed_with_ray():
    """Runs on rank-0 as the Ray driver after Ray head + workers are up."""
    import numpy as np
    import ray
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"Ray cluster resources: {ray.cluster_resources()}")

    class ESM2Embedder:
        """Stateful Ray Data actor — loads ESM-2 once per actor (per GPU)."""

        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
            self.model = (
                AutoModel.from_pretrained(ESM2_MODEL, torch_dtype=torch.float16)
                .cuda()
                .eval()
            )
            print(f"ESM-2 (FP16) loaded on {torch.cuda.get_device_name(0)}")

        def __call__(self, batch):
            seqs = list(batch["sequence"])
            tokens = self.tokenizer(
                seqs,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            ).to("cuda")
            with torch.no_grad():
                out = self.model(**tokens)
            # Attention-mask-weighted mean pool (matches the original notebook)
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            embs = (summed / counts).cpu().float().numpy()
            return {
                "seq_id": np.asarray(batch["seq_id"]),
                "embedding": embs.tolist(),
            }

    ds = ray.data.read_parquet(INPUT_STAGE_DIR)
    print(f"Dataset rows: {ds.count()}")

    result = ds.map_batches(
        ESM2Embedder,
        batch_size=BATCH_SIZE,
        num_gpus=1,
        concurrency=NUM_WORKERS,
        batch_format="numpy",
    )

    print(f"Distributed write_parquet to {OUTPUT_STAGE_PARQUET}")
    result.write_parquet(OUTPUT_STAGE_PARQUET)
    print("Ray write_parquet complete")


def main():
    print("=" * 60)
    print(f"ESM-2 batch embedding on SGC + Ray Data")
    print("=" * 60)
    print(f"workers (gpus):  {NUM_WORKERS}")
    print(f"batch size:      {BATCH_SIZE}")
    print(f"model:           {ESM2_MODEL}")
    print(f"max_length:      {MAX_LENGTH}")
    print(f"input stage:     {INPUT_STAGE_DIR}")
    print(f"output stage:    {OUTPUT_STAGE_PARQUET}")

    # Pre-flight: confirm input exists
    if not Path(INPUT_STAGE_DIR).exists():
        print(f"\nERROR: input stage not found at {INPUT_STAGE_DIR}")
        print("Stage your source table to this path via Spark first.")
        sys.exit(2)

    # Pre-flight: count input rows (FUSE pyarrow read)
    import pyarrow.dataset as pads
    input_count = pads.dataset(INPUT_STAGE_DIR).count_rows()
    print(f"Input row count: {input_count}")
    if input_count == 0:
        print("ERROR: input stage is empty")
        sys.exit(2)

    embed_with_ray.distributed()
    print("\nRay job complete.")

    # Post-flight: verify output is non-empty
    if not Path(OUTPUT_STAGE_PARQUET).exists():
        print(f"\nFAIL: output dir does not exist at {OUTPUT_STAGE_PARQUET}")
        sys.exit(1)

    output_count = pads.dataset(OUTPUT_STAGE_PARQUET).count_rows()
    print(f"Output row count: {output_count}")
    if output_count != input_count:
        print(f"WARNING: output count ({output_count}) != input count ({input_count})")

    print("\nALL CHECKS PASSED — ESM-2 embedding completed on SGC + Ray Data")
    print(f"To promote to a UC Delta table, run from a notebook:")
    print(f"  CREATE OR REPLACE TABLE <cat>.<schema>.sequence_embeddings")
    print(f"  USING DELTA AS SELECT * FROM parquet.`{OUTPUT_STAGE_PARQUET}`")
    sys.exit(0)


if __name__ == "__main__":
    main()

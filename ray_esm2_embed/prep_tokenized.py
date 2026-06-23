"""
Pre-tokenize an existing parquet dataset for ESM-2 batch embedding.

Reads `seq_id` + `sequence` columns from INPUT, tokenizes with the ESM-2
tokenizer (padded to MAX_LENGTH), and writes parquet with:
  - seq_id          (passthrough)
  - input_ids       (list[int])
  - attention_mask  (list[int])

This eliminates per-batch tokenization in the Ray actor, the suspected
data-feed bottleneck behind the H100 utilization gap (73% in our 500K
run, ~57-60% in the 100K sweep).

Usage:
  pip install transformers
  python prep_tokenized.py \
    --input  /Volumes/<cat>/<schema>/sgc/ray_input_stage_200k \
    --output /Volumes/<cat>/<schema>/sgc/ray_input_stage_200k_tokenized \
    --n 100000 \
    --profile <PROFILE>
"""

import argparse
import os
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="UC Volume parquet input dir")
    p.add_argument("--output", required=True, help="UC Volume tokenized parquet output dir")
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--n", type=int, default=0, help="0 = all rows")
    p.add_argument("--profile", default=None)
    p.add_argument("--shard-rows", type=int, default=500)
    args = p.parse_args()

    import pyarrow as pa
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    print(f"Loading ESM-2 tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # UC Volume paths are only FUSE-mounted on Databricks. From a laptop we
    # need to pull the parquet down via the CLI first.
    profile_args = ["-p", args.profile] if args.profile else []
    input_tmp = tempfile.mkdtemp(prefix="ray_in_")
    print(f"Downloading {args.input} -> {input_tmp}")
    subprocess.check_call([
        "databricks", "fs", "cp", "--recursive",
        f"dbfs:{args.input}", input_tmp, "--overwrite", *profile_args,
    ])

    print(f"Reading {input_tmp}")
    ds = pads.dataset(input_tmp)
    table = ds.to_table(columns=["seq_id", "sequence"])
    if args.n and args.n < table.num_rows:
        table = table.slice(0, args.n)
    print(f"  {table.num_rows} rows to tokenize")

    # Batch tokenize to keep memory bounded
    seq_ids = table.column("seq_id").to_pylist()
    seqs    = table.column("sequence").to_pylist()
    all_input_ids: list[list[int]] = []
    all_masks:     list[list[int]] = []
    BATCH = 1024
    for i in range(0, len(seqs), BATCH):
        toks = tokenizer(
            seqs[i:i+BATCH],
            return_tensors=None,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",   # fixed shape — better for downstream + torch.compile
        )
        all_input_ids.extend(toks["input_ids"])
        all_masks.extend(toks["attention_mask"])
        if (i // BATCH) % 10 == 0:
            print(f"  ...tokenized {i+len(toks['input_ids'])} / {len(seqs)}")

    out_table = pa.table({
        "seq_id":         pa.array(seq_ids),
        "input_ids":      pa.array(all_input_ids),
        "attention_mask": pa.array(all_masks),
    })
    print(f"Tokenized: {out_table.num_rows} rows, schema={out_table.schema}")

    # Shard + upload
    with tempfile.TemporaryDirectory() as tmp:
        n_shards = (out_table.num_rows + args.shard_rows - 1) // args.shard_rows
        local_paths = []
        for i in range(n_shards):
            shard = out_table.slice(i * args.shard_rows, args.shard_rows)
            path = os.path.join(tmp, f"part-{i:05d}.parquet")
            pq.write_table(shard, path)
            local_paths.append(path)
        print(f"Wrote {len(local_paths)} parquet shard(s) locally")

        profile_args = ["-p", args.profile] if args.profile else []
        target = f"dbfs:{args.output}"
        subprocess.run(["databricks", "fs", "mkdir", target, *profile_args],
                       check=False, capture_output=True)
        for local in local_paths:
            basename = os.path.basename(local)
            print(f"  Uploading {basename}")
            subprocess.check_call([
                "databricks", "fs", "cp", local,
                f"{target}/{basename}", "--overwrite", *profile_args,
            ])
    print(f"Done. Tokenized data staged at {args.output}")


if __name__ == "__main__":
    main()

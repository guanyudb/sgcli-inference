"""
Stage an ESM-2 input dataset to a UC Volume.

Three data sources supported (pick one with --source):

  uniref50 (default, REAL public protein data):
    Downloads from `agemagician/uniref50` on HuggingFace. UniRef50 is
    UniProt's reference clusters at 50% identity — the canonical public
    protein database used to train ESM-2 itself. Streaming download, only
    the first N rows are pulled, no full-dataset cache needed.
    Requires: `pip install datasets`

  swissprot (REAL public protein data):
    Downloads UniProt SwissProt FASTA directly from EBI FTP. ~570K
    manually-curated, reviewed sequences. ~290 MB compressed.

  synthetic:
    Generates N random amino acid sequences. No network. Useful when
    HuggingFace/UniProt is unreachable from your network.

Output schema matches Srijit's `sequence_db`: columns `seq_id` (str) +
`sequence` (str). Sharded parquet uploaded to a UC Volume via the
Databricks CLI.

Usage (real data, ~1000 sequences):
  python prep_data.py \
    --output /Volumes/main/guanyu_chen/sgc/ray_input_stage \
    --n 1000 \
    --profile SGCCLI

Synthetic fallback:
  python prep_data.py --output /Volumes/... --source synthetic --n 1000
"""

import argparse
import gzip
import io
import os
import subprocess
import tempfile
import urllib.request

import numpy as np


# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------

def from_uniref50(n: int, max_len: int) -> tuple[list[str], list[str]]:
    """Stream N sequences from HuggingFace agemagician/uniref50 (UniRef50)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Missing 'datasets'. Install with: pip install datasets\n"
            "Or run with --source synthetic to skip the HF download."
        )

    print(f"Streaming first {n} sequences from huggingface://agemagician/uniref50 ...")
    ds = load_dataset("agemagician/uniref50", split="train", streaming=True)
    seq_ids, sequences = [], []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        seq = ex.get("text") or ex.get("sequence")
        if not seq:
            continue
        sequences.append(seq[:max_len])
        # UniRef50 has an "id" field like "UniRef50_P12345"
        seq_ids.append(str(ex.get("id", f"uniref50_{i:06d}")))
    print(f"  Got {len(sequences)} sequences "
          f"(lens: min={min(len(s) for s in sequences)}, "
          f"median={sorted(len(s) for s in sequences)[len(sequences)//2]}, "
          f"max={max(len(s) for s in sequences)})")
    return seq_ids, sequences


def from_swissprot(n: int, max_len: int) -> tuple[list[str], list[str]]:
    """Download UniProt SwissProt FASTA, take first N sequences."""
    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
    print(f"Downloading SwissProt FASTA from {url} ...")
    print("  (~290MB compressed — may take a few minutes)")

    seq_ids, sequences = [], []
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            stream = io.TextIOWrapper(gz, encoding="utf-8")
            current_id = None
            current_seq = []
            for line in stream:
                line = line.rstrip()
                if line.startswith(">"):
                    if current_id is not None and current_seq:
                        sequences.append("".join(current_seq)[:max_len])
                        seq_ids.append(current_id)
                        if len(sequences) >= n:
                            break
                    # >sp|P12345|GENE_HUMAN Description
                    parts = line[1:].split("|")
                    current_id = parts[1] if len(parts) >= 2 else f"sp_{len(sequences):06d}"
                    current_seq = []
                else:
                    current_seq.append(line)
            else:
                # Last record (if loop didn't break)
                if current_id is not None and current_seq and len(sequences) < n:
                    sequences.append("".join(current_seq)[:max_len])
                    seq_ids.append(current_id)
    print(f"  Got {len(sequences)} sequences from SwissProt")
    return seq_ids, sequences


def from_synthetic(n: int, min_len: int, max_len: int, mean_len: int,
                   seed: int) -> tuple[list[str], list[str]]:
    """Generate N random protein sequences with realistic AA frequencies."""
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    aa_freqs = np.array([
        0.082, 0.014, 0.054, 0.067, 0.039,
        0.071, 0.023, 0.060, 0.058, 0.097,
        0.024, 0.040, 0.047, 0.040, 0.057,
        0.066, 0.054, 0.069, 0.011, 0.029,
    ])
    aa_freqs = aa_freqs / aa_freqs.sum()

    print(f"Generating {n} synthetic protein sequences "
          f"(len {min_len}-{max_len}, mean {mean_len})...")
    rng = np.random.default_rng(seed)
    lengths = np.clip(
        rng.normal(loc=mean_len, scale=80, size=n).astype(int),
        min_len, max_len,
    )
    seq_ids = [f"synth_{i:06d}" for i in range(n)]
    sequences = [
        "".join(rng.choice(amino_acids, size=int(l), p=aa_freqs))
        for l in lengths
    ]
    return seq_ids, sequences


# --------------------------------------------------------------------------
# Parquet + upload
# --------------------------------------------------------------------------

def stage_to_uc_volume(seq_ids, sequences, output: str, shard_rows: int,
                       profile: str | None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"seq_id": seq_ids, "sequence": sequences})
    print(f"\nTable: {table.num_rows} rows, columns={table.column_names}")
    print(f"First record: id={seq_ids[0]}  len={len(sequences[0])}")
    print(f"  {sequences[0][:60]}{'...' if len(sequences[0]) > 60 else ''}")

    with tempfile.TemporaryDirectory() as tmp:
        n_shards = (len(seq_ids) + shard_rows - 1) // shard_rows
        local_paths = []
        for i in range(n_shards):
            start = i * shard_rows
            end   = min(start + shard_rows, len(seq_ids))
            shard = table.slice(start, end - start)
            path = os.path.join(tmp, f"part-{i:05d}.parquet")
            pq.write_table(shard, path)
            local_paths.append(path)
        print(f"Wrote {len(local_paths)} parquet shard(s) locally")

        profile_args = ["-p", profile] if profile else []
        target = f"dbfs:{output}"
        subprocess.run(
            ["databricks", "fs", "mkdir", target, *profile_args],
            check=False, capture_output=True,
        )
        for local in local_paths:
            basename = os.path.basename(local)
            print(f"  Uploading {basename} -> {output}/{basename}")
            subprocess.check_call([
                "databricks", "fs", "cp", local,
                f"{target}/{basename}", "--overwrite", *profile_args,
            ])
    print(f"\nDone. {len(seq_ids)} sequences staged at {output}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True,
                        help="UC Volume path for the staged parquet")
    parser.add_argument("--source", choices=["uniref50", "swissprot", "synthetic"],
                        default="uniref50",
                        help="Where to source protein sequences (default: uniref50)")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of sequences to stage (default: 1000)")
    parser.add_argument("--max-len", type=int, default=1024,
                        help="Truncate sequences to this length (default 1024 = ESM-2 max)")
    parser.add_argument("--min-len", type=int, default=50,
                        help="(synthetic only) min sequence length")
    parser.add_argument("--mean-len", type=int, default=200,
                        help="(synthetic only) mean sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="(synthetic only) RNG seed")
    parser.add_argument("--profile", default=None,
                        help="Databricks CLI profile for upload")
    parser.add_argument("--shard-rows", type=int, default=500,
                        help="Max rows per parquet shard (default 500)")
    args = parser.parse_args()

    if args.source == "uniref50":
        seq_ids, sequences = from_uniref50(args.n, args.max_len)
    elif args.source == "swissprot":
        seq_ids, sequences = from_swissprot(args.n, args.max_len)
    else:
        seq_ids, sequences = from_synthetic(
            args.n, args.min_len, args.max_len, args.mean_len, args.seed,
        )

    stage_to_uc_volume(seq_ids, sequences, args.output, args.shard_rows, args.profile)


if __name__ == "__main__":
    main()

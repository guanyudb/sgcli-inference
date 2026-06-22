#!/usr/bin/env python3
"""
run_inference.py — TEDDY-G embedding extraction (from TEDDY tutorial).

Steps:
  1. Preprocess    .h5ad → data/processed/
  2. Tokenize      data/processed/ → data/tokenized/
  3. Load model    Merck/TEDDY/teddy/models/teddy_g/{model_size}
  4. Embed         → outputs/embeddings.npy + metadata.csv
  5. UMAP          → outputs/umap.png

Usage:
    python sgc/container/run_inference.py \
        --input /path/to/cells.h5ad \
        --model-size 70M \
        --output outputs/embeddings \
        --max-cells 500

    # Inside container (model already mounted at /workspace/teddy_repo):
    python sgc/container/run_inference.py \
        --input /workspace/data/sample_data.h5ad \
        --teddy-path /workspace/teddy_repo \
        --output /workspace/outputs/embeddings

    # From a pretrain checkpoint (skips Merck/TEDDY weights, uses your checkpoint):
    python sgc/container/run_inference.py \
        --input /path/to/cells.h5ad \
        --checkpoint outputs/pretrain_minimal/checkpoint.pt
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

# ── Compatibility shim ────────────────────────────────────────────────────────
# Older versions of `datasets` (<2.14) don't accept `num_shards` in
# save_to_disk().  Monkey-patch it to silently drop unknown kwargs so
# the TEDDY tokenisation code works regardless of the installed version.
try:
    import datasets as _datasets
    _orig_save = _datasets.Dataset.save_to_disk
    if "num_shards" not in inspect.signature(_orig_save).parameters:
        def _save_to_disk_compat(self, dataset_path, *args, num_shards=None, **kwargs):
            return _orig_save(self, dataset_path, *args, **kwargs)
        _datasets.Dataset.save_to_disk = _save_to_disk_compat
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TEDDY-G embedding extraction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", required=True,
        help="Input .h5ad file (raw scRNA-seq counts).",
    )
    p.add_argument(
        "--model-size", default="70M", choices=["70M", "160M", "400M"],
        help="TEDDY-G model size variant to load.",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to pretrain_minimal.py checkpoint.pt (optional — overrides model weights).",
    )
    p.add_argument(
        "--output", default="outputs/embeddings",
        help="Output directory for embeddings.npy, metadata.csv, umap.png.",
    )
    p.add_argument(
        "--max-cells", type=int, default=500,
        help="Max cells to embed (reduce for CPU speed).",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for embedding loop.",
    )
    p.add_argument(
        "--seq-len", type=int, default=2048,
        help="Tokenizer max sequence length.",
    )
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "mps", "cuda"],
        help="Compute device.",
    )
    p.add_argument(
        "--teddy-path",
        default=os.environ.get("TEDDY_MODEL_PATH", "/Users/matusr/proj/llm/Merck/TEDDY"),
        help="Root of Merck/TEDDY repo (or set TEDDY_MODEL_PATH env var).",
    )
    p.add_argument(
        "--work-dir", default="/tmp/teddy_inference",
        help="Temp directory for preprocessed/tokenized data.",
    )
    p.add_argument(
        "--skip-umap", action="store_true",
        help="Skip UMAP plot (faster, useful in CI).",
    )
    p.add_argument(
        "--umap-n-neighbors", type=int, default=15,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    teddy_path = Path(args.teddy_path)
    if not teddy_path.exists():
        print(f"ERROR: TEDDY repo not found at {teddy_path}")
        print("  Set --teddy-path or TEDDY_MODEL_PATH env var.")
        sys.exit(1)

    # Add TEDDY repo to path
    sys.path.insert(0, str(teddy_path))

    import gc
    import json

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    model_path = teddy_path / "teddy" / "models" / "teddy_g" / args.model_size
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Available: {[p.name for p in model_path.parent.iterdir() if p.is_dir()]}"
        )

    work_dir = Path(args.work_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = work_dir / "processed"
    tokenized_dir = work_dir / "tokenized"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    device = torch.device(args.device)

    # ── 1. Preprocess ─────────────────────────────────────────────────────────
    print("==> Step 1/5: Preprocess")
    from teddy.data_processing.preprocessing.preprocess import preprocess

    # Build metadata path (same stem, .json suffix, next to input)
    metadata_path = input_path.with_suffix(".json")
    if not metadata_path.exists():
        # Create minimal metadata if not present
        meta = {"dataset": input_path.stem}
        metadata_path = processed_dir / f"{input_path.stem}_metadata.json"
        metadata_path.write_text(json.dumps(meta))

    preprocessing_config = {
        "min_gene_counts": None,
        "remove_assays": [],
        "max_mitochondrial_prop": None,
        "remove_cell_types": [],
        "hvg_method": None,
        "normalized_total": 10000,
        "median_dict": str(teddy_path / "teddy/data_processing/utils/medians/data/teddy_gene_medians.json"),
        "log1p": False,
        "compute_medians": False,
        "median_column": "index",
        "reference_id_only": False,
        "load_dir": str(input_path.parent),
        "save_dir": str(processed_dir),
    }
    preprocess(
        data_path=str(input_path),
        metadata_path=str(metadata_path),
        hyperparameters=preprocessing_config,
    )
    print(f"   Processed → {processed_dir}")

    # ── 2. Tokenize ───────────────────────────────────────────────────────────
    print("==> Step 2/5: Tokenize")
    from teddy.data_processing.tokenization.tokenization import tokenize

    processed_h5ad = processed_dir / f"{input_path.stem}.h5ad"
    processed_meta = processed_dir / f"{input_path.stem}_metadata.json"

    tokenizer_config = {
        "tokenizer_name_or_path": str(model_path),
        "gene_id_column": "index",
        "bio_annotations": True,
        "disease_mapping": str(
            teddy_path / "teddy/data_processing/utils/bio_annotations/data/mappings/all_filtered_disease_mapping.json"
        ),
        "tissue_mapping": str(
            teddy_path / "teddy/data_processing/utils/bio_annotations/data/mappings/all_filtered_tissue_mapping.json"
        ),
        "cell_mapping": str(
            teddy_path / "teddy/data_processing/utils/bio_annotations/data/mappings/all_filtered_cell_mapping.json"
        ),
        "sex_mapping": str(
            teddy_path / "teddy/data_processing/utils/bio_annotations/data/mappings/all_filtered_sex_mapping.json"
        ),
        "max_shard_samples": 500,
        "max_seq_len": args.seq_len,
        "pad_length": args.seq_len,
        "add_cls": False,
        "bins": 0,
        "continuous_rank": True,
        "truncation_method": "max",
        "add_disease_annotation": False,
        "include_zero_genes": False,
        "load_dir": str(processed_dir),
        "save_dir": str(tokenized_dir),
    }
    tokenize(
        data_path=str(processed_h5ad),
        metadata_path=str(processed_meta),
        tokenization_args=tokenizer_config,
    )
    print(f"   Tokenized → {tokenized_dir}")

    # ── 3. Load model ─────────────────────────────────────────────────────────
    print(f"==> Step 3/5: Load TeddyG {args.model_size}")
    from teddy.models.model_directory import get_architecture, model_dict

    architecture = get_architecture(str(model_path))
    config_cls = model_dict[architecture]["config_cls"]
    model_cls = model_dict[architecture]["model_cls"]

    config = config_cls.from_pretrained(str(model_path))
    model = model_cls.from_pretrained(str(model_path), config=config)
    model.return_all_embs = True

    # Optionally load pretrain_minimal.py checkpoint weights
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"   Loaded checkpoint weights from {args.checkpoint}")

    model = model.to(device)
    model.eval()
    print(f"   Model ready on {device}")

    # ── 4. Load tokenized data + collate ──────────────────────────────────────
    print("==> Step 4/5: Embed cells")
    from datasets import load_from_disk
    from teddy.tokenizer.gene_tokenizer import GeneTokenizer

    tokenizer = GeneTokenizer.from_pretrained(str(model_path))

    ds = load_from_disk(str(tokenized_dir / input_path.stem))

    if args.max_cells < len(ds):
        ds = ds.select(range(args.max_cells))
        print(f"   Using {args.max_cells} of {len(ds)} cells")

    max_seq_len = args.seq_len

    def collate_fn(batch):
        batch_size = len(batch)
        input_ids = torch.full(
            (batch_size, max_seq_len), tokenizer.pad_token_id, dtype=torch.long
        )
        for i, sample in enumerate(batch):
            seq = sample["gene_ids"]
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        return {"gene_ids": input_ids, "attention_mask": attention_mask}

    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn)

    all_embeddings = []
    with torch.no_grad():
        for batch_tensors in tqdm(loader, desc="Embedding"):
            gene_ids = batch_tensors["gene_ids"].to(device)
            attn_mask = batch_tensors["attention_mask"].to(device)
            outputs = model(gene_ids=gene_ids, attention_mask=attn_mask, return_outputs=True)
            emb = outputs["all_embs"].cpu()
            all_embeddings.append(emb)
            gc.collect()

    final_embeddings = torch.cat(all_embeddings, dim=0)
    n_cells, seq_len, hidden_dim = final_embeddings.shape
    print(f"   Embeddings shape: {final_embeddings.shape}")

    # Mean pool → [n_cells, hidden_dim]
    pooled = final_embeddings.mean(dim=1).numpy()

    # Save
    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, pooled)
    pd.DataFrame(pooled).to_csv(out_dir / "embeddings.csv", index=False)
    print(f"   Saved → {emb_path}  ({pooled.shape})")

    # ── 5. UMAP ───────────────────────────────────────────────────────────────
    if not args.skip_umap:
        print("==> Step 5/5: UMAP")
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import umap

        reducer = umap.UMAP(
            n_neighbors=args.umap_n_neighbors,
            random_state=42,
            metric="cosine",
        )
        coords = reducer.fit_transform(pooled)

        plt.figure(figsize=(7, 6))
        plt.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.7)
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.title(f"UMAP of Mean-Pooled Cell Embeddings ({args.model_size})")
        plt.tight_layout()
        umap_path = out_dir / "umap.png"
        plt.savefig(umap_path, dpi=150)
        print(f"   Saved → {umap_path}")
    else:
        print("==> Step 5/5: UMAP skipped (--skip-umap)")

    print(f"\n✅ Inference complete! All outputs in {out_dir}")
    print(f"   embeddings.npy  ({pooled.shape[0]} cells × {pooled.shape[1]} dims)")


if __name__ == "__main__":
    main()

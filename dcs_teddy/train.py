"""
Entry script for the SGC docker test.

1. Print GPU / distributed sanity info.
2. Locate run_inference.py inside the image and invoke it on the sample
   data that ships with Srijit's teddy-inference image.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch


def gpu_sanity_check():
    print(f"torch:              {torch.__version__}")
    print(f"CUDA available:     {torch.cuda.is_available()}")
    print(f"CUDA version:       {torch.version.cuda}")
    print(
        f"NCCL version:       "
        f"{torch.cuda.nccl.version() if torch.cuda.is_available() else 'n/a'}"
    )
    print(f"NUM_NODES:          {os.environ.get('NUM_NODES', 'unset')}")
    print(f"LOCAL_WORLD_SIZE:   {os.environ.get('LOCAL_WORLD_SIZE', 'unset')}")
    print(f"WORLD_SIZE:         {os.environ.get('WORLD_SIZE', 'unset')}")
    print(f"POD_RANK:           {os.environ.get('POD_RANK', 'unset')}")
    print(f"MASTER_ADDR:        {os.environ.get('MASTER_ADDR', 'unset')}")
    print(f"IS_HOST:            {os.environ.get('IS_HOST', 'unset')}")

    if not torch.cuda.is_available():
        print("CUDA not available — aborting before inference")
        sys.exit(1)

    print(f"Device count:       {torch.cuda.device_count()}")
    print(f"Device name:        {torch.cuda.get_device_name(0)}")
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = torch.matmul(a, b)
    print(f"Matmul OK, shape={c.shape}, sum={c.sum().item():.2f}")
    print("CUDA sanity check passed.\n")


def find_run_inference():
    """Search common in-container paths for run_inference.py."""
    candidates = [
        "/app/sgc/container/run_inference.py",
        "/workspace/sgc/container/run_inference.py",
        "/app/run_inference.py",
        "/workspace/run_inference.py",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def main():
    gpu_sanity_check()

    teddy_path = os.environ.get("TEDDY_MODEL_PATH", "/workspace/teddy_repo")
    sample_data = os.environ.get("SAMPLE_DATA", "/workspace/data/sample_data.h5ad")
    out_dir = os.environ.get("OUTPUT_DIR", "/workspace/outputs/embeddings")

    script = find_run_inference()
    if not script:
        print("run_inference.py not found in expected paths — listing /app and /workspace:")
        subprocess.run(["ls", "-la", "/app"], check=False)
        subprocess.run(["ls", "-la", "/workspace"], check=False)
        sys.exit(2)

    cmd = [
        sys.executable, script,
        "--input", sample_data,
        "--teddy-path", teddy_path,
        "--output", out_dir,
        "--device", "cuda",
        "--model-size", os.environ.get("MODEL_SIZE", "70M"),
        "--max-cells", os.environ.get("MAX_CELLS", "100"),
        "--skip-umap",
    ]
    print(">>> Invoking:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    print(f">>> run_inference.py exited with code {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quantize Google Gemma 4 E2B to an FP8 compressed-tensors checkpoint.

This uses LLM Compressor's data-free model_free_ptq entrypoint, so it does not
load the full Transformers model or require a calibration dataset.

Example:
    uv run python quantize_gemma4_fp8.py

The Gemma checkpoint is gated. Accept its license and authenticate first with
`hf auth login` or set the HF_TOKEN environment variable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

import torch
from llmcompressor import model_free_ptq


DEFAULT_MODEL = "google/gemma-4-E2B"
DEFAULT_IGNORE = (
    "re:.*vision.*",
    "lm_head",
    "re:.*embed_tokens.*",
)
SUPPORTED_SCHEMES = ("FP8_DYNAMIC", "FP8_BLOCK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize Gemma 4 E2B on CUDA and save a vLLM-compatible "
            "compressed-tensors FP8 checkpoint."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model ID or local safetensors directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: ./<model-name>-<scheme>).",
    )
    parser.add_argument(
        "--scheme",
        choices=SUPPORTED_SCHEMES,
        default="FP8_DYNAMIC",
        help=(
            "FP8_DYNAMIC is recommended for H100/Hopper; FP8_BLOCK is "
            "primarily optimized for Blackwell (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used for quantization (default: %(default)s).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of checkpoint shards processed concurrently (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and CUDA, then print the operation without quantizing.",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def validate_cuda(device: str) -> None:
    if not device.startswith("cuda"):
        fail(f"--device must be a CUDA device, not {device!r}")

    if not torch.cuda.is_available():
        fail("CUDA is not available to PyTorch")

    try:
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        properties = torch.cuda.get_device_properties(index)
    except (AssertionError, RuntimeError, ValueError) as exc:
        fail(f"cannot use {device!r}: {exc}")

    capability = (properties.major, properties.minor)
    if capability < (8, 9):
        fail(
            f"{properties.name} has compute capability {properties.major}.{properties.minor}; "
            "hardware-accelerated W8A8 FP8 requires compute capability 8.9 or newer"
        )

    print(
        f"CUDA device: {properties.name} "
        f"(compute capability {properties.major}.{properties.minor})"
    )


def default_output_dir(model: str, scheme: str) -> Path:
    model_name = model.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
    return Path(f"{model_name}-{scheme.replace('_', '-')}")


def main() -> None:
    args = parse_args()

    if not args.model.strip():
        fail("--model cannot be empty")
    if args.max_workers < 1:
        fail("--max-workers must be at least 1")

    output_dir = args.output_dir or default_output_dir(args.model, args.scheme)
    if output_dir.exists():
        if not output_dir.is_dir():
            fail(f"output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            fail(f"output directory is not empty: {output_dir}")

    validate_cuda(args.device)

    print(f"Source:  {args.model}")
    print(f"Output:  {output_dir.resolve()}")
    print(f"Scheme:  {args.scheme}")
    print(f"Ignored: {', '.join(DEFAULT_IGNORE)}")

    if args.dry_run:
        print("Dry run complete; no files were written.")
        return

    model_free_ptq(
        model_stub=args.model,
        save_directory=str(output_dir),
        scheme=args.scheme,
        ignore=list(DEFAULT_IGNORE),
        max_workers=args.max_workers,
        device=args.device,
    )

    print(f"Quantized checkpoint saved to {output_dir.resolve()}")
    print(f"Serve it with: vllm serve {output_dir.resolve()}")


if __name__ == "__main__":
    main()

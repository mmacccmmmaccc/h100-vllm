#!/usr/bin/env python3
"""Quantize Typhoon 2 Audio 8B to weight-only NVIDIA FP4.

The source checkpoint is stored as PyTorch ``.bin`` shards, so this script uses
LLM Compressor's model-based ``oneshot`` entrypoint rather than model_free_ptq.
Accelerated NVFP4 inference is intended for NVIDIA Blackwell GPUs (compute
capability 10.0+), although the checkpoint can be produced on an H100.

Example:
    uv run python quantize_typhoon2_audio_nvfp4.py

The model contains custom Hugging Face code. Review the repository before
running this script; loading it requires ``trust_remote_code=True``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

import torch


DEFAULT_MODEL = "typhoon-ai/llama3.1-typhoon2-audio-8b-instruct"
DEFAULT_OUTPUT_DIR = Path("llama3.1-typhoon2-audio-8b-instruct-NVFP4A16")
SCHEME = "NVFP4A16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize Typhoon 2 Audio 8B weights to NVFP4 and save a "
            "compressed-tensors checkpoint."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model ID or local model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used for quantization (default: %(default)s).",
    )
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face model revision or commit hash.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and GPU compatibility without loading the model.",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def validate_device(device: str) -> None:
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
    print(
        f"CUDA device: {properties.name} "
        f"(compute capability {properties.major}.{properties.minor})"
    )
    if capability < (9, 0):
        fail(
            f"{properties.name} has compute capability "
            f"{properties.major}.{properties.minor}; use an H100 or newer GPU "
            "to run this quantization job"
        )
    if capability < (10, 0):
        print(
            "Warning: this GPU can produce the checkpoint, but accelerated "
            "NVFP4 inference requires a Blackwell GPU (compute capability 10.0+).",
            file=sys.stderr,
        )


def validate_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        fail(f"output path exists and is not a directory: {output_dir}")
    if any(output_dir.iterdir()):
        fail(f"output directory is not empty: {output_dir}")


def language_model_linear_names(model: torch.nn.Module) -> tuple[str, list[str]]:
    """Return the Llama backbone name and its quantizable Linear module names."""
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "LlamaForCausalLM"
    ]
    if len(candidates) != 1:
        found = ", ".join(name or "<root>" for name, _ in candidates) or "none"
        fail(
            "expected exactly one LlamaForCausalLM language backbone; "
            f"found {len(candidates)} ({found})"
        )

    backbone_name, backbone = candidates[0]
    prefix = f"{backbone_name}." if backbone_name else ""
    targets = [
        f"{prefix}{name}"
        for name, module in backbone.named_modules()
        if name and isinstance(module, torch.nn.Linear) and name != "lm_head"
    ]
    if not targets:
        fail(f"no Linear layers found under language backbone {backbone_name!r}")
    return backbone_name or "<root>", targets


def install_typhoon_transformers_compat() -> None:
    """Restore generation.utils exports expected by Typhoon's remote code."""
    from transformers import cache_utils, utils
    from transformers.generation import utils as generation_utils
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    moved_exports = {
        "is_deepspeed_zero3_enabled": is_deepspeed_zero3_enabled,
        "is_torchdynamo_compiling": utils.is_torchdynamo_compiling,
        "is_hqq_available": utils.is_hqq_available,
        "QuantizedCacheConfig": cache_utils.QuantizedCacheConfig,
        "DynamicCache": cache_utils.DynamicCache,
        "EncoderDecoderCache": cache_utils.EncoderDecoderCache,
    }
    if hasattr(utils, "is_quanto_available"):
        moved_exports["is_quanto_available"] = utils.is_quanto_available
    elif hasattr(utils, "is_optimum_quanto_available"):
        moved_exports["is_quanto_available"] = utils.is_optimum_quanto_available

    for name, value in moved_exports.items():
        if not hasattr(generation_utils, name):
            setattr(generation_utils, name, value)


def main() -> None:
    args = parse_args()

    if not args.model.strip():
        fail("--model cannot be empty")
    validate_device(args.device)
    validate_output_dir(args.output_dir)

    print(f"Source:  {args.model}")
    print(f"Output:  {args.output_dir.resolve()}")
    print(f"Scheme:  {SCHEME} (weight-only)")
    print("Scope:   Llama language backbone only; audio stack remains dense")

    if args.dry_run:
        print("Dry run complete; no model was downloaded and no files were written.")
        return

    # Keep these imports here so --dry-run can validate a host before the large
    # optional model dependencies (including fairseq) are installed.
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModel

    install_typhoon_transformers_compat()

    load_kwargs: dict[str, object] = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
        "device_map": {"": args.device},
        "low_cpu_mem_usage": True,
    }
    if args.revision:
        load_kwargs["revision"] = args.revision

    # Typhoon2AudioConfig registers its custom implementation with AutoModel,
    # not AutoModelForCausalLM (despite containing a Llama causal LM).
    model = AutoModel.from_pretrained(args.model, **load_kwargs)
    backbone_name, targets = language_model_linear_names(model)
    print(f"Language backbone: {backbone_name}")
    print(f"Quantizing {len(targets)} language-model Linear layers")
    print("Leaving the audio encoders, Q-Former, projector, and speech decoder dense")
    recipe = QuantizationModifier(
        targets=targets,
        scheme=SCHEME,
        ignore=["lm_head"],
    )
    oneshot(model=model, recipe=recipe)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    print(f"Quantized checkpoint saved to {args.output_dir.resolve()}")
    print(
        "Note: this model uses a custom audio architecture; verify that your "
        "inference runtime supports it before deployment."
    )


if __name__ == "__main__":
    main()

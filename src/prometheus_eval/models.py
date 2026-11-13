"""Pinned acquisition and bounded-memory conversion of the original v1 weights."""

import gc
import json
import shutil
from pathlib import Path

from .io import environment, file_hash, write_json

MODEL_REPO = "prometheus-eval/prometheus-7b-v1.0"
MODEL_REVISION = "e7fc9f231ee722e81ce594536dfd0796881a34e2"
DATA_REPO = "prometheus-eval/Feedback-Collection"
DATA_REVISION = "22a339f961c40b07261bea44ec10849c7440b75f"


def fetch_data(output):
    from huggingface_hub import hf_hub_download

    source = hf_hub_download(
        DATA_REPO, "new_feedback_collection.json", repo_type="dataset", revision=DATA_REVISION
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    write_json(
        output.with_suffix(".source.json"),
        {
            "repository": DATA_REPO,
            "revision": DATA_REVISION,
            "sha256": file_hash(output),
            "environment": environment(),
        },
    )
    return str(output)


def convert_model(output, source=None, bits=4, group_size=64):
    """Quantize one tensor at a time, without materializing the 27GB FP32 model.

    mmap + weights_only avoids eager checkpoint deserialization. Conversion runs
    on CPU; GPU allocations are reserved for actual training and inference.
    Each completed shard is at most roughly 256 MiB plus one tensor.
    """
    import mlx.core as mx
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download

    output = Path(output)
    if (output / "conversion.json").exists():
        manifest = json.loads((output / "conversion.json").read_text())
        if manifest["bits"] != bits or manifest["group_size"] != group_size:
            raise ValueError(
                "Existing model has different quantization; use another output directory"
            )
        if source is not None:
            requested = {
                p.name: file_hash(p) for p in sorted(Path(source).glob("pytorch_model*.bin"))
            }
            if requested != manifest["source_sha256"]:
                raise ValueError(
                    "Existing model belongs to a different source; use another output directory"
                )
        for name, sha in manifest["output_sha256"].items():
            if file_hash(output / name) != sha:
                raise ValueError(f"Model integrity check failed: {name}")
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Incomplete/nonempty output {output}; use a new directory")
    if source is None:
        source = snapshot_download(
            MODEL_REPO,
            revision=MODEL_REVISION,
            allow_patterns=["*.json", "*.model", "*.bin"],
            max_workers=3,
        )
    source = Path(source)
    config = json.loads((source / "config.json").read_text())
    if config["model_type"] != "llama" or config.get("quantization_config"):
        raise ValueError("Streaming converter accepts unquantized HF Llama checkpoints only")
    shards = sorted(source.glob("pytorch_model*.bin"))
    if not shards:
        raise ValueError("No PyTorch .bin shards found")
    staging = output.with_name(output.name + ".converting")
    staging.mkdir(parents=True, exist_ok=True)
    # Restart overwrites our own numbered staging shards only; no finished output is touched.
    if any(staging.glob("*.safetensors")):
        raise ValueError(f"Interrupted conversion at {staging}; use a fresh output directory")
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    torch.set_num_threads(2)
    bucket, weight_map, total_size, parts = {}, {}, 0, []
    source_hashes = {}

    def flush():
        nonlocal bucket
        if not bucket:
            return
        name = f"model-{len(parts) + 1:05d}.safetensors"
        mx.save_safetensors(str(staging / name), bucket, metadata={"format": "mlx"})
        weight_map.update({key: name for key in bucket})
        parts.append(name)
        bucket = {}
        gc.collect()

    try:
        for shard in shards:
            print(f"Converting {shard.name} (memory mapped)", flush=True)
            source_hashes[shard.name] = file_hash(shard)
            weights = torch.load(shard, map_location="cpu", mmap=True, weights_only=True)
            for name, tensor in weights.items():
                if "rotary_emb.inv_freq" in name:
                    continue
                arr = mx.array(tensor.to(torch.float16).numpy().copy().astype(np.float16))
                if arr.ndim == 2 and arr.shape[-1] % group_size == 0:
                    q, scales, biases = mx.quantize(arr, group_size=group_size, bits=bits)
                    mx.eval(q, scales, biases)
                    bucket[name] = q
                    bucket[name.removesuffix("weight") + "scales"] = scales
                    bucket[name.removesuffix("weight") + "biases"] = biases
                else:
                    mx.eval(arr)
                    bucket[name] = arr
                total_size += sum(
                    x.nbytes
                    for k, x in bucket.items()
                    if k == name
                    or k
                    in (
                        name.removesuffix("weight") + "scales",
                        name.removesuffix("weight") + "biases",
                    )
                )
                del arr
                if sum(x.nbytes for x in bucket.values()) >= 256 * 1024**2:
                    flush()
            del weights, tensor
            gc.collect()
        flush()
        config["quantization"] = {"group_size": group_size, "bits": bits, "mode": "affine"}
        config["torch_dtype"] = "float16"
        write_json(staging / "config.json", config)
        for name in (
            "tokenizer.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "added_tokens.json",
        ):
            if (source / name).exists():
                shutil.copyfile(source / name, staging / name)
        write_json(
            staging / "model.safetensors.index.json",
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        )
        downloaded = source / "download_source.json"
        origin = json.loads(downloaded.read_text()) if downloaded.exists() else {}
        manifest = {
            "source_repository": origin.get(
                "repo", MODEL_REPO if source.name == MODEL_REVISION else str(source)
            ),
            "source_revision": origin.get("revision", source.name),
            "source_sha256": source_hashes,
            "bits": bits,
            "group_size": group_size,
            "method": "source floating-point weights to FP16 to MLX affine",
            "environment": environment(),
            "output_sha256": {
                p.name: file_hash(p) for p in sorted(staging.iterdir()) if p.is_file()
            },
        }
        write_json(staging / "conversion.json", manifest)
        staging.rename(output)
        return manifest
    finally:
        mx.set_default_device(previous_device)

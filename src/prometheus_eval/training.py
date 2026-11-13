"""Completion-only QLoRA training with grouped holdouts and persisted measurements."""

import json
import math
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from .backends import model_identity
from .io import append_jsonl, environment, file_hash, read_records, write_json
from .prompts import SYSTEM, ablate, llama2_chat, parse_grade


@dataclass
class TrainingConfig:
    model: str
    data: str
    output: str
    variant: str = "full"
    steps: int = 8
    learning_rate: float = 1e-5
    num_layers: int = 4
    rank: int = 8
    max_seq_length: int = 2048
    seed: int = 42
    val_examples: int = 2
    memory_gb: float = 10
    chat_format: str = "llama2"


def tokenize_completion(tokenizer, prompt, completion, chat_format="llama2"):
    if chat_format == "llama2":
        rendered = llama2_chat(prompt)
    else:
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    # Encode at the actual prefix boundary, keeping the target score and EOS intact.
    target = tokenizer.encode(" " + completion.strip(), add_special_tokens=False)
    if not target:
        raise ValueError("Empty completion")
    return prefix + target + [tokenizer.eos_token_id], len(prefix)


def completion_mask(lengths, width):
    """Target positions are 1..L-1; EOS counts, padding and prompt do not."""
    import mlx.core as mx

    positions = mx.arange(1, width + 1)
    return (positions >= lengths[:, 0:1]) & (positions < lengths[:, 1:2])


def completion_loss(model, batch, lengths):
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(batch[:, :-1])
    mask = completion_mask(lengths, batch.shape[1] - 1)
    tokens = mask.sum()
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), batch[:, 1:])
    return (losses * mask).sum() / tokens, tokens


def single_example_batches(
    dataset, batch_size, max_seq_length, loop=False, seed=42, comm_group=None
):
    """Keep ablation example order identical, independent of transformed token length."""
    import mlx.core as mx
    import numpy as np

    if batch_size != 1 or (comm_group is not None and comm_group.size() != 1):
        raise ValueError("The native Mac training iterator requires one worker and batch size one")
    rng = np.random.default_rng(seed)
    while True:
        indices = rng.permutation(len(dataset)) if loop else range(len(dataset))
        for index in indices:
            tokens, offset = dataset[index]
            if len(tokens) > max_seq_length or not 0 < offset < len(tokens):
                raise ValueError("Unfiltered overlength or empty-target training example")
            yield (
                mx.array([tokens], dtype=mx.int32),
                mx.array([[offset, len(tokens)]], dtype=mx.int32),
            )
        if not loop:
            break


def train_adapter(config):
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    if min(config.steps, config.num_layers, config.rank, config.val_examples) < 1:
        raise ValueError("Training steps, layers, rank and validation examples must be positive")
    if config.learning_rate <= 0 or config.max_seq_length < 8:
        raise ValueError("Invalid learning rate or sequence length")
    output = Path(config.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Training output {output} is nonempty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.seed)
    mx.random.seed(config.seed)
    mx.set_cache_limit(128 * 1024**2)
    mx.set_memory_limit(int(config.memory_gb * 1024**3))
    model, tokenizer = load(config.model, tokenizer_config={"trust_remote_code": False})
    if config.num_layers > len(model.layers):
        raise ValueError("Requested LoRA layers exceed model layers")
    prepared, selection = {}, {}
    for split in ("train", "valid", "test"):
        data, accepted, rejected = [], [], []
        for row in read_records(Path(config.data) / f"{split}.jsonl"):
            full, _ = tokenize_completion(
                tokenizer, row["instruction"], row["output"], config.chat_format
            )
            if len(full) > config.max_seq_length:
                rejected.append({"source_index": row["source_index"], "tokens": len(full)})
                continue
            completion = (
                row["output"]
                if config.variant != "score_only"
                else (f"[RESULT] {parse_grade(row['output']).score}")
            )
            tokens, offset = tokenize_completion(
                tokenizer,
                ablate(row["instruction"], config.variant),
                completion,
                config.chat_format,
            )
            if len(tokens) > config.max_seq_length or offset >= len(tokens) - 1:
                raise ValueError("Ablation produced an invalid training target")
            data.append((tokens, offset))
            accepted.append(row["source_index"])
        if not data:
            raise ValueError(
                f"No {split} examples fit; increase --max-seq-length. No targets were truncated."
            )
        prepared[split] = data
        selection[split] = {
            "accepted_source_indices": accepted,
            "overlength_exclusions": rejected,
            "source_sha256": file_hash(Path(config.data) / f"{split}.jsonl"),
        }
    model.freeze()
    lora = {
        "rank": config.rank,
        "scale": 16.0,
        "dropout": 0.0,
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
    }
    linear_to_lora_layers(model, config.num_layers, lora)
    adapter_config = {
        "model": config.model,
        "num_layers": config.num_layers,
        "fine_tune_type": "lora",
        "lora_parameters": lora,
    }
    write_json(output / "adapter_config.json", adapter_config)
    manifest = {
        "config": asdict(config),
        "environment": environment(),
        "base_model": model_identity(config.model),
        "selection": selection,
        "trainable_parameters": sum(x.size for _, x in tree_flatten(model.trainable_parameters())),
        "objective": "causal cross entropy on feedback + [RESULT] + score + EOS only",
        "scope": "QLoRA continuation; not full-parameter training from Llama-2-Chat",
    }
    order_rng = np.random.default_rng(config.seed)
    training_order = []
    while len(training_order) < config.steps:
        training_order.extend(
            order_rng.permutation(selection["train"]["accepted_source_indices"]).tolist()
        )
    manifest["training_order_source_indices"] = training_order[: config.steps]
    write_json(output / "manifest.json", manifest)

    class Recorder(TrainingCallback):
        def on_train_loss_report(self, info):
            if not math.isfinite(info["train_loss"]):
                raise FloatingPointError("Non-finite training loss")
            append_jsonl(output / "losses.jsonl", {"kind": "train", **info})

        def on_val_loss_report(self, info):
            if not math.isfinite(info["val_loss"]):
                raise FloatingPointError("Non-finite validation loss")
            append_jsonl(output / "losses.jsonl", {"kind": "validation", **info})

    start = time.perf_counter()
    train(
        model,
        optim.AdamW(learning_rate=config.learning_rate, weight_decay=0.0),
        prepared["train"],
        prepared["valid"][: config.val_examples],
        args=TrainingArgs(
            batch_size=1,
            iters=config.steps,
            val_batches=config.val_examples,
            steps_per_report=1,
            steps_per_eval=config.steps + 1,
            steps_per_save=config.steps,
            max_seq_length=config.max_seq_length,
            adapter_file=str(output / "adapters.safetensors"),
            grad_checkpoint=True,
            clear_cache_threshold=128 * 1024**2,
        ),
        loss=completion_loss,
        iterate_batches=partial(single_example_batches, seed=config.seed),
        training_callback=Recorder(),
    )
    # Evaluate after the last optimizer update, rather than reporting the preceding validation.
    model.eval()
    val = evaluate(
        model,
        prepared["valid"][: config.val_examples],
        1,
        config.val_examples,
        max_seq_length=config.max_seq_length,
        loss=completion_loss,
        iterate_batches=partial(single_example_batches, seed=config.seed),
    )
    test = evaluate(
        model,
        prepared["test"][: config.val_examples],
        1,
        config.val_examples,
        max_seq_length=config.max_seq_length,
        loss=completion_loss,
        iterate_batches=partial(single_example_batches, seed=config.seed),
    )
    if not math.isfinite(val) or not math.isfinite(test):
        raise FloatingPointError("Non-finite final holdout loss")
    summary = {
        "steps": config.steps,
        "final_validation_loss": val,
        "test_loss": test,
        "holdout_examples_evaluated_per_split": config.val_examples,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "adapter_sha256": file_hash(output / "adapters.safetensors"),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary

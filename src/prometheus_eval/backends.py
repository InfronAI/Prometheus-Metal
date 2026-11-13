"""Local MLX / PyTorch and optional TGI adapters with a common generation contract."""

import time
from dataclasses import dataclass
from pathlib import Path

from .io import file_hash
from .prompts import SYSTEM, llama2_chat


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.03
    max_input_tokens: int = 3072
    seed: int = 42

    def __post_init__(self):
        if self.max_tokens < 1 or self.max_input_tokens < 1:
            raise ValueError("Token limits must be positive")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.repetition_penalty <= 0:
            raise ValueError("Invalid sampling parameters")


class ContextOverflow(ValueError):
    pass


def model_identity(model, adapter=None):
    path = Path(model)
    identity = {"model": str(path.resolve()) if path.exists() else str(model)}
    if (path / "conversion.json").exists():
        identity["conversion_sha256"] = file_hash(path / "conversion.json")
        import json

        manifest = json.loads((path / "conversion.json").read_text())
        for name, expected in manifest["output_sha256"].items():
            if file_hash(path / name) != expected:
                raise ValueError(f"Converted model integrity check failed: {name}")
    elif path.exists():
        identity["files"] = {p.name: file_hash(p) for p in sorted(path.glob("*.json"))}
        # Hash local weights once per backend creation; avoid silently resuming another model.
        identity["weights"] = {p.name: file_hash(p) for p in sorted(path.glob("*.safetensors"))}
    if adapter:
        identity["adapter_sha256"] = file_hash(Path(adapter) / "adapters.safetensors")
        identity["adapter_config_sha256"] = file_hash(Path(adapter) / "adapter_config.json")
    return identity


class MLXBackend:
    def __init__(self, model, adapter=None, chat_format="llama2", memory_gb=10):
        import mlx.core as mx
        from mlx_lm import load

        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal is unavailable. Use --backend torch --device cpu.")
        self.identity = model_identity(model, adapter)
        self.identity.update(backend="mlx", chat_format=chat_format)
        self.chat_format = chat_format
        mx.set_cache_limit(256 * 1024**2)
        mx.set_memory_limit(int(memory_gb * 1024**3))
        self.model, self.tokenizer = load(
            str(model), adapter_path=adapter, tokenizer_config={"trust_remote_code": False}
        )
        self.model.eval()
        cfg = getattr(self.model, "args", None)
        self.context = getattr(cfg, "max_position_embeddings", None) or 4096

    def prompt(self, text):
        if self.chat_format == "llama2":
            return llama2_chat(text)
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt, config):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        tokens = self.tokenizer.encode(self.prompt(prompt), add_special_tokens=False)
        if len(tokens) > config.max_input_tokens or len(tokens) + config.max_tokens > self.context:
            raise ContextOverflow(
                f"{len(tokens)} input tokens + {config.max_tokens} output; "
                f"limits input={config.max_input_tokens}, context={self.context}"
            )
        mx.random.seed(config.seed)
        start = time.perf_counter()
        text, last = "", None
        for part in stream_generate(
            self.model,
            self.tokenizer,
            tokens,
            max_tokens=config.max_tokens,
            sampler=make_sampler(temp=config.temperature, top_p=config.top_p),
            logits_processors=make_logits_processors(repetition_penalty=config.repetition_penalty),
            prefill_step_size=256,
        ):
            text += part.text
            last = part
        elapsed = time.perf_counter() - start
        result = {
            "text": text,
            "elapsed_seconds": elapsed,
            "prompt_tokens": len(tokens),
            "generation_tokens": last.generation_tokens if last else 0,
            "generation_tps": last.generation_tps if last else 0,
            "prompt_tps": last.prompt_tps if last else 0,
            "peak_memory_gb": last.peak_memory if last else 0,
            "finish_reason": getattr(last, "finish_reason", None),
            "seed": config.seed,
        }
        mx.clear_cache()
        return result


class TorchBackend:
    def __init__(self, model, adapter=None, device="auto", chat_format="llama2", **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        self.device, self.chat_format = device, chat_format
        self.identity = model_identity(model, adapter)
        self.identity.update(backend="torch", device=device, chat_format=chat_format)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=dtype, trust_remote_code=False
        ).to(device)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, prompt, config):
        import torch

        prompt = (
            llama2_chat(prompt)
            if self.chat_format == "llama2"
            else self.tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.device
        )
        n = inputs.input_ids.shape[1]
        if (
            n > config.max_input_tokens
            or n + config.max_tokens > self.model.config.max_position_embeddings
        ):
            raise ContextOverflow(f"Prompt too long: {n} tokens; no silent truncation")
        torch.manual_seed(config.seed)
        kwargs = {
            "max_new_tokens": config.max_tokens,
            "do_sample": config.temperature > 0,
            "repetition_penalty": config.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if config.temperature > 0:
            kwargs.update(temperature=config.temperature, top_p=config.top_p)
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.model.generate(**inputs, **kwargs)
        output = result[0, n:]
        elapsed = time.perf_counter() - start
        return {
            "text": self.tokenizer.decode(output, skip_special_tokens=True),
            "prompt_tokens": n,
            "generation_tokens": len(output),
            "elapsed_seconds": elapsed,
            "generation_tps": len(output) / elapsed,
            "seed": config.seed,
        }


class TGIBackend:
    def __init__(self, server, tokenizer_model, timeout=120):
        from transformers import AutoTokenizer

        self.server, self.timeout = server, timeout
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=False)
        self.identity = {"backend": "tgi", "server": server, "tokenizer": tokenizer_model}

    def generate(self, prompt, config):
        import requests

        prompt = llama2_chat(prompt)
        n = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if n > config.max_input_tokens:
            raise ContextOverflow(f"Prompt too long: {n}")
        params = {
            "max_new_tokens": config.max_tokens,
            "do_sample": config.temperature > 0,
            "repetition_penalty": config.repetition_penalty,
            "seed": config.seed,
            "return_full_text": False,
        }
        if config.temperature > 0:
            params.update(temperature=config.temperature, top_p=config.top_p)
        start = time.perf_counter()
        response = requests.post(
            self.server, json={"inputs": prompt, "parameters": params}, timeout=self.timeout
        )
        response.raise_for_status()
        value = response.json()
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if not isinstance(value, dict) or not isinstance(value.get("generated_text"), str):
            raise ValueError("Invalid TGI response: expected generated_text")
        return {
            "text": value["generated_text"],
            "elapsed_seconds": time.perf_counter() - start,
            "prompt_tokens": n,
            "seed": config.seed,
        }


def backend_from_config(config):
    options = dict(config)
    backend = options.pop("backend", "mlx")
    if backend == "mlx":
        return MLXBackend(**options)
    if backend == "torch":
        return TorchBackend(**options)
    if backend == "tgi":
        return TGIBackend(**options)
    raise ValueError(f"Unknown backend {backend}")

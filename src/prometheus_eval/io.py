"""Small, auditable artifact and data utilities. No model imports."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_records(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if text.startswith("version https://git-lfs.github.com/spec"):
        raise ValueError(f"{path} is a Git LFS pointer, not data. Run git lfs pull or fetch-data.")
    if not text.strip():
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    value.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n}: invalid JSON: {e.msg}") from e
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: expected a JSON array of objects or JSONL objects")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(canonical(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(canonical(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    versions = {}
    for name in ("mlx", "mlx-lm", "torch", "transformers", "numpy", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "timestamp_utc": utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "packages": versions,
        "upstream_commit": commit,
    }

#!/usr/bin/env python3
"""Bounded, resumable HTTP ranges for a pinned checkpoint; verify every LFS SHA.

Useful when the Hub's adaptive Xet downloader reduces concurrency on a slow link.
No authentication, arbitrary code execution, or unpinned model revisions.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests


def download(repo, revision, output, workers=16, chunk_mb=16):
    if not 1 <= workers <= 32 or not 1 <= chunk_mb <= 128:
        raise ValueError("Use 1..32 workers and 1..128 MiB chunks")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://huggingface.co/api/models/{repo}/tree/{revision}", timeout=60)
    response.raise_for_status()
    files = [f for f in response.json() if f["path"].endswith((".bin", ".json", ".model"))]
    (output / "download_source.json").write_text(
        json.dumps({"repo": repo, "revision": revision, "files": files}, indent=2)
    )
    chunk_size = chunk_mb * 1024**2
    jobs, states, locks, descriptors = [], {}, {}, {}
    for item in files:
        name, size = item["path"], item["size"]
        target = output / name
        if target.exists() and target.stat().st_size == size:
            continue
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        if size < chunk_size:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            if len(r.content) != size:
                raise ValueError(f"Size mismatch: {name}")
            target.write_bytes(r.content)
            continue
        state_path = output / (name + ".progress.json")
        states[name] = json.loads(state_path.read_text()) if state_path.exists() else []
        locks[name] = threading.Lock()
        partial = output / (name + ".partial")
        if states[name] and not partial.exists():
            raise ValueError("Progress exists without partial data")
        fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, size)
        descriptors[name] = fd
        for start in range(0, size, chunk_size):
            end = min(start + chunk_size, size) - 1
            if start not in states[name]:
                jobs.append((name, url, start, end, size))
    total = len(jobs)
    finished = 0
    counter_lock = threading.Lock()
    began = time.perf_counter()

    def fetch(job):
        nonlocal finished
        name, url, start, end, size = job
        for attempt in range(4):
            try:
                # Cache-busting range in URL prevents proxies from reusing a different byte range.
                with requests.get(
                    url,
                    params={"range_start": start},
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(30, 120),
                    stream=True,
                ) as r:
                    r.raise_for_status()
                    if (
                        r.status_code != 206
                        or r.headers.get("Content-Range") != f"bytes {start}-{end}/{size}"
                    ):
                        raise ValueError("Server did not honor exact byte range")
                    offset = start
                    for block in r.iter_content(1024**2):
                        if offset + len(block) > end + 1:
                            raise ValueError("Oversized range")
                        os.pwrite(descriptors[name], block, offset)
                        offset += len(block)
                    if offset != end + 1:
                        raise ValueError("Incomplete range")
                with locks[name]:
                    os.fsync(descriptors[name])
                    states[name].append(start)
                    state_path = output / (name + ".progress.json")
                    tmp = state_path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(sorted(states[name])))
                    tmp.replace(state_path)
                with counter_lock:
                    finished += 1
                    if finished % 8 == 0 or finished == total:
                        print(
                            f"{finished}/{total} chunks; elapsed {time.perf_counter() - began:.0f}s",
                            flush=True,
                        )
                return
            except (requests.RequestException, ValueError) as e:
                if attempt == 3:
                    raise RuntimeError(
                        f"Range failed: {name} bytes {start}-{end}: {type(e).__name__}"
                    ) from None
                time.sleep(2**attempt)

    errors = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except RuntimeError as exc:
                    errors.append(str(exc))
                    print(str(exc), flush=True)
    finally:
        for fd in descriptors.values():
            os.close(fd)
    if errors:
        raise RuntimeError(
            f"{len(errors)} ranges failed; rerun the same command to resume verified ranges"
        )
    for item in files:
        name = item["path"]
        target = output / name
        partial = output / (name + ".partial")
        source = target if target.exists() else partial
        expected = item.get("lfs", {}).get("oid")
        if expected:
            h = hashlib.sha256()
            with source.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024**2), b""):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError(f"SHA256 mismatch: {name}; do not use this checkpoint")
        if source == partial:
            partial.rename(target)
        print(f"Verified {name}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="uukuguy/prometheus-7b-v1.0-fp16")
    p.add_argument("--revision", default="88dddaa493cb90c372e736a7804ceea798e2d369")
    p.add_argument("--output", default="artifacts/models/source-fp16")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk-mb", type=int, default=16)
    args = p.parse_args()
    download(args.repo, args.revision, args.output, args.workers, args.chunk_mb)

#!/usr/bin/env python3
"""Record a real browser evaluation against a running local Prometheus server."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from prometheus_eval.io import file_hash, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", default="docs/media")
    parser.add_argument("--browser", help="Optional Chromium executable path")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw = Path("artifacts/demo/browser-video")
    raw.mkdir(parents=True, exist_ok=True)
    console_errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=args.browser)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
            record_video_dir=str(raw),
            record_video_size={"width": 1440, "height": 1000},
            reduced_motion="reduce",
        )
        page = context.new_page()
        page.on("pageerror", lambda error: console_errors.append(str(error)))
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function(
            "document.getElementById('device').textContent === 'Local model ready'"
        )
        health = page.request.get(args.url + "/api/health").json()
        assert (
            page.request.post(
                args.url + "/api/evaluate", data={}, headers={"Origin": "https://example.invalid"}
            ).status
            == 403
        )
        assert page.request.post(args.url + "/api/evaluate", data={}).status == 400
        assert page.locator("body").evaluate("el => el.scrollWidth <= window.innerWidth")
        page.wait_for_timeout(1500)
        page.locator("summary").click()
        page.locator("#rubric").scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
        page.locator("summary").click()
        page.evaluate("window.scrollTo(0, 0)")
        start = time.perf_counter()
        page.get_by_role("button", name="Evaluate response").click()
        page.wait_for_function(
            "document.getElementById('state').textContent.startsWith('Complete')", timeout=240000
        )
        elapsed = time.perf_counter() - start
        assert page.locator(".score").count() == 1
        assert len(page.locator(".feedback").inner_text()) > 20
        page.screenshot(path=str(output / "demo-poster.png"))
        page.wait_for_timeout(4000)
        page.locator(".bench").scroll_into_view_if_needed()
        page.wait_for_timeout(3500)
        page.screenshot(path=str(output / "demo-results.png"))
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(1000)
        video = page.video
        context.close()
        video_path = Path(video.path())
        # Verify responsive layout separately; it is not spliced into the recording.
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(args.url, wait_until="networkidle")
        assert mobile.locator("body").evaluate("el => el.scrollWidth <= window.innerWidth")
        mobile.screenshot(path=str(output / "demo-mobile.png"), full_page=True)
        browser.close()
    if console_errors:
        raise RuntimeError(f"Browser errors: {console_errors}")
    mp4 = output / "prometheus-m1-demo.mp4"
    gif = output / "prometheus-m1-demo.gif"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(mp4),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-filter_complex",
            "setpts=PTS/4,fps=5,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse",
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
    )
    record = {
        "recorded_utc": utc_now(),
        "backend": health["backend"],
        "evaluation_wall_seconds": elapsed,
        "browser_console_errors": console_errors,
        "viewport": [1440, 1000],
        "mobile_viewport": [390, 844],
        "mp4_speed": "1x, continuous real browser capture",
        "gif_speed": "4x preview",
        "checks": [
            "real evaluation completed",
            "nonempty feedback and score",
            "invalid request rejected",
            "cross-origin request rejected",
            "desktop and mobile do not overflow",
            "no browser exceptions",
        ],
        "artifacts": {
            p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
            for p in (mp4, gif, output / "demo-poster.png")
        },
    }
    write_json(output / "recording.json", record)
    result = Path("artifacts/demo/live_result.json")
    if result.exists():
        write_json(output / "demo-result.json", json.loads(result.read_text()))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

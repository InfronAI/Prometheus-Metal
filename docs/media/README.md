# Demo recording

[Play the continuous MP4](prometheus-m1-demo.mp4) · [4× animated preview](prometheus-m1-demo.gif) · [Local video page](index.html)

The video is a continuous Playwright browser capture of a real local evaluation by the converted Prometheus v1 7B model on an M1 Pro. It shows the supplied instruction, response, reference and scoring rubric; the actual wait for inference; generated feedback and score; timing and MLX memory; and the saved experiment notebook. It contains no simulated outputs or spliced benchmark results.

The MP4 runs at **1× wall-clock speed**. The smaller GIF is explicitly a **4× preview**. FFmpeg transcodes the raw WebM to H.264/YUV420p with fast-start metadata; the source WebM remains in ignored `artifacts/demo/browser-video/`.

- [recording.json](recording.json): timestamp, backend identity, elapsed evaluation time, browser checks and artifact hashes.
- [demo-result.json](demo-result.json): exact input, raw model output, parsed feedback/score and generation measurements.
- [demo-poster.png](demo-poster.png): actual desktop screenshot after evaluation.
- [demo-results.png](demo-results.png): experiment notebook screenshot.
- [demo-mobile.png](demo-mobile.png): responsive-layout verification.

To record a fresh demo after running the experiments:

```bash
# Terminal 1, from the project root:
source .venv/bin/activate
prometheus serve --results experiments/mac-validation

# Terminal 2:
source .venv/bin/activate
# Install Chromium only if Playwright does not already have it:
python -m playwright install chromium
# ffmpeg must be on PATH, for example installed with Homebrew.
python scripts/record_demo.py
```

Recording also checks the live feedback/score, JavaScript errors, request validation, origin protection and desktop/mobile horizontal overflow. The local demo is not deployed or exposed publicly.

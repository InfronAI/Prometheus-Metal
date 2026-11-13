# Reproduction protocol and limits

This repository distinguishes **implemented functionality**, **measured local experiments**, and **unreproduced paper results**. Running a command to completion does not establish that the original scientific result has been reproduced.

The reference is [Prometheus, arXiv:2310.08491v2](https://arxiv.org/html/2310.08491v2), the original v1 work, not Prometheus 2. The starting repository commit is `60e2c58956d670532d48ae94fefe278809b98aee`.

## Experiment correspondence

| Paper component | Local implementation | What can be measured here | Remaining difference |
|---|---|---|---|
| §3.1 / Appendix G: rubric → paraphrase → instruction/reference → five responses/feedback | `prometheus synthesize` | Bounded local-teacher construction, raw generations, sentence-count and score checks | New illustrative seeds and a local teacher; original author seeds and historical GPT-4-0613 generation are not recreated |
| §3.2 / Appendix C: feedback-first supervised learning | `prometheus train` | MLX QLoRA, completion-only cross entropy, real released training examples, held-out loss, reloadable adapters | Continuation from an already trained 7B checkpoint; not full-parameter training from Llama-2-Chat |
| Tables 3–4: absolute scoring | `prometheus suite` | Seen/unseen Feedback Bench, Vicuna, MT-Bench, FLASK; three draws; Pearson, Spearman, Kendall τ-b | Local samples and 4-bit weights; all 320 released MT-Bench rows lack GPT-4 labels |
| Table 5: preference ranking | Relative benchmark mode | HHH categories and MT human judgments; ties and failed draws retained | HHH tie resampling is capped, unlike the original unbounded procedure |
| Table 6: training ablations | Train `full`, `no_reference`, `no_rubric`, `score_only` separately | Same selected examples, initial checkpoint, LoRA layers, seed and update budget | Short QLoRA continuation does not replace four from-base full-training runs |
| Reference-material sensitivity | `configs/prompt-ablations.json` | Same checkpoint and evaluation rows, altered inference inputs | **Inference ablations are not Table 6 training ablations** |
| Table 6: base-model ablations | Configurable MLX / Transformers model and chat format | Infrastructure supports independently supplied base models | Llama-2, Vicuna, Code-Llama training comparison not executed |
| §5.1 / Appendix E,J: human scores and feedback preferences | `prometheus human-study` | Blinded export, annotation CSV, real-label aggregation; FLASK's existing human labels | New human annotators have not been recruited; no synthetic “human” judgments |
| Appendices A–B: corpus and rubric diagnostics | `scripts/analyze_dataset.py` | Full released corpus counts, length distributions, word-level ROUGE-L random/nearest criterion overlap | No claim of semantic equivalence to the paper's diversity visualization |
| Appendix F: length bias | Every absolute run's `metrics.json` | Response length/score association | Observational correlation is not an adversarial or causal bias test |
| Baseline tables / coarse FLASK training | Generic backend and training interfaces | User-supplied checkpoints and data can be evaluated | Historical proprietary baselines, coarse training split and large reward-model comparison not rerun |

The paper's training setup uses eight A100 80 GB GPUs. Its reported training context is 4096 tokens, AdamW with a StepLR schedule, and learning rate `1e-5`; generation uses temperature `1.0`, top-p `0.9`, repetition penalty `1.03`, and 256 output tokens. These reference settings are from [Appendix C](https://arxiv.org/html/2310.08491v2#A3). Local generation keeps those sampling settings. Local training uses a constant learning rate, four adapted layers, rank 8, batch 1 and at most 2048 tokens, with the difference recorded in its manifest.

## Data provenance and integrity

The [released Feedback Collection](https://huggingface.co/datasets/prometheus-eval/Feedback-Collection/tree/22a339f961c40b07261bea44ec10849c7440b75f) is pinned to `22a339f961c40b07261bea44ec10849c7440b75f`. It contains **99,952 rows**, rather than exactly 100,000. The small prepared training corpus is committed under [experiments/training-data](../experiments/training-data/manifest.json). Each row carries its original source index. The complete downloaded release stays in the local cache or `artifacts/data`.

The seven original benchmark files are preserved byte-for-byte. Their hashes and schemas are in [data_audit.json](../experiments/data_audit.json). HHH has no `tie` field; it is a binary preference task. MT human judgments contains 454 exact duplicate rows; these are retained to preserve the released evaluation population, and the audit exposes them. MT-Bench's missing labels cannot be recovered by treating another model's generated scores as historical GPT-4 ground truth.

Training/validation/test splits are grouped by original instruction text. Each selected group includes exactly one example for each target score. Benchmark instruction overlap is excluded. All training ablations apply the **same full-prompt length filter** before their transformation, preserving paired comparisons. Overlong examples are listed rather than truncating away their feedback or target score.

## Model provenance

`prometheus prepare-model` can download the [original FP32 release](https://huggingface.co/prometheus-eval/prometheus-7b-v1.0/tree/e7fc9f231ee722e81ce594536dfd0796881a34e2). Its three shards total about 27 GB. To reduce download volume, the local experiment uses the [community FP16 export](https://huggingface.co/uukuguy/prometheus-7b-v1.0-fp16/tree/88dddaa493cb90c372e736a7804ceea798e2d369), pinned at `88dddaa493cb90c372e736a7804ceea798e2d369` (about 13.5 GB). This is a derivative checkpoint, not an assertion of bitwise equivalence with the original FP32 release.

The downloader validates exact HTTP byte ranges and the published SHA-256 of every large file. The converter memory-maps a shard and quantizes one tensor at a time on CPU, writing bounded MLX shards. `conversion.json` records the actual source, source hashes, output hashes, quantization and environment. Four-bit affine quantization changes numerical values; it is not full-precision parity.

## Measurement rules

- No evaluation label is placed in the model input. The original instruction/rubric/reference prompt is wrapped in the Llama-2 chat template, with one BOS token.
- Each draw has a reproducible row-derived seed. A malformed result or backend error gets a bounded retry, with **all attempts** retained. Context overflow is recorded immediately. There is no guessed or default score.
- The score-only ablation removes feedback targets and requests only a score. Its parser permits optional prose, so the final report separately counts generations that still contain feedback. Short continuation does not guarantee compliance with the intended no-feedback behavior.
- Absolute correlations use only rows with all requested successful draws, averaging the model draws and the available reference scores. Missing labels and constant predictions produce `null` plus a reason, not zero.
- Bootstrap intervals resample instruction groups, rather than treating multiple responses to one question as independent. Tiny samples remain exploratory.
- MT human ties are kept. For HHH, three resampling rounds are permitted; an exhausted predicted tie counts as incorrect in the all-row accuracy. The result also reports valid-only accuracy, coverage and category macro accuracy.
- Resume validates data, model identity, generation configuration, runtime versions and evaluator source hashes. A changed run must use a new output directory. Only an interrupted final JSONL fragment may be repaired; its bytes are saved.
- `peak_memory_gb` is the process-lifetime MLX allocator high-water mark, not total system memory or process RSS. Tasks sharing a suite process can inherit an earlier task's peak, so per-task values are not independent memory measurements. Timing includes local generation and prefill where reported; model download/loading time is separate.

## Full evaluation and external work

```bash
# All released rows; may take many hours on an M1. Supports safe resume.
prometheus suite --config configs/full-evaluation.json --output experiments/full-evaluation

# Prepare blinded feedback comparisons from two completed absolute runs.
prometheus human-study --left experiments/mac-validation/feedback_seen \
  --right experiments/prompt-ablations/no_reference --output experiments/my-human-study

# After real annotators fill the CSV:
prometheus human-study --left experiments/mac-validation/feedback_seen \
  --right experiments/prompt-ablations/no_reference --output experiments/my-human-study \
  --annotations experiments/my-human-study/annotations.csv
```

An existing ten-pair study comparing the full and no-reference trained adapters is prepared in [experiments/human-study](../experiments/human-study/label_studio_tasks.json). Its [summary](../experiments/human-study/summary.json) contains zero actual annotations; it is not a completed human experiment.

The all-row configuration does not invent the missing MT labels, generate historical proprietary baselines, or conduct a human study. A strict 100% reproduction still requires those materials, original-scale training resources, and real human annotations. The repository does not claim that this has happened.

# The Duck harness (vendored)

Source: the Kaggle dataset [`jeroencottaar/taaf-kaggle-source-share`](https://www.kaggle.com/datasets/jeroencottaar/taaf-kaggle-source-share),
licence **MIT** (as listed on the dataset), downloaded 2026-09-26 unchanged:

- `ARC3-Inference/` — "The Duck" tool-using solver (commit aa69123 per the bundle's git_status)
- `tufa-arc-agi-framework/` — TAAF, the game/benchmark framework it runs in (commit fe9f7c4)
- `kaggle_setup_commands.original.json` — the original notebook's vLLM setup (Qwen3.6-27B-FP8)

Authors (Tufa Labs, from the notebook): Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tesnar,
Stefano Viel. Winner of ARC Prize 2026 ARC-AGI-3 Milestone #1. Write-up:
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133

Our changes live outside this folder (`agent/duck/`, `eval/duck_run.py`); this folder stays as downloaded so diffs to
upstream are easy to see. Outline: `docs/research/DUCK_HARNESS.md`.

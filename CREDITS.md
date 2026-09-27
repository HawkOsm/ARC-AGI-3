# Credits

Third-party code and ideas this repo uses or was shaped by, and exactly how.

## Code in this repo

**The Duck — Tufa Labs** (ARC Prize 2026 ARC-AGI-3 Milestone #1 winner; Harold Bessis, Jeroen Cottaar, Isaiah
Pressman, Andries Smit, Michal Tesnar, Stefano Viel). Source read from the Kaggle dataset
[`jeroencottaar/taaf-kaggle-source-share`](https://www.kaggle.com/datasets/jeroencottaar/taaf-kaggle-source-share)
(licence **MIT** as listed on the dataset), 2026-09-26/27. `agent/duck.py` is **our own rewrite of its design**:
one `python` tool whose code calls `action(...)`, a segmentation view of the screen, before/after frames, the
model-written world-model notes carried between turns, context eviction, the prompt's playing advice (HUD bars are
not puzzle pieces; probe when unsure; search and batch when the goal is clear). No code was copied; the
segmentation, sandbox, prompt text and loop were written from scratch and are simpler than theirs.

**Public notebook** [`keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp`](https://www.kaggle.com/code/keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp)
(a fork of Tufa Labs' notebook serving Qwen3.8-Flash-Next NVFP4) — copied unchanged into `notebooks/duck_repro/`
as our Kaggle baseline and template.

**ARC Prize Kaggle starter kit** — [arcprize/ARC-AGI-3-Kaggle-Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
@ `eeb1535`: the project began from it (2026-09-25); the `Makefile` has since been rewritten and its scripts removed.

## Studied, not shipped

**Official sample — "ARC3 Sample Submission - Stochastic Goose"**
([Kaggle](https://www.kaggle.com/code/inversion/arc3-sample-submission-stochastic-goose), itself from
[DriesSmit/ARC3-solution](https://github.com/DriesSmit/ARC3-solution), Tufa Labs), pulled with
`make pull-sample` into `reference/` (gitignored). A CNN that learns which actions change the frame.
Not used. Two operational patterns were adopted after reading it: capping the framework's
ever-growing `self.frames` list (`append_frame` override, `KEEP_FRAMES`), and a wall-clock guard in
`is_done` (it uses 8 h − 5 min; we use `TIME_BUDGET_S`). No code copied.

**Graph-Based Exploration for ARC-AGI-3** — Rudakov, Shock & Cowley,
[arXiv:2512.24156](https://arxiv.org/abs/2512.24156) (3rd place, ARC-AGI-3 Preview;
[code](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore)). Conceptual prior art for the
floor agent's shape: segment the frame into components, prioritise click targets by salience,
keep a directed graph of explored states, walk to the nearest state with untested actions. Only
the abstract-level idea was used (recalled, not re-read); their code has **not** been read. If it is
studied later, update this entry with what was taken.

**Architecture brief** (from the project owner, describing a team that iterated 79+ versions on
this competition): volatile-cell-masked state hashing, a learned transition model with verified
BFS planning, object-centric avatar identification, no-op/undo interface learning,
programmatic-first with LLM-as-rescue. The former `agent/brain/` (deleted 2026-09-27, in git history)
implemented the first four; the project has since moved to an LLM-first design (`agent/duck.py`).

**Tufa Labs' harness repo** (cloned 2026-09-26 to `.claude/scratch/llm/duck-harness`; that GitHub copy carries no
licence file) was read for its serving setup: one vLLM OpenAI-compatible server inside the Kaggle notebook from an
offline wheelhouse installed with `--target`, prefix caching, `--reasoning-parser qwen3`, thinking-mode sampling
(temperature 0.6, top-p 0.95). `notebooks/llm_prep/` and `notebooks/llm_dev/` were written from scratch.

**vLLM recipe for Qwen3.8-27B** ([recipes.vllm.ai](https://recipes.vllm.ai/Qwen/Qwen3.8-27B), read
2026-09-26): vLLM ≥ 0.17 supports the model; the RTX PRO 6000 / Blackwell notes informed the serve flags.

**Model** — [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) @ `017b9c7af6` (Apache-2.0),
used as-is (no training). On Kaggle it is attached from the public mirror
[michaelpoluektov/qwen3-8-27b-fp8](https://www.kaggle.com/models/michaelpoluektov/qwen3-8-27b-fp8),
accepted only after `notebooks/llm_prep/` checks every file against the Hugging Face sha256 manifest
(`notebooks/llm_prep/hf_manifest.json`).

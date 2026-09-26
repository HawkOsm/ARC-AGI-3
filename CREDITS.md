# Credits

Third-party code and ideas this repo uses or was shaped by, and exactly how.

## Code in this repo

**ARC Prize Kaggle starter kit** — [arcprize/ARC-AGI-3-Kaggle-Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
@ `eeb1535` (cloned 2026-09-25). `Makefile`, `scripts/play_local.py`, `scripts/build_notebook.py`,
`scripts/slim_framework.py`, `notebooks/kernel-metadata.json` and the original random
`agent/my_agent.py` (frozen at `experiments/2026-09-25_random_baseline/my_agent.py`) come from it.
Local modifications, each commented in place: uv-managed venv, `~/.kaggle/access_token` fallback,
pinned framework commit + `arc-agi`/`arcengine`/`numpy` versions matching Kaggle's runtime,
`AGENT=` override for the notebook build, CPU accelerator, project make targets. Its README is kept
as [`docs/STARTER_README.md`](docs/STARTER_README.md).

**ARC-AGI-3-Agents framework** — [arcprize/ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents),
vendored by `make setup` into `vendor/` (gitignored), pinned to `135f20a`. `agent/brain/kaggle_agent.py`
subclasses its `Agent` and overrides `do_action_request`/`take_action`/`append_frame` to harden it
(see AGENT.md "Verified interface facts").

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
programmatic-first with LLM-as-rescue. `agent/brain/` implements the first four; see AGENT.md for
where the implementation deviates and why.

**"The Duck" — Tufa Labs, 1st place ARC-AGI-3 Milestone #1** ([arcprize.org blog](https://arcprize.org/blog/arc-prize-2026-milestone-1);
harness repo cloned to `.claude/scratch/llm/duck-harness`, **no license**, so nothing is copied). Read
2026-09-26 for serving setup only: running one vLLM OpenAI-compatible server inside the Kaggle notebook
from an offline wheelhouse installed with `--target`, attached model snapshot, prefix caching,
`--reasoning-parser qwen3`, Qwen's thinking-mode sampling (temperature 0.6, top-p 0.95). Our
`notebooks/llm_prep/`, `notebooks/llm_dev/` and `agent/brain/llm.py` were written from scratch; the
agent design (brain in charge, LLM proposes short plans) is ours, not theirs (theirs: the LLM writes
and runs Python in a REPL).

**vLLM recipe for Qwen3.8-27B** ([recipes.vllm.ai](https://recipes.vllm.ai/Qwen/Qwen3.8-27B), read
2026-09-26): vLLM ≥ 0.17 supports the model; the RTX PRO 6000 / Blackwell notes informed the serve flags.

**Model** — [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) @ `017b9c7af6` (Apache-2.0),
used as-is (no training). On Kaggle it is attached from the public mirror
[michaelpoluektov/qwen3-8-27b-fp8](https://www.kaggle.com/models/michaelpoluektov/qwen3-8-27b-fp8),
accepted only after `notebooks/llm_prep/` checks every file against the Hugging Face sha256 manifest
(`notebooks/llm_prep/hf_manifest.json`).

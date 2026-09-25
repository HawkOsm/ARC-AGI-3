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
vendored by `make setup` into `vendor/` (gitignored), pinned to `135f20a`. `agent/core/framework.py`
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
programmatic-first with LLM-as-rescue. `agent/core/` implements the first four; see AGENT.md for
where the implementation deviates and why.

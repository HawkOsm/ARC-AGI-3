# What the community is doing (verified digest, 2026-09-26)

Collected by `agy` (Claude Opus 4.6, spec in `.claude/scratch/research/agy_spec.md`, raw output in
[`agy_community_survey_raw.md`](agy_community_survey_raw.md)), then **checked by Claude Code against
the primary sources**. Only rows marked ✅ were re-read at the source; everything else is agy's
report and may be wrong (agy read many pages only through search snippets, and gave one wrong URL).

## The one thing that matters most

✅ **The top Kaggle ARC-AGI-3 entries run an open-weight LLM locally inside the notebook — pre-trained,
used as-is, no training.** Source: [ARC Prize 2026: ARC-AGI-3 Milestone Prize #1](https://arcprize.org/blog/arc-prize-2026-milestone-1).

| place (Milestone #1, June 2026) | team | model inside the notebook | approach |
|---|---|---|---|
| 1 | Tufa Labs, "The Duck" | Qwen 3.6 27B FP8 | LLM writes and runs Python in a live REPL; old messages evicted to play indefinitely |
| 2 | Reki | Gemma-4-31B | recent frames rendered as labelled images; model returns one JSON action |
| 3 | "forge" (Md Boktiar Mahbub Murad) | Gemma-4-31B | similar; best run had "all of the extra machinery" turned off |

Lessons stated in that post: the winners found "hand-crafted tools actually hurt the model; letting it
improvise worked better" — the same direction as our own ablation (special-case rules measured
nothing or hurt).

Where we stand: our programmatic agent scored **0.44** on the public LB; the top is **19.45**.

## Research agents (outside Kaggle) — powerful, but NOT runnable in the competition

✅ These are real and their numbers are quoted correctly, but they drive **frontier API models over
the internet** (GPT-5.x, Claude Opus); Kaggle's scored run has no internet.

| paper | result (verbatim) | model |
|---|---|---|
| [Twin: test-time digital twin](https://arxiv.org/abs/2608.14490) (Aug 2026) | "Twin clears 179 out of 183 levels (97.8%), and does so more efficiently than humans in 158 out of 179 levels (88.3%)." | "a frontier coding agent" |
| [Tycho: active abstraction](https://arxiv.org/abs/2607.28287) (Jul 2026) | "GPT-5.6 Sol and Opus 5 both reach 100.00 RHAE and complete all 183 levels." | GPT-5.6 Sol, Claude Opus 4.8 / 5 |
| [OPINE-World](https://arxiv.org/abs/2607.01531) (Jul 2026) | "OPINE-World solves 20 of 25 games without per-game training and reaches an action-efficiency score of 78.4 against the human baseline." | LLM + counterexample-guided synthesis |

What they share (the idea, which does transfer): **the agent writes an executable model of the game
("world model" / "digital twin"), and it is not allowed to act on it until the model reproduces every
transition seen so far; mismatches (counterexamples) repair the model.** Twin: "The harness enforces
that an action is not made until the program reproduces every previous observed game transition."

## Other reported items (agy, not re-verified)

- ARC-AGI-3 Preview (2025): graph exploration "Just Explore" 3.64%, "Blind Squirrel" (state graph +
  small ResNet value model, i.e. training) 6.71% (2nd), DSL + LLM "Fluxonian" 8.04%.
- "Explore Before You Solve" (arXiv 2605.25931): claims the public games can be beaten by
  non-intelligent heuristics, so only the private set measures the real thing — consistent with our
  0.70 hidden/public calibration.
- Kaggle discussions: Tufa Labs not open-sourcing for Milestone #2; long queues (~8 h) for the RTX
  machines.
- ARC-AGI-2 / 2024-2025 winners (NVARC, the ARChitects, MindsAI): test-time *training* of models —
  out of scope under our no-training rule.

## Implications for this repo (Claude Code's reading, for the owner to decide)

1. The competitive entries are **LLM-driven with a local open-weight model**; a purely hand-coded agent
   is ~40x behind on the LB. This fits the no-training rule (inference only) but is ROADMAP Phase 3 and
   needs a GPU notebook (RTX 6000 queue) and a local test setup (our laptop GPU has 8 GB — too small
   for a 27-31B model; a small model could test the plumbing only).
2. The owner's World-of-beliefs design and the "executable world model + verify against every past
   transition" idea are the same idea at different levels: our `world.levels` map already verifies
   every transition exactly; what the research agents add is a *generalising* model (code) instead of
   a lookup table.
3. "Hand-crafted tools hurt" (winners) and our ablation agree: keep the core small; let the model (or
   the decision rule) do the work.

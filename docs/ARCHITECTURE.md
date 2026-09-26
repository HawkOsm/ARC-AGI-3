# Architecture — start here

One page to find your way around. Details live in AGENT.md (rules and constraints), ROADMAP.md
(plans) and docs/tests/LOG.md (every measured run).

## The picture in one paragraph

We write an **agent** that plays unknown 64x64 grid games. It is a set of fixed, game-agnostic
functions (`agent/brain/`) plus a **World of beliefs** that starts empty and fills up while it
plays. Those files are glued into **one file** (`agent/my_agent.py`), which is pasted into a
**Kaggle notebook**. Kaggle then runs it against hidden games. Locally we measure every change
with the **eval harness** (`eval/`) before it is allowed to replace the **champion**. Phase 3,
now in progress, adds an optional **LLM advisor**: a local open-weight model (Qwen3.8-27B-FP8)
served by vLLM *inside* the Kaggle notebook, which proposes actions the brain carries out.

## The big map

```
  YOU EDIT                    BUILD                     SHIPS                        RUNS ON
  agent/brain/*.py  ──make build──►  agent/my_agent.py  ──make notebook──►  notebooks/submission.ipynb ──► Kaggle
      │                              (generated, never                       (you run `make submit`
      │                               edit by hand)                            yourself; 1 submission/day)
      │
      └──► eval/run.py  (local, offline games in environment_files/, real scorecard)
               │  paired run vs eval/champion/my_agent.py, N≥30 + regression check
               ▼
           promote = copy into eval/champion/  (local only; never implies a Kaggle submit)
```

## One game step (what the agent does every turn)

`Brain.next_action()` in `agent/brain/brain.py` — read this file first.

```
 frame ──► 1 LOOK    settled picture (last animation layer)                         look.py
           2 EVENTS  died? level up? whole game restarted?                           brain.py
           3 LEARN   first look at a new level (item kinds, patterns)                look.py
                     counters to ignore, which colour is "me", what each action did, learn.py
                     what touching each colour did, goals from wins                  goals.py
                     record  (screen, action) -> next screen  in the level map       world.py
           4 DECIDE  a) advisor on + level unsolved -> next action of the LLM's plan  llm.py
                     b) otherwise (or no usable answer) -> the brain decides:        decide.py
                        known solution -> replay it; else the cheapest untried
                        (screen, action) = (walk distance + 1) / chance of something new;
                        else reset / finer click grid / random
 ◄── action (id, x, y)
```

The world's beliefs (`world.py`): `counters`, `avatar`, `kinds`, `colour_roles`, `patterns`,
`actions`, `rules`, `levels` (the per-level map of screens), `goals`, `deaths`. Everything is
learned during play; nothing is trained offline and nothing is keyed on a game's name.

## The files

| file | what it is | read it when |
|---|---|---|
| `agent/brain/brain.py` | the loop above | always first |
| `agent/brain/world.py` | the belief containers (World, LevelMemory, Screen, Tally) | you wonder "where is X stored" |
| `agent/brain/look.py` | screen reading: objects, changed blobs, first look | perception questions |
| `agent/brain/learn.py` | writes beliefs: counters, avatar, touch roles, action effects | the agent believes something wrong |
| `agent/brain/goals.py` | goal hypotheses from what changed when a level was won | goal questions |
| `agent/brain/decide.py` | reads beliefs, picks the action (exploration frontier) | the agent does something dumb |
| `agent/brain/llm.py` | **the LLM advisor**: prompt, HTTP client, answer parser, shared pause | LLM questions |
| `agent/brain/settings.py` | every tunable number, with its reason | tuning |
| `agent/brain/kaggle_agent.py` | `MyAgent`: plugs the brain into Kaggle's framework + safety guards | crashes, threads, time/memory limits |
| `scripts/build_submission.py` | glues the brain files into `agent/my_agent.py` (`MODULE_ORDER`) | adding a module |
| `eval/run.py` | local scoring: single / paired / swarm | measuring anything |
| `eval/regression_check.py`, `mask_check.py`, `interface_checks.py` | CI guards | `make ci` fails |
| `eval/kaggle_sim.py` | runs the real notebook against a local competition-mode gateway | "does it behave the same on Kaggle?" |
| `eval/mock_llm.py` | fake LLM server (random legal plans, can fail or answer garbage) | testing advisor plumbing without a GPU |
| `eval/explain.py` | plays a game for a while, then prints the World's beliefs in words | "what does it think this game is?" |
| `notebooks/llm_prep/` | Kaggle CPU notebook: verifies the model mirror, builds the vLLM wheelhouse | LLM setup on Kaggle |

## The LLM part (Phase 3) — what runs where

```
 KAGGLE NOTEBOOK (RTX PRO 6000, 96 GB, internet OFF)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  vLLM server  (one, started once)  Qwen3.8-27B-FP8 ◄── attached Kaggle   │
 │     ▲  OpenAI-compatible HTTP, localhost          model mirror (verified │
 │     │                                             sha256 vs Hugging Face)│
 │     │  installed offline from the wheelhouse ◄── output of arc3-llm-prep │
 │     │                                                                    │
 │  main.py ─► Swarm: one thread per game ─► MyAgent ─► Brain               │
 │                                              └─► llm.advisor() (shared)  │
 │                                                   ask -> plan of ≤8 actions
 │                                                   no/slow/bad answer -> brain decides
 └──────────────────────────────────────────────────────────────────────────┘
```

- The advisor is **off** unless the environment variable `ARC3_LLM_URL` is set, so the same
  `my_agent.py` still works as the plain brain (CPU notebook, local eval).
- The brain keeps learning from every step, even when the LLM chose it. On solved levels the
  brain replays the known route, and it alone may RESET: a RESET with no steps taken restarts
  the whole game.
- A plan is dropped on any surprise: death, level up, or an advised action that changed nothing.
  Three failed requests in a row pause the advisor for all games for 5 minutes.
- Locally (8 GB laptop GPU) the 27B model cannot run, so local tests use `eval/mock_llm.py` for
  plumbing only. **Play quality can only be measured in a Kaggle GPU notebook.**

Note: ROADMAP.md still says "no LLM-primary agent / LLM as rescue only". When the advisor is on,
the LLM picks most actions on unsolved levels. That is the owner's 2026-09-26 decision (going
local LLM after the community research) and must pass the same promotion rule as any change.

## Where things are

| path | what | in git? |
|---|---|---|
| `eval/champion/` | the current champion agent + CHAMPION.md | yes |
| `eval/results/<stamp>_<label>/` | every eval run: runs.jsonl + summary.md | no |
| `docs/tests/LOG.md`, `IDEAS_TRIED.md` | what was run and what it measured / every idea tried | yes |
| `docs/research/` | community research digest (verified) + raw agy output | yes |
| `experiments/<date>_<name>/` | frozen agent versions from past experiments | yes |
| `models/` | local model weights (Qwen3.8-27B-FP8, ~31 GB) | no (gitignored) |
| `.claude/scratch/<task>/` | scratch work, logs, pushed-notebook copies | no |
| `environment_files/` | the 25 public games, cached offline | no |

## Status (2026-09-26)

- Champion: the pruned brain (local only; Kaggle still has the older floor agent, LB 0.44).
- The LLM advisor is written and plumbing-tested against the mock server (0 errors, 10 runs).
  It has not been tested with a real model yet.
- Kaggle prep notebook `hawkosm/arc3-llm-prep` (CPU) is running: model-mirror hash check +
  wheelhouse.
- Next: a GPU dev notebook that serves the model and plays the 25 public games with the advisor
  on vs off. Its result decides whether the advisor goes further.

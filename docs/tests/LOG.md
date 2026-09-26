# Test Log

See [`README.md`](README.md) for the entry format and convention. Newest first.

---

## 2026-09-26 — Step 3: control modes (selection), level lessons, surprises → LLM, several targets
- Change: `agent/brain/control.py` `ControlMap` — control modes (which object each key moves), switches (a click /
  ACTION5 that changed something; the next key press shows the mode), how to select an object (click it if clicks
  select, else the switch key); `agent/brain/lessons.py` `GameLessons` — how each level was won (mover colour →
  target colour, controls), NEW / CHANGED objects vs earlier levels; object question gets WHAT FINISHED EARLIER
  LEVELS, news, control modes, goals already reached, SURPRISE; a click / switch doing something never seen in the
  game → one key press to see what it produced, then the LLM is asked at once (owner: "when a click does something new
  the LLM should decide"); a probe switch always gets that key press; steering selects its mover first; arriving
  while more target-like objects of the target's colour remain = DONE (kept, next goal); an LLM test that would move a
  placed piece is skipped; shapes compared with holes filled (ar25's dotted pieces vs solid outlines); goal list
  won-colour boosts; `Controls` removed. Fixes found in the debug runs: endless key presses after a no-op select,
  arrow keys counted as switches, marker changes read as a 21-cell jump (click not seen as a switch), bottom bar and
  step counter counted as "more targets", LLM tests interrupting steering.
- Command: `eval/debug_run.py ar25 --hypothesis --steps 300 --trace` (7 runs); then
  `ARC3_LLM_MODE=hypothesis eval/run.py --swarm --max-actions 200 --seed 0 --label hyp3_local_200` (local Qwen3.5-4B).
- Result: ar25 debug (last run): level 1 in 26 actions, level 2 in 35, level 3 not solved (the L piece's outline lies
  across the mirror line: the mirror must move first). 25 games, 200 actions, one run each:
  **step 3 0.519 / 9 levels** vs step 2 0.690 / 11 vs brain 0.251 / 6. Per game: sc25 2→0 levels, vc33 2→1, lp85
  2.78→0.06, bp35 0→1, r11l 0.11→0.85; ar25 level 1 took 50 actions in the swarm run vs 26 in the debug run (same
  code). In both LLM runs 110–130 of the 200 actions are the brain's fallback exploration.
- Verdict: INCONCLUSIVE — one run per arm with a sampling LLM; more runs of both arms queued.

## 2026-09-26 — Step 2 on all 25 games (hypothesis mode, local 4B, 200 actions)
- Command: `ARC3_LLM_MODE=hypothesis eval/run.py --swarm --max-actions 200 --seed 0 --label hyp_local_200`
- Result: **0.690 / 11 levels** vs brain alone 0.251 / 6 (`brain_swarm_200`). Gains: ar25 0→2 levels, sc25 0→2,
  tn36 0→1, vc33 1→2, lp85 0.10→2.78; losses: bp35 1→0, m0r0 4.76→0.93, lf52 0.31→0.13. 0 errors, 1493 s wall.
- Verdict: INFO — one run; first sign the object pipeline beats the brain outside ar25.

## 2026-09-26 — Step 2: probes, steering toward the picked goal, switch discovery — ar25 levels 1 and 2 solved
- Change: `agent/brain/steer.py`: `Prober` (level start: each key once, one click per object kind — owner's "click
  to explore"); `Controls` (latest effect of each key on an object; frames where it was reshaped/hidden are no
  evidence); `Navigator` (BFS over positions the learned steps reach, around blocked moves, nearest reachable if not
  exact); `Steerer` (follows the plan to put the mover's centre on the target's, checks each move, replans; when a
  whole direction is missing it tries ONE switch at a time — ACTION5 or clicking the objects involved — then re-tests
  the keys; gives the goal up if arriving does not finish the level). LLM roles kept as object purposes in later
  questions. No LLM calls while steering.
- Command: `eval/debug_run.py ar25 --hypothesis --steps 200 --trace` (6 runs while fixing: a KeyError when a key's
  effect changed mid-plan; false "blocked" + 45 wasted unlock moves; lost tracking when the L overlaps the target).
- Result (final run): **ar25 level 1 in 27 actions (human baseline 32) and level 2 in 26 (baseline 50)** — both
  capped at 115 per level; 0 lives lost; level 3 not solved by step 200. Level 2 needed the switch: the keys moved the
  mirror line until ACTION5 (or clicking the black shape — confirmed by hand in the engine) made ACTION1/2 move the
  shapes; the steerer found it by itself. The brain alone never finished ar25 level 1 (0 levels in every run).
- Verdict: INFO — first game solved by the object/goal/steer pipeline; not yet measured on the other 24 games.

## 2026-09-26 — Goals as a ranked list of shape correspondences; the LLM picks (local 4B, ar25)
- Change: owner's direction — goals come from object purposes and shapes, not a list of genres. `shapes.py`: pairwise
  correspondences between object kinds (identical / turned or flipped / scaled / fills a gap / fits inside / same
  colour), bounded (kinds, not pairs; <=3 relations per object; cached per level). `goals.py`: `GoalTracker` renamed
  `Goals` (owner); `GoalBoard.possible_goals()` = correspondence pairs (+ "clear all copies" for kinds with >=3
  copies), ranked by shape strength x reacts-to-actions x LLM picks x relation kinds that held at earlier wins;
  `Goals.purposes()` (how each object reacted per action), `Goals.progress_text()`. The object question now shows the
  ranked list and purposes; the answer has `goal_choice` (a number, 0 = none); the pick re-ranks the list.
- Command: shape relations on all 25 first screens (timing); `eval/debug_run.py ar25 --hypothesis --steps 80`.
- Result: first version of the relations exploded (tr87 521 relations, 216 ms: rectangles "fit" every frame) ->
  tightened (irregular pieces only for fills-gap, snug fits-inside, 3 per object): worst tr87 41 relations, 118 ms,
  most games < 20 ms. ar25: the brain's list has "#5 and #6 overlap (same shape)" at #1 from the first screen; the 4B
  model **picked it in 15 of 20 answers** (before the list it never named the yellow block in 40 answers) and named
  #5 as controlled. Prompt median 2,267 tokens (hex question ~4,700), answers 4.3 s median. Progress for the picked
  goal: distance 33-55 -> 4.2 cells, box overlap 1.00 near the end by chance (the brain still explores; step 2 steers).
  Tests 7/13 correct this run. 0 levels.
- Verdict: INFO — the right goal is now found and picked on ar25; next: steer the brain toward it (exact coverage,
  not box overlap). `_is_just_moving` removal (owner) still pending, as its own measured change (it alters the default
  brain's win learning).

## 2026-09-26 — Hypothesis mode, step 1 (object questions + tested predictions), local 4B on ar25
- Change: `agent/brain/scene.py` (numbered objects with ids kept through movement, twins, partial covering and
  splits; composites = parts that moved together touching >=2 times and never apart; relations: mirror images across
  a line, same size/shape in different colours; per-action effects). `ARC3_LLM_MODE=hypothesis`: every 4 brain
  actions the brain asks the object question; the answer (JSON schema) gives rule, controlled ids, goal, roles and ONE
  test action with a predicted effect; the brain runs the test and judges it (`Lab.check`); verdicts feed the next
  question. Debug runner `--hypothesis --show-question`; traces carry verdicts; 5 new tests (35 total).
- Command: `eval/debug_run.py ar25 --hypothesis --steps 80 --trace` (4 runs while fixing the tracker).
- Result: owner's reading of ar25 = grey L (controlled) mirrored as a black L across the light-blue line, goal = cover
  the yellow 9x9 block. The brain's own scene finds "#7 is where #5 would be mirrored across #2" and "#5, #6 same
  shape". Tracker fixes were needed (twins matched by common displacement; ids kept when partly covered; strict
  composites — a loose rule merged the mirror line with the L). Final runs: tests 13/7 and 9/10 correct/wrong over
  ~20 questions; the model converges to the right controls ("ACTION3 moves #5 right and the black group left, ACTION4
  the reverse") but never names the yellow block as the goal ("#15 covers #5", "#5 returns to its uncut state");
  the same-shape relation disappears once #5 is partly covered, so later questions lose that clue. 0 levels in 80 steps
  (the brain still chooses all non-test actions; step 2 would use the goal).
- Verdict: INFO — the object question gets the mechanics right where the hex question never did (9,292 v5 answers,
  no mirror); goal inference is the open problem.

## 2026-09-26 — GPU dev run v5 (27B, no thinking, structured JSON): reliable answers, far worse play
- Change: structured JSON output + compact JSON + `--language-model-only` (agent from before the in-flight limit).
- Command: `notebooks/llm_dev/build.py --arm nothink:0:5400 --max-actions 1000`; brain reference via
  `brain_at_same_actions.py` (advisor-off replay per game, seed 0, same action counts).
- Result: 9,292 requests, **0 errors, 9,292/9,292 finish=stop**, median 12.7 s (p90 16.9 s), ~94 tokens. The LLM chose
  **98%** of actions (19,313 / 19,723). **LLM 1 level / 0.094 vs brain 17 levels / 0.682 at the same per-game actions.**
  Answers: 1,800 distinct plans in 9,292; the most common are one key repeated ("ACTION4" x8: 691, x4: 433,
  "ACTION1" x8: 298); theories mostly guess a genre ("Sokoban-style"); "mirror"/"reflect" 0 times in 9,292.
- Verdict: ❌ an LLM that drives from a stateless hex prompt is much worse than the brain. v4's 12 levels came
  mainly from brain actions filling in for failed answers. Next step to be chosen by the owner from the replays.

## 2026-09-26 — Local 4B LLM fully in charge (after the queueing fix) + ar25 debug run
- Change: advisor with an in-flight limit (8 locally) so the opening burst no longer times out and pauses it;
  `eval/debug_run.py` / `make debug` (one game, a printed block per decision: action, who/why, options weighed,
  result); prompt fix: first-look guesses were presented to the LLM as "Earlier wins came with ...".
- Commands: `make eval-llm BUDGET=200` (25 games, swarm, seed 0) vs `eval/run.py --swarm --max-actions 200 --seed 0`;
  `eval/debug_run.py ar25 --llm --steps 120 --trace`.
- Result: **LLM 4B: 2 levels / 0.170 vs brain 6 levels / 0.251** at 200 actions per game. ar25 with the LLM: all 120
  actions from the LLM (77 questions, 453 s), 0 levels, 64/120 actions changed nothing; its theory never mentions the
  mirror/reflection (owner's reading of ar25: a grey L-shape and a yellow goal mirrored across a middle line) and kept
  citing "previous wins" that were only the first-look guess (prompt bug, fixed + test).
- Verdict: ❌ the 4B model in charge is worse than the brain; kept only as a development stand-in.

## 2026-09-26 — Visibility: tests, Run Board, Agent Replay (owner: "I need to see what is going on")
- Change: `tests/` (pytest, 29 tests: model, reader, advisor against a fake server, brain playing ls20/vc33; `make test`,
  in `make ci` and GitHub CI). `eval/dashboard.py` → Run Board page. `agent/brain/trace.py` + `eval/trace_viewer.py` →
  Agent Replay page (every step: screen, action, who chose it and why, what it did, LLM theory). The explorer now
  records a reason per action (`WorkingMemory.why`), the advisor returns `Advice(plan, theory)`, LLM request logs
  carry the game id. None of this changes a decision (tests + identity below).
- Commands: `make test`; tests mutation-checked (removing the in-flight limit / accepting illegal plan actions each
  turned a test red); `ARC3_TRACE_DIR=.claude/scratch/trace/brain_300 eval/run.py --swarm --max-actions 300 --seed 0`.
- Result: 29/29 pass. Brain traces, 25 games x 300 actions (3.5 MB): share of actions that changed nothing —
  ft09 82% (37 in a row), su15 67%, cd82 53%, tn36 51%, sb26 46%, sk48 44% (35 in a row); most games lose 2+ lives
  (step counter) in 300 actions; 10 levels total (0.291).
- Verdict: INFO — tools for the owner to decide the next step from what the agent visibly does.

## 2026-09-26 — GPU dev run v4 (27B): thinking is unusable; no-thinking LLM ≈ brain at equal actions
- Change: agent with advisor (pre-structured-output, pre-refactor; advisor-off play identical to the champion).
  Arm `nothink` 5,400 s, then arm `think` (16,384 tokens, 1,200 s timeout) 2,700 s, same server.
- Command: `notebooks/llm_dev/build.py --arm nothink:0:5400 --arm think:1:2700 --max-actions 1000`; brain reference:
  `.claude/scratch/llm/kaggle/brain_at_same_actions.py` (each game replayed advisor-off, seed 0, same action count).
- Result: server up in 270 s.
  - `think`: 109 requests, **97 finish=length at 16,384 tokens**, median **685 s**/answer; 20 advised actions in 45 min; 0 levels.
  - `nothink`: 3,525 requests, 0 errors, **2,057 (58%) finish=length at 1,024 tokens** (prose, no plan), median 46.5 s;
    2,772 of 10,997 actions advised; 337-691 actions per game in 90 min.
    **LLM 12 levels / 0.267 vs brain 11 levels / 0.292 at the same per-game actions** (n=1 seed). LLM-only levels:
    cd82, s5i5, tn36 (3.57); r11l 0.92 vs 0.12. Brain-only: ar25, ls20; m0r0 4.76 vs 0.29 (efficiency).
  GPU quota after v4: see `kaggle quota`.
- Verdict: INFO — thinking mode is out (9/10 answers never finish). No-thinking plays about as well as the brain even
  with most answers wasted; structured output (found locally the same day) should fix that. Next Kaggle run: schema
  + compact JSON + `--language-model-only`, nothink only.

## 2026-09-26 — Local dev LLM (RTX 4070 8 GB) + structured JSON answers
- Change: `make llm-local-setup` / `make llm-local` / `make eval-llm`: vLLM 0.30.0 (same as Kaggle) in `.venv-llm/`,
  model `cyankiwi/Qwen3.5-4B-AWQ-4bit@ef85d23beb` (4.0 GB, Apache-2.0; Qwen3.8 is built on Qwen3.5) in `models/`,
  `--language-model-only --max-num-seqs 32 --max-num-batched-tokens 4096` (the defaults ran out of memory while
  profiling CUDA graphs: tried to allocate 4.12 GiB). Server up in ~135 s; 5.7 of 8 GB used; KV cache 47,427 tokens.
  Advisor: vLLM structured outputs (`response_format` json_schema: theory ≤400 chars + plan of 1-8
  `ACTION[1-7]( x y)` strings), JSON-only instructions, and the prompt now says every action counts.
- Command: `.claude/scratch/llm/local/smoke.py ls20,vc33,ft09[,sp80]` (first-screen prompts, thinking off)
- Result: before: **3/3 answers hit the 1,024-token limit** reasoning in prose (counting hex digits) and never
  produced a plan, 17.4 s each (~60 tokens/s single stream). After: **4/4 parse**, 2.1-6.0 s, 80-196 tokens,
  finish=stop.
- Verdict: INFO — local loop works; the same prose failure is likely on Kaggle without the schema (v4 was pushed
  before this fix). Brain-only reference for the local run: swarm, seed 0, 200 actions: 0.251 / 6 levels.

## 2026-09-26 — PLAYABLES step 1 (game mode, one click per kind, keys-first in mixed games): rejected
- Change: `GameMode` from the offered actions; click candidates tagged with their item kind and a per-kind
  click tally (`CLICK_TALLY_BY_KIND`); click priors x0.5 in MIXED games (`MIXED_MODE_CLICK_PRIOR_FACTOR`).
  Both switches off = 50/50 identical to the champion (plumbing check).
- Command: `eval/run.py --agent eval/champion/my_agent.py --variant kind=... --variant mixed=... --variant both=... --reps 30 --max-actions 1000`
- Result (reference 0.574 / 16.80 levels per rep): kind −0.029 ± 0.053 / −0.33 ± 0.75 (no effect);
  mixed **−0.064 ± 0.054** / +0.50 ± 0.60; both **−0.091 ± 0.067** / +0.07 ± 0.82. Per game (levels):
  kind s5i5 +0.67, r11l +0.33, lp85 −0.40, sc25 −0.63; mixed bp35 −0.43, sc25 +0.63, sp80 +0.23.
  Also measured first (`.claude/scratch/design/count_check.py`): every action counts on the scorecard,
  including a legal click on background and a button the game doesn't offer (5/5 in all 25 games).
- Verdict: ❌ both behaviours deleted (ablation rule); `GameMode` / `world.mode` kept (no decision uses it;
  step 2 builds on it). After the revert: 50/50 identical to the champion again.

## 2026-09-26 — Refactor: agent/brain in "Python like Java" (behaviour-identical)
- Change: owner request. Dict beliefs → typed classes (`CounterMemory`, `AvatarBelief`, `ColourRoles`, `Kind`,
  `Pattern`, `Item`, `ActionBeliefs`, `Rules`, `Goal`, `Death`, `WorkingMemory`), strings → enums (`Outcome`, `Role`),
  tuples → records (`Action`, `ClickInfo`); free functions → classes with explicit dependencies (`ScreenReader`,
  `FirstLook`, `CounterLearner`, `AvatarLearner`, `TouchLearner`, `ActionLearner`, `DeathJudge`, `ScreenFeatures`,
  `GoalTracker`, `Explorer`, `PromptWriter`). mypy added to `make check` / CI (clean). explain.py, mask_check.py updated.
- Commands: `eval/run.py --champion eval/champion/my_agent.py --reps 2 --max-actions 1000`; `make ci`;
  mock-LLM run (5 games x 2 reps, 10% failures + 10% garbage); swarm timing x2 each.
- Result: **50/50 runs identical** to the champion, 0 errors. CI: 11/11 interface, 9/9 masks, regression PASS 1.1812.
  Mock LLM: 1,709 advised actions, 0 errors. Swarm (25 threads, 25,025 actions, same score 0.682): **107.8 / 109.6 s
  vs champion 103.7 / 101.6 s → ~6% slower per action** (still ~4.3 ms/action).
- Verdict: INFO — pure restructure; the 6% is accepted for now (the brain is stuck, not time-limited), profile later.

## 2026-09-26 — GPU dev run v3: server up, but every thinking answer ran out of tokens
- Change: v2's sampler/CUDA_HOME fix. nvcc seen: system **12.8** (too old for sm_120), server 13.4 (wheelhouse).
- Result: server up after 300 s. Games 477 s, aggregate **0.6816 / 17 levels** — the same as the local
  advisor-off swarm baseline (0.682 / 17, seed 0, 1000 actions), because the LLM contributed **0 actions**:
  44 requests, 38 answered and all 38 `finish=length` at 6,144 reasoning tokens with an empty answer
  (prompt ~4.7k tokens), 6 timed out at 240 s → the advisor paused itself and the brain played the rest.
  Throughput with 25 concurrent requests: ~26 tokens/s per request, ~650 tokens/s total.
- Consequence (arithmetic, not measured play): thinking mode at >6k tokens/answer gives ~9 h × 650 tok/s
  ≈ 21M tokens ≈ a dozen answers per game over ~110 hidden games.
- Fix: thinking off by default (`enable_thinking: false`, max 1,024 tokens, 180 s), Qwen3.8's recommended
  sampling per mode (my 0.6/0.95 matched neither), everything overridable per run (`ARC3_LLM_THINKING`,
  `ARC3_LLM_MAX_TOKENS`, `ARC3_LLM_TIMEOUT_S`). dev.py now plays several arms in one session, each with a
  fresh copy of the agent module. Request bodies checked in-process; advisor-off regression PASS (1.1812).
- Verdict: INFO — v4 pushed: arm `nothink` (5,400 s) then arm `think` (16k tokens, 2,700 s).

## 2026-09-26 — GPU dev run v2: past the forward pass, died in the sampler's JIT compile
- Change: the `.pth` fix from v1.
- Result: pre-flight import check rc=0 (`cutlass` found via `/tmp/vllm-site/nvidia_cutlass_dsl/dsl_packages`).
  Weights loaded, the profiling forward pass ran (the GDN linear-attention kernels work), flashinfer autotuning ran;
  then the first sampling step JIT-compiled flashinfer's top-k/top-p kernel and failed:
  `FlashInfer requires GPUs with sm75 or higher`, preceded by `SM 12.x requires CUDA >= 12.9` — flashinfer
  judged the GPU arch through the Kaggle image's older system CUDA toolkit. ~6 GPU-minutes.
- Fix (checked in the vLLM 0.30.0 wheel source before pushing): `VLLM_USE_FLASHINFER_SAMPLER=0` makes
  `flashinfer_sampler_supported()` return False, so the model-runner-v2 sampler uses `apply_top_k_top_p` +
  `gumbel_sample` (no JIT). Also `CUDA_HOME`/`PATH` point at the wheelhouse's CUDA 13 nvcc (nvidia-cuda-nvcc
  13.4.92) for any other JIT; the report records both nvcc versions.
- Verdict: 🐛 fixed; v3 pushed. GPU quota used so far 0.22 h of 30 h.

## 2026-09-26 — GPU dev run v1 (hawkosm/arc3-llm-dev, RTX PRO 6000): server failed to start
- Change: none to the agent. First run of `notebooks/llm_dev/dev.py`.
- Command: `notebooks/llm_dev/build.py --budget-s 9000 --max-actions 1000` + `kaggle kernels push -t 14400`
- Result: GPU = RTX PRO 6000 Blackwell, 97,887 MiB, **driver 580.159.04, CUDA 13.0** (the cu130 wheelhouse is
  compatible). arc-agi + wheelhouse installs rc=0 (95 s). Weights loaded, then the engine died in its first
  forward pass: `flashinfer.gdn_prefill` → `import cutlass` → ModuleNotFoundError. Cause (verified on the
  real wheels): `nvidia-cutlass-dsl` *was* installed but exposes `cutlass` only via a `.pth` file, and `.pth`
  files are ignored in a `--target` dir used through PYTHONPATH. ~7 GPU-minutes used.
  Also seen: the competition dataset itself ships the 25 public games (`environment_files/`).
- Fix: dev.py now runs `site.addsitedir()` once to collect the `.pth` paths into the server's PYTHONPATH,
  and does a pre-flight `import flashinfer.gdn_prefill, cutlass` that aborts before loading the model.
- Verdict: 🐛 fixed; v2 pushed.

## 2026-09-26 — Kaggle LLM prep notebook (hawkosm/arc3-llm-prep v1, CPU, internet on)
- Change: none to the agent. `notebooks/llm_prep/prep.py` with the HF manifest embedded; attaches the
  public Kaggle model mirror `michaelpoluektov/qwen3-8-27b-fp8/transformers/default/1`.
- Command: `kaggle kernels push -p .claude/scratch/llm/kaggle/prep_push` → output `prep_report.json`
- Result: mirror **81/81 files match** Hugging Face `Qwen/Qwen3.8-27B-FP8@017b9c7af6` (sha256 for LFS,
  git-blob sha1 otherwise), 0 missing, 0 extra, 112 s. Wheelhouse: 198 wheels, 3.89 GB, `pip install
  --no-index --target` rc=0; import check `vllm 0.30.0, torch 2.13.0+cu130, CUDA 13.0, transformers 5.17.0`,
  arch list includes `sm_120`. Kaggle image: Python 3.12.13, glibc 2.35. Pip warned about conflicts
  with image packages (protobuf, gradio, ...) — irrelevant for a separate `--target` dir used only by
  the server process.
- Verdict: INFO — model and wheelhouse usable offline. Unknown until the GPU run: whether the RTX
  driver supports CUDA 13.0.

## 2026-09-26 — LLM advisor (agent/brain/llm.py): plumbing only, no real model yet
- Change: optional advisor, on only when `ARC3_LLM_URL` is set. Prompt = beliefs + last 16 steps +
  the 64x64 screen as hex rows; answer = `{"plan": [...]}` (≤8 legal actions, never RESET). Plan dropped on
  death / level up / an advised action that changed nothing; 3 failed requests in a row pause it 5 min.
  Optional per-request log (`ARC3_LLM_LOG`). Mock server `eval/mock_llm.py`; Kaggle notebooks
  `notebooks/llm_prep/` (CPU: mirror hash check + vLLM 0.30.0 wheelhouse) and `notebooks/llm_dev/` (GPU dev run).
- Commands:
  - advisor OFF identity: `eval/run.py --agent agent/my_agent.py --champion eval/champion/my_agent.py --reps 2 --max-actions 1000`
  - mock ON, 10% failures + 10% garbage: `eval/run.py --games ls20,vc33,ft09,tu93,sp80 --reps 2 --max-actions 300`
    and `--swarm --max-actions 300`
  - GPU notebook dry run locally against the mock: `DEV_INPUT=... DEV_LLM_URL=... dev.py` (25 threads, 150 actions)
- Result: OFF = **50/50 runs identical** to the champion, wall 560 s vs 556 s. Mock ON: 0 errors / 0 crashes in
  10 single + 25 swarm + 25 dry-run games; prompt median **6.5k chars (~2.6k tokens, rough)**, max 7.2k.
  Random mock plans score below the brain, as expected (they measure plumbing, not play).
- Verdict: INFO — plumbing ready. Play quality is unmeasured until the Kaggle GPU dev run.

## 2026-09-26 — Promoted: pruned brain becomes the local champion (owner decision; not submitted)
- Change: `eval/champion/my_agent.py` = current `agent/my_agent.py` (pruned brain + look merge);
  previous champion archived as `experiments/2026-09-25_floor_v1/my_agent.py`.
- Command: `cp agent/my_agent.py eval/champion/my_agent.py && eval/regression_check.py --update`
- Result: CI smoke baseline 1.1812 score / 1.00 levels (ls20+vc33, 3 reps, 400 actions); check PASS.
- Verdict: PROMOTED on the owner's call (tie at N=30 with better robustness). No Kaggle push/submission.

## 2026-09-26 — Merge first_look.py into look.py (behaviour-identical)
- Change: owner request — both modules read the screen; `look.py` now holds the pure reading
  helpers and `first_look` (the level inventory). Counter hiding in first_look inlined to avoid a
  look <-> learn import cycle. Build order: settings, world, look, learn, goals, decide, brain, kaggle_agent.
- Command: `eval/run.py --agent agent/my_agent.py --champion <pre-merge build> --reps 2 --max-actions 1500` + exact per-run comparison
- Result: **50/50 identical runs**, 0 swallowed errors; mask_check 9/9.
- Verdict: INFO — pure restructure.

## 2026-09-25 — Pruned agent (after ablation) vs full and vs champion
- Change: deletions listed in the ablation entry + avatar step measured on the moved piece.
- Command: `eval/run.py --agent experiments/2026-09-25_pruned/my_agent.py --champion experiments/2026-09-25_ablation/full.py --reps 30 --max-actions 2000`
  (+ paired against the champion runs of the goals eval, same seeds)
- Result: vs full **+0.019 ± 0.013 score (significant)**, +0.13 ± 0.38 levels; sp80 -0.07 (0/30,
  lost the new-position/quick-test gain), m0r0 +0.41. vs champion: -0.024 ± 0.106 score,
  **+0.87 ± 1.09 levels** (tie). 0 crashes, 0 swallowed errors. Size: 1256-line shipped file
  (champion 836), 54 settings (was 58); the growth is the owner's belief modules.
- Verdict: KEEP as the development baseline (better than the full build, smaller). Not promoted over
  the champion (tie at N=30, rule 3); owner decides whether robustness fixes (memory guard,
  background-avatar fix) justify replacing the champion anyway.

## 2026-09-25 — Ablation: what each addition since the first champion contributes
- Change: none to behaviour (switches added; all-on verified identical to the goals build, 50/50
  runs). Each variant = one addition off; 10 agents in one run, same seeds.
- Command: `eval/run.py --agent experiments/2026-09-25_ablation/full.py --variant <name>=<file> ... --reps 30 --max-actions 2000`
- Result (Δ = variant minus full; negative = worse without the addition), 7500 runs, 0 crashes:
  | addition removed | Δscore | Δlevels | verdict |
  |---|---|---|---|
  | timeout detection | -0.018 ± 0.066 | **-1.23 ± 1.02** | needed (tu93 -0.90, lp85 -0.37) |
  | expected-cost rule | -0.017 ± 0.050 | -0.70 ± 0.80 | leaning useful |
  | life-aware planning | -0.003 ± 0.080 | -0.60 ± 0.86 | leaning useful |
  | new-position chance floor | +0.000 ± 0.001 | +0.03 ± 0.07 | nothing |
  | quick tests | +0.013 ± 0.018 | +0.23 ± 0.41 | nothing |
  | counter tracks on any line | -0.000 ± 0.000 | +0.00 ± 0.09 | nothing |
  | hidden-state screen ids | +0.000 ± 0.000 | +0.07 ± 0.16 | nothing |
  | colour wall/hazard predictions | **+0.018 ± 0.009** | +0.07 ± 0.16 | harmful |
  | goal steering | +0.000 ± 0.006 | -0.03 ± 0.37 | nothing (at this budget) |
- Verdict: PRUNE — deleted new-position floor, quick tests, hidden-state ids (+ avatar lock), colour
  predictions and all switches; kept timeout detection, the expected-cost rule and life-aware
  planning (the core decision rule), any-line counter tracks (the simpler rule, masks stay
  correct), and the owner's beliefs (first look, colour roles, goals; goal steering on probation
  pending a large-budget test). Found while checking: avatar step read (0,0) whenever the avatar
  colour also sits still elsewhere (tu93) — now measured on the moved piece.

## 2026-09-25 — Goals decided by the game (feature changes on each win) — NO EFFECT at 2000 actions
- Change: `goals.py` — every screen gets generic features (colour counts, colour-colour touch
  edges, what the avatar touches); on each level win the biggest relative changes between the
  level's first and winning screen become goal hypotheses; later wins support/contradict them;
  first-look same-shape pairs are weak pre-win cues. Frontier cost divided by 1 + progress.
- Command: `eval/run.py --agent experiments/2026-09-25_goals/my_agent.py --champion eval/champion/my_agent.py --reps 30 --max-actions 2000`
- Result: vs champion -0.043 ± 0.105 score, +0.73 ± 1.06 levels (NOISE). Paired vs first-look
  (same seeds): score **-0.000 ± 0.006**, levels **+0.03 ± 0.37**. Qualitatively right: on tu93
  hypotheses from the level-0 win were supported by the next 2 wins (3/0); wrong first-look cues
  fell to 0.20. 0 crashes, 0 swallowed errors.
- Verdict: NOISE. Goals exist only after a win and only reorder exploration; at 2000 actions most
  games reach at most level 1-2, so there is little for them to steer. Needs a larger-budget test
  (Kaggle gives far more actions per game) and/or stronger use than a frontier discount.

## 2026-09-25 — First look + colour roles (owner's World design) vs champion and vs phase 2b — NOISE / tiny loss
- Change: `first_look.py` (item kinds with purpose "?", patterns), `learn_what_touching_does`
  (floor/wall/hazard/goal/pushable/collectible per colour, avatar step per key, multi-colour
  avatar parts, collectible judged on stepping off), decide predicts keys into known wall/hazard
  colours as useless/deadly; bug fix: background colour can no longer be picked as the avatar
  (wa30/ka59/re86/tr87 had it). `eval/explain.py` prints the beliefs.
- Command: `eval/run.py --agent experiments/2026-09-25_first_look/my_agent.py --champion eval/champion/my_agent.py --reps 30 --max-actions 2000`
- Result: vs champion **-0.043 ± 0.104** score, **+0.70 ± 0.97** levels (NOISE). Paired against the
  phase 2b candidate on the same seeds: score **-0.015 ± 0.012** (small but significant loss),
  levels **-0.03 ± 0.24** (none). bp35 0/29/1 vs champion; sp80 30/0, tu93 +0.53 levels kept.
  0 crashes, 0 swallowed errors. Beliefs verified readable/correct on ls20 (walls = colour 4,
  avatar 12 + part 9, step 5) and wa30 (avatar 14 + part 0, step 4, a push-the-boxes layout).
- Verdict: NOISE — not promoted. Knowing what things ARE does not yet change what the agent
  DOES beyond wall avoidance; the payoff has to come from goals built on these beliefs.

## 2026-09-25 — Restructure agent/core/ -> agent/brain/ (World of beliefs + fixed functions), behaviour-identical
- Change: owner's design — `world.py` (belief dicts: counters, avatar, actions, rules, levels,
  goals, deaths), `look.py`/`learn.py`/`decide.py` (fixed functions), `brain.py` (the loop),
  `kaggle_agent.py` (framework adapter); clearer names throughout (settings renamed, see
  settings.py). Removed code for rejected/unused features (replay, novelty bonus; both off).
- Command: `eval/run.py --agent agent/my_agent.py --champion <pre-refactor build> --reps 2 --max-actions 1500`
  then a per-run exact comparison of (levels, per-level actions, total actions).
- Result: first attempt 21/50 identical — a missed rename (`arrived_by`) raised in every
  choose_action and the MyAgent safety net silently played randomly (296/300 steps). Fixed:
  **50/50 identical, 0 swallowed errors**. `eval/regression_check.py` now fails on any swallowed
  choose_action exception. mask_check 9/9, build check OK.
- Verdict: INFO — pure restructure. Smoke regression still FAILs vs the champion baseline
  (1.1857 < 1.2847) because agent/brain carries the unpromoted phase 2 changes (N=30 all-games:
  NOISE, -0.028 ± 0.107). Owner decision pending: re-baseline or revert.

## 2026-09-25 — Phase 2b (2a + any-line tick masking, adaptive occlusion keys, speedups) vs champion — NOISE
- Change: counter-track masking on any line with >=3 progressing tick positions (eval/mask_check.py
  9/9); conflict-triggered occlusion underlay in state keys (g50t conflicts 153 -> ~44, fallbacks
  878 -> 0; always-on version broke tu93 3 -> 0 levels on 3/3 seeds, hence adaptive); tracker
  counters + periodic refit (~5.3 -> 3.9 ms/action on re86).
- Command: `eval/run.py --agent experiments/2026-09-25_phase2b/my_agent.py --champion eval/champion/my_agent.py --reps 30 --max-actions 2000`
- Result: 0.601 ± 0.072 vs 0.628 ± 0.106; paired **-0.028 ± 0.107**; total levels 20.77 vs 20.03,
  paired **+0.73 ± 0.92**. sp80 +0.08 (30/0/0), tu93 +0.53 levels, m0r0 +0.20 levels, sk48 first
  levels (3 reps); lp85 -0.78 ± 2.27 and vc33 -0.51 ± 0.80 dominate the score noise. 0 crashes.
- Verdict: NOISE — not promoted. Two rounds of fixes to the screen-graph explorer moved levels
  by < 1/rep; supports pivoting to a belief-state (objects/map/rules/goals) architecture.

## 2026-09-25 — Phase 2a (timeouts, expected-cost frontier, life-aware planning, position novelty, local probe) vs champion — NOISE
- Change: `_is_timeout` (deaths at a repeated life length = step-budget timeouts, un-recorded);
  frontier cost `(d+1)/p` replacing `d + 10(1-p)`; frontiers beyond the life budget skipped,
  voluntary RESET (life > 0 only) when nothing reachable remains; `P_NOVEL_POSITION` floor for
  (avatar position, action) never tried; local probe of usually-stationary simple actions.
  Motivation (dev diag, single seeds): sp80's 4-step solution `4,4,4,5` took 938 actions because
  ACTION5's global no-op rate demoted it everywhere; 44/44 sp80 "deaths" were timeouts.
- Command: `eval/run.py --agent experiments/2026-09-25_phase2a/my_agent.py --champion eval/champion/my_agent.py --reps 30 --max-actions 2000`
- Result: 0.658 ± 0.128 vs 0.628 ± 0.106; paired **+0.030 ± 0.136**; total levels 20.93 vs
  20.03, paired +0.90 ± 1.03. Per game: sp80 +0.12 (30/0/0), tu93 +0.63 levels (20/10), sc25
  +0.26; lp85 ±3.21 CI dominates the aggregate noise. 0 crashes.
- Verdict: NOISE — not promoted. Targeted gains real but small; carried into phase2b.

## 2026-09-25 — First competition submission: floor champion (kernel v2)
- Change: none — `kaggle competitions submit ... -k hawkosm/arc-agi-3-agent -v 2` (run by owner)
- Result: ref 56548792, PENDING at 11:00 UTC. CLI: "0 submissions remaining today" → rules §2
  confirmed **1 submission per day** (not 5 as the starter README says).
- Result (COMPLETE): public LB **0.44** (local N=30 on the 25 public games: 0.628 ± 0.106 @2000
  actions; Kaggle ran until the 7.5 h time budget with no action cap).
- Verdict: INFO — first calibration point: hidden-set score ≈ 0.70 × local public-set score, even
  with a far larger action budget on Kaggle. Local gains must be large to be visible on the LB.

## 2026-09-25 — Parity fix: Kaggle runtime numpy is 2.0.2, not the 2.4.4 wheel
- Change: Makefile pin numpy==2.4.4 → 2.0.2. Kernel v2 log shows pip keeping the image's
  preinstalled numpy 2.0.2 ("Requirement already satisfied ... (2.0.2)").
- Also: kernel v2 (floor champion) Phase A COMPLETE, clean log, dummy submission.parquet.
- Command: `make setup && make interface && eval/regression_check.py && eval/run.py --reps 3 --max-actions 1000`
- Result: interface 11/11; regression PASS with an exact match (1.2847 / 1.17 levels); all-games
  sanity 0.577 ± 0.264 (n=3), consistent with the N=30 0.628 ± 0.106 at 2000 actions.
- Verdict: INFO — no behaviour change from the numpy version; parity restored.

## 2026-09-25 — Kaggle Phase A: random baseline kernel v1
- Change: none — `make submit AGENT=experiments/2026-09-25_random_baseline/my_agent.py` after the
  owner accepted the rules (CLI now `userHasEntered: True`)
- Result: hawkosm/arc-agi-3-agent v1 → `KernelWorkerStatus.COMPLETE`; output has the expected dummy
  `submission.parquet` (1 row) + log. Kaggle's ARC-AGI-3-Agents agent.py/swarm.py/main.py
  downloaded and diffed: byte-identical to commit 135f20a (the local pin).
- Verdict: INFO — Phase A pipeline OK. Phase B (competition submission) awaiting owner go-ahead.

## 2026-09-25 — Floor through the real notebook, competition-mode gateway, all 25 games
- Change: none — champion (agent/my_agent.py == eval/champion/my_agent.py) after re-pinning to
  arc-agi 0.9.8 / numpy 2.4.4 (later corrected to 2.0.2, see next entry) / framework 135f20a (Kaggle's dataset files verified byte-identical
  to 135f20a after rules acceptance)
- Command: `eval/kaggle_sim.py --agent agent/my_agent.py --games all --max-actions 500`
- Result: runs end to end (25 threads, HTTP, competition mode), 0 errors in logs.log, 119 s,
  aggregate **0.407** at a 500-action cap (single run; not comparable to the 2000-action N=30).
  `make interface` 11/11 on the pinned versions; `regression_check.py` PASS (baseline written).
- Verdict: INFO — the submission artifact works under Kaggle-like execution.

## 2026-09-25 — PROMOTED: floor agent vs random starter, all 25 games, N=30 paired
- Change: floor agent (agent/core/, incl. click lattice fix) vs experiments/2026-09-25_random_baseline
- Command: `eval/run.py --agent agent/my_agent.py --champion experiments/2026-09-25_random_baseline/my_agent.py --reps 30 --max-actions 2000`
- Result: floor **0.628 ± 0.106** vs random **0.063 ± 0.030**; paired **+0.565 ± 0.110** (n=30,
  SIGNIFICANT); 0/1500 crashes. Wins on 16 games (30/0 on ar25, bp35, lf52, lp85, ls20, s5i5,
  su15, tn36, vc33). Biggest: lp85 +5.18 (3.0 levels, max 5), tn36 +3.56, vc33 +2.39, sc25 +1.38.
  **Regression: sp80 −1.18 ± 0.62** (2/28/0) — random clears L1 in ~242 actions, floor in ~1214.
  0 levels on 9 games (cn04, dc22, g50t, ka59, re86, sb26, sk48, tr87, wa30); ft09 0.5 levels.
  (Ran on arc-agi 0.9.9 before the pin; 0.9.8 differs only in an error message.)
- Verdict: PROMOTED — first champion; frozen to eval/champion/my_agent.py, CI baseline written.
  sp80 regression logged in IDEAS_TRIED (likely click-target dilution in a mixed game).

## 2026-09-25 — Replay variant under a competition-mode gateway (kaggle_sim)
- Change: none — measurement of experiments/2026-09-25_replay/my_agent.py (rebuilt with the
  `ARC3_MAX_ACTIONS` override) through the real notebook
- Command: `eval/kaggle_sim.py --agent experiments/2026-09-25_replay/my_agent.py --games vc33,tu93,lp85,m0r0 --max-actions 600 [--no-competition]`
- Result: competition_mode=True **1.364** aggregate, every game `plays=1` (full resets
  intercepted); competition_mode=False **9.757**, plays 2–4 (vc33 L1 3 actions, lp85 4 levels at
  5/41/18/60 actions). Single run each.
- Verdict: REJECTED for Kaggle unless Kaggle's gateway is shown to run non-competition mode. The
  lever is real only under the local non-competition scorer. Do not pursue further without a
  Kaggle-side observation.

## 2026-09-25 — Kaggle emulator works; competition-mode gateway (random baseline)
- Change: new `eval/kaggle_sim.py` — builds the real notebook, runs its cells against a local
  arc_agi gateway with `competition_mode=True`, parses the framework's FINAL SCORECARD.
- Command: `.venv/bin/python eval/kaggle_sim.py --agent experiments/2026-09-25_random_baseline/my_agent.py --games ls20,vc33`
- Result: runs end to end in 9 s; scorecard reports `competition_mode: true`; ls20/vc33 81 actions
  each (MAX_ACTIONS 80 + framework off-by-one), score 0.
- Verdict: INFO — the notebook artifact itself (not just the agent file) is now testable locally.

## 2026-09-25 — Replay-after-level-up variant (max-over-plays), local scorer
- Change: `REPLAY_ENABLED=True` (experiments/2026-09-25_replay/my_agent.py)
- Command: `eval/run.py --agent experiments/2026-09-25_replay/my_agent.py --champion agent/my_agent.py --games vc33,lp85,tu93,tn36,bp35,m0r0 --reps 2 --max-actions 2000`
- Result: 6.940 ± 4.105 vs 3.082 ± 2.945; paired +3.858 ± 1.160 (n=2). vc33 L1 in 3 actions
  (baseline 7), tu93 L2 in 10 (baseline 16). 0 plan mismatches during replays.
- Verdict: ⚠️ scorer-dependent. Source reading afterwards: arc_agi's server in competition mode
  intercepts the level-start RESET that triggers the full reset → expected dead on Kaggle. n=2
  and 6 games; not a promotion candidate.

## 2026-09-25 — Click lattice + exhausted-state refinement
- Change: `perception.components()` rewritten (row runs + union-find, ~2 ms vs ~11 ms, 0/300
  mismatches vs reference BFS); lattice click points in components > 24 cells; stride 4→2→1
  whole-grid refinement when no untried action is reachable.
- Command: `eval/run.py --games ft09,lp85,vc33,r11l,s5i5,tn36,su15 --reps 2 --max-actions 2000`
- Result: lp85 1.0 → 2.5 levels (max 4); tn36 0 → 1 level (11 actions, 3.57); ft09 still 0 but now
  explores (153 states vs 1). Diagnosis probe (`click_scan.py ft09`): 72/180 colour-9 lattice
  clicks are effective; the centroid click of the connected tile block is not.
- Verdict: 🐛 fix kept; n=2 dev run, covered by the N=30 run below.

## 2026-09-25 — Floor budget scaling 1000 → 5000 actions (pre-lattice floor)
- Command: `eval/run.py --reps 3 --max-actions 5000` vs `--reps 2 --max-actions 1000`
- Result: 0.415 ± 0.307 (5000) vs 0.546 ± 0.290 (1000) — within noise. Levels barely move.
  Diagnostics: ft09 1 state / lp85 6 states with ~4950 random fallbacks (click stall); g50t 615
  graph conflicts (hidden state); re86/tr87/su15 ~3000 states, 0 new levels (state explosion);
  bp35/sb26 ~38 ms/action (half engine, half `components()`).
- Verdict: INFO — the agent is stuck, not budget-limited. Motivated the click fixes above and
  ROADMAP Phase 2.

## 2026-09-25 — Floor agent first light (smoke + all games, dev)
- Change: `agent/core/` floor agent (masked state graph, learned action model, avatar tracker,
  verified BFS navigation), flattened by `scripts/build_submission.py`.
- Command: `eval/run.py --games all --reps 2 --max-actions 1000`; `eval/run.py --swarm --max-actions 300`
- Result: 0.546 ± 0.290 (n=2) vs random 0.063 at the same budget; ≥1 level on 17/25 games; 0
  crashes. Swarm (25 threads, one process): 0 crashes.
- Verdict: INFO — dev numbers, superseded by the N=30 paired run.

## 2026-09-25 — Random baseline, all 25 games, N=30
- Change: none — starter-kit random agent (experiments/2026-09-25_random_baseline/)
- Command: `eval/run.py --agent experiments/2026-09-25_random_baseline/my_agent.py --reps 30 --max-actions 1000`
- Result: **0.063 ± 0.030** (n=30). sp80 1.19 ± 0.62 (level 1 every run), cd82 0.34, rest ≈0.
  0 crashes. 750 runs in 3m14s on 20 workers.
- Verdict: INFO — the floor for "beats random".

## 2026-09-25 — Interface facts (eval/interface_checks.py)
- Command: `.venv/bin/python eval/interface_checks.py`
- Result: 11/11 match. Bare `GameAction(6)` → ValueError (spec's claim confirmed; my first
  expectation was wrong); ACTION6 x=64 → ValidationError; illegal ACTION6 on ls20 counts, no
  change; RESET after an action = level reset, RESET with 0 level actions = full reset; initial
  frames deterministic; ACTION6 data shared across threads (both threads read x=50); RESET frames
  can have 2+ layers; post-GAME_OVER actions return empty frames.
- Verdict: INFO — see AGENT.md "Verified interface facts".

## 2026-09-25 — Pipeline: make setup + make verify-local (random starter)
- Command: `make setup`, `make list-games`, `make verify-local`
- Result: 25 environments; verify-local exit 0, ls20/vc33 51 actions each, score 0.0.
- Verdict: INFO — local pipeline OK. Kaggle half (Phase A/B) blocked: rules not accepted
  (`userHasEntered: False`, 403 on submissions/data).

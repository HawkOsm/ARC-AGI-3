# Ideas Tried

Concept-indexed ledger of strategic ideas, so nothing gets re-tested under a new name.
[`LOG.md`](LOG.md) has the chronological detail (command, raw numbers); this file is the
"have we already tried this?" lookup — one row per **idea**, updated when it gets a verdict.

Legend: ✅ worked (kept) · ❌ regressed / rejected · ➖ null result · 🐛 bug fix · ⏳ pending (needs
Kaggle or more data) · ⚠️ scorer-dependent

---

## Ideas to test next (collected; none run yet)

| idea | why | how to test | setting / status |
|---|---|---|---|
| **More thinking per game** — the run was starved: 87% of each call was queue wait (124 of 143 s); the model weights (81.8 GB) left a 5 GB cache = 105k tokens, so only ~5 of 25 games could think at once; each game got ~53 model calls in 2.2 h and 24/25 ran out of time | run A analysis (LOG 2026-09-27) | (a) shorter conversations per call (avg prompt 20k tokens; a 10–12k budget ≈ 2× games at once); (b) the smaller Qwen3.8-27B-FP8 (~27 GB weights → ~60 GB cache, all games at once, weaker per call); (c) prefix cache on (it was off: 27.3M prompt tokens re-read) | `AGENT_CONTEXT`; model choice; server flags |
| Fewer wasted actions: 314 of 3900 actions (8%) changed nothing on the board (ar25: 169 of 409) | run A analysis | tell the model / refuse repeating an action that just did nothing on the same screen | not built |
| Fewer game-overs mid-experiment: 34 in total (sp80 11, tn36 7, tu93 6); sp80's notes blame "mid-sequence restarts" for corrupted measurements | run A analysis | say when the step counter is nearly out; tell the model plainly that a restart happened and what it undid | not built |
| More actions per call: only ~1/3 of actions ran in batches | run A analysis | prompt: plan a sequence and batch it once confident | prompt |
| Screen in each message as letters instead of an image (owner) | text grids may be easier for an LLM than pictures (one unverified paper: 80–86% vs 16–34%); the Duck sends both | same games and budget, `AGENT_SCREEN=text` vs `image` (also `both`, `none`) | `AGENT_SCREEN`, ready |

## State identity

| idea | verdict | evidence | notes |
|---|---|---|---|
| Raw-frame hashing | ❌ | probe 2026-09-25 | Every game has a per-action HUD counter → every frame novel; graph never closes. |
| Per-cell change-frequency mask only | ❌ (by construction) | probe 2026-09-25 | Counter is monotone: each counter cell changes once, so frequency never crosses a threshold. |
| Border-tick mask (small border diff components → mask whole border line) | ✅ | LOG 2026-09-25 floor | Mask stabilises within ~2–5 changes/game (`mask_changes`). Re-key graph instead of dropping it. |

## Exploration

| idea | verdict | evidence | notes |
|---|---|---|---|
| Graph exploration: untried action with best learned p, else cheapest frontier (d + λ(1−p)) | ✅ floor | LOG 2026-09-25 PROMOTED | 0.628 ± 0.106 vs random 0.063 (N=30, paired +0.565 ± 0.110). Open regression: sp80 −1.18 (random finds L1 5× faster — suspect 96 click targets/state diluting keyboard exploration in a mixed game). |
| One click target per colour component (centroid-nearest cell) | ❌ | LOG 2026-09-25 click lattice | ft09/lp85 stall: the clickable unit is a sub-part of a connected component. |
| Lattice click points inside large components + whole-grid refinement (stride 4→2→1) when a state is exhausted | ✅ | LOG 2026-09-25 click lattice | lp85 1.0→2.5 levels, tn36 0→1. ft09 still 0 (combinatorial). |
| More action budget (1000 → 5000) with the pre-lattice floor | ➖ | LOG 2026-09-25 budget | 0.546 → 0.415, within noise: the agent is stuck, not budget-limited. |
| "Free" probing with no-op / unavailable actions (owner's hypothesis) | ❌ (by measurement) | LOG 2026-09-26 count check | Every action counts on the scorecard, including background clicks and buttons the game doesn't offer (5/5 in all 25 games). |
| Click chance learned per item kind (colour + shape) instead of per colour — "one click per kind before twins" (PLAYABLES step 1) | ➖ removed | LOG 2026-09-26 step 1, N=30 | Δscore −0.029 ± 0.053, Δlevels −0.33 ± 0.75. Splits evidence thinner than the per-colour tally. |
| Mixed games (keys + clicks): halve click priors so keys go first (PLAYABLES step 1) | ❌ removed | LOG 2026-09-26 step 1, N=30 | Δscore **−0.064 ± 0.054** (levels +0.50 ± 0.60, n.s.): the tallies already learn fast; the prior only delays needed clicks (bp35 −0.43 levels). Both together −0.091 ± 0.067. |

## Beliefs about the game (owner's World design, 2026-09-25)

| idea | verdict | evidence | notes |
|---|---|---|---|
| Avatar = colour whose changed-cell shift depends on the action | 🐛 fixed | explain.py wa30 | The **background** mirrors every move and scored as the avatar (wa30 colour 1; ka59/re86/tr87 colour 0 in earlier diags). Colours covering > AVATAR_MAX_CELLS are now excluded. |
| Multi-colour avatar: colours co-moving ≥ AVATAR_PART_RATE of moves are avatar parts | ✅ (dev) | explain.py ls20/wa30 | Without it the avatar's own body colour was learned as "pushable" (ls20 colour 9, 711 evidence). 0.8 missed wa30's top edge (~73%) → 0.6. |
| Colour roles from touch (floor/wall/hazard/goal/pushable/collectible) + predicting blocked/deadly keys by colour | ➖ | LOG 2026-09-25 first look | N=30: levels unchanged vs phase 2b (-0.03 ± 0.24), score -0.015 ± 0.012. Beliefs are right; using them only for wall avoidance doesn't move levels. | ls20: colour 4 = wall, 3 = floor after 800 actions; ambiguous colours stay "?" (needs ≥60% share). |
| Collectible = "something vanished when the avatar covered it" | ❌ (by construction) | explain.py ls20 | Every floor colour looked collectible (covering hides it). Now judged when the avatar steps OFF a cell. |
| Goals decided by the game: feature changes (counts, touches, avatar touches) between level start and win; frontier discount by progress | ➖ | LOG 2026-09-25 goals | N=30 @2000: +0.03 ± 0.37 levels vs first-look. Hypotheses are sensible (tu93 3/0 support) but barely used at this budget. |
| Game view scrolls (camera) → map in world coordinates | ➖ not needed yet | scroll_probe, 25 games × 300 moves | 0 scrolling transitions in every public game; engine supports cameras, so keep a detector on the roadmap. |

## Ablation 2026-09-25 (additions since the first champion, one off at a time, N=30)

| idea | verdict | Δlevels when removed | notes |
|---|---|---|---|
| Timeout deaths not blamed on the last action | ✅ kept | -1.23 ± 1.02 | the only individually significant addition |
| Expected-cost frontier `(d+1)/chance` | ✅ kept | -0.70 ± 0.80 | leaning useful; core decision rule |
| Life-budget-aware planning + voluntary reset | ✅ kept | -0.60 ± 0.86 | leaning useful; core decision rule |
| Chance floor for actions new at this avatar position | ❌ deleted | +0.03 ± 0.07 | fixed sp80 alone (dev) but zero effect with everything else on |
| Quick tests of usually-stationary keys | ❌ deleted | +0.23 ± 0.41 | |
| Hidden-state (under-avatar) screen ids | ❌ deleted | +0.07 ± 0.16 | g50t contradictions fell 153 -> 44 but no levels |
| Predict keys into wall/hazard colours | ❌ deleted | score +0.018 ± 0.009 without it | harmful |
| Goal steering (frontier discount by progress) | ⏳ probation | -0.03 ± 0.37 | needs a large-budget test |

## Scoring levers

| idea | verdict | evidence | notes |
|---|---|---|---|
| Replay after level-up (full reset, replay known shortest solutions, explore on) | ⚠️❌ | LOG 2026-09-25 replay + kaggle_sim | Local non-competition scorer: +3.86 ± 1.16 (6 games, n=2); kaggle_sim 9.76 vs 1.36. Under a **competition-mode** gateway the level-start RESET is intercepted → plays=1, gain gone. Kaggle's gateway almost certainly runs competition mode. Rejected unless a Kaggle-side observation says otherwise. |

## Local LLM advisor (Phase 3)

| idea | verdict | evidence | notes |
|---|---|---|---|
| Qwen thinking mode as the advisor | ❌ | LOG v3, v4 | 27B: 9/10 answers never finish even at 16,384 tokens (685 s median). |
| LLM drives (stateless prompt: beliefs + 16 steps + hex screen), brain only on solved levels / failures | ❌ | LOG v5 | 27B with reliable JSON answers: 1 level vs brain 17 at equal actions; 4B locally 2 vs 6. Plans mostly one key repeated. |
| Structured JSON output (vLLM json_schema, compact) | ✅ plumbing | LOG local + v5 | 3/3 → 0% prose failures; 9,292/9,292 clean answers on Kaggle. |

## Who decides: brain vs LLM (2026-09-26/27)

| idea | verdict | evidence | notes |
|---|---|---|---|
| LLM picks every action from a hex screen (drive mode) | ❌ | LOG 2026-09-26 (Kaggle v5, local 4B) | Worse than the brain alone (4B: 2 vs 6 levels). |
| Brain in charge; LLM explains objects and picks a goal from the brain's shape list; brain steers (hypothesis mode, step 2) | ✅ then superseded | LOG 2026-09-26 | 0.697 / 10–12 levels over 3 runs vs brain 0.251 / 6. |
| + control modes, level lessons, surprises → LLM, several targets (step 3); own goals, click tests by object (4/4b); event-based asking (5/5b/5c) | ➖ | LOG 2026-09-26 | None beat step 2 (0.38–0.62). The LLM's own clicks cost more than they found. |
| Brain findings as `brain_hints` for the Duck's model | ⏳ dropped | LOG 2026-09-27 | Built and verified in the sandbox; Kaggle A/B not run; brain deleted (owner: LLM first). |
| **LLM writes Python that plays (the Duck, Tufa Labs)** | ✅ | LOG 2026-09-27 run A | Public notebook unchanged: 9.81 / 38 levels on the 25 public games. Our rewrite: `agent/duck.py`. |


# Ideas Tried

Concept-indexed ledger of strategic ideas, so nothing gets re-tested under a new name.
[`LOG.md`](LOG.md) has the chronological detail (command, raw numbers); this file is the
"have we already tried this?" lookup — one row per **idea**, updated when it gets a verdict.

Legend: ✅ worked (kept) · ❌ regressed / rejected · ➖ null result · 🐛 bug fix · ⏳ pending (needs
Kaggle or more data) · ⚠️ scorer-dependent

---

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

## Scoring levers

| idea | verdict | evidence | notes |
|---|---|---|---|
| Replay after level-up (full reset, replay known shortest solutions, explore on) | ⚠️❌ | LOG 2026-09-25 replay + kaggle_sim | Local non-competition scorer: +3.86 ± 1.16 (6 games, n=2); kaggle_sim 9.76 vs 1.36. Under a **competition-mode** gateway the level-start RESET is intercepted → plays=1, gain gone. Kaggle's gateway almost certainly runs competition mode. Rejected unless a Kaggle-side observation says otherwise. |

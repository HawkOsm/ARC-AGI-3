# Game Guide

Mechanics observed per public game. **Local knowledge only**: the scored set is ~4× larger and hidden,
so nothing here may be hard-coded into the agent (no per-game branches keyed on `game_id`). Use it
to understand failure modes and to design *general* mechanisms, then check they generalise.

Sources: `.claude/scratch/probe/game_meta.py` (levels, baselines, actions),
`.claude/scratch/probe/action_effects.py` (per-action diffs from the level-1 start),
`.claude/scratch/probe/click_scan.py` (which clicks do anything), and floor-agent diagnostics
(`eval/results/*/runs.jsonl` → `agent_stats`). Update the row when a game is investigated.

## Cross-game facts (all verified 2026-09-25)

- **Every game has a HUD counter** that changes 1–3 border pixels on *every* action (including
  no-ops): bottom row y=63 (cd82, dc22, g50t, ka59, s5i5, su15, tu93, bp35), top row y=0 (lf52,
  sp80, vc33, sc25), y=1 (tn36), column x=63 (ar25), x=0 (r11l), y=53 right edge (sb26, sk48), two
  rows y=61–62 (ls20), opposite corners (m0r0). It is monotone within a level, so per-cell
  change-frequency masking cannot catch it → `VolatileMask` border-tick rule.
- The counter is an **action budget per life**: when it runs out → `GAME_OVER` (ft09 dies every
  ~180 actions under exploration). A RESET restores it but restarts the level.
- **Animations**: some actions return many frame layers (sp80 A5: 22, sb26 A5: 42, bp35 A4: 25,
  cd82 A5: 15). The last layer is the settled state.
- **Clicks mostly do nothing**: in click games most pixels are inert. The clickable unit is not
  always a colour component (ft09: a connected block of colour-9 6×6 tiles only reacts on some
  cells → lattice click targets).
- 7/25 games are click-only (`available=[6]`), 5 keyboard-only, 13 mixed.

## Per-game table

`lv` = levels; `base` = human baseline actions summed over levels; counter = where the HUD tick
lives. "Floor" = floor agent @ 5000 actions, 3 reps (dev run 2026-09-25, before click lattice).

| game | tags | avail | lv | base | counter | observed mechanics | floor: levels / avatar colour |
|---|---|---|---|---|---|---|---|
| ar25 | keyboard_click | 1-7 | 8 | 748 | col x=63 | A1-A4 move a 109-cell object (3 px/step); A5 only ticks; A7 no-op at start | 1.3 / 11 |
| bp35 | keyboard_click | 3,4,6,7 | 9 | 651 | row y=63 | A4 causes 25-layer animations; slow engine (~10 ms/step) | 1.0 / 5 |
| cd82 | keyboard_click | 1-6 | 6 | 171 | row y=63 | A3/A4 move large object once then block; A5 15-layer animation | 1.7 / 9 |
| cn04 | keyboard_click | 1-6 | 6 | 789 | row y=0 (x≈16-17) | A1-A4 move a ~144-cell piece; A5 rotates/transforms it | 0 / 8 |
| dc22 | keyboard_click | 1-4,6 | 6 | 1228 | row y=63 | small 9-cell mover; most second presses no-op | 0 / 2 |
| ft09 | (none) | 6 | 6 | 208 | — | colour-9 6×6 tiles toggle on click (72/180 lattice clicks effective); combinatorial | 0 / – |
| g50t | keyboard | 1-5 | 7 | 879 | row y=63 | A2/A4 move 48-cell object with 7-layer animation; A1/A3/A5 no-op at start; 615 graph conflicts (hidden state) | 0 / 9 |
| ka59 | keyboard_click | 1-4,6 | 7 | 730 | row y=63 | 18-19-cell mover, 3 px/step | 0 / 0 |
| lf52 | click | 1-4,6,7 | 10 | 1339 | row y=0 | start frame has 2 layers; A1-A4 no-op at start | 1.7 / 11 |
| lp85 | click | 6 | 8 | 388 | — | clicks on colours 8/14 and specific colour-4 edge cells rotate a 288-cell region | 1.0 (2.5 after lattice fix) |
| ls20 | keyboard | 1-4 | 7 | 776 | rows y=61-62 | 52-cell avatar moves 5 px/step; A2 blocked at start | 1.0 / 3 |
| m0r0 | keyboard_click | 1-6 | 6 | 1107 | corners (0,0)/(63,63) | 100-cell object moves; symmetric counter | 1.0 / 5 |
| r11l | click | 6 | 6 | 233 | col x=0 | clicks move ~100-cell objects | 1.0 / – |
| re86 | keyboard_click | 1-5 | 8 | 1255 | not located | 60-cell mover 3 px/step; A5 small toggle; 3295 states, 0 levels | 0 / 0 |
| s5i5 | click | 6 | 8 | 638 | row y=63 | | 1.0 / – |
| sb26 | keyboard_click | 5,6,7 | 8 | 213 | y=53 right edge | A5 = 42-layer animation; A7 no-op | 0 / – |
| sc25 | keyboard_click | 1-4,6 | 6 | 350 | top-right corner | first action = 22-layer intro animation | 2.0 / 2 |
| sk48 | keyboard_click | 1-4,6,7 | 8 | 1070 | y=53 right edge | A1 moves 96-cell object; A2 no-op | 0 / 8 |
| sp80 | keyboard_click | 1-6 | 6 | 518 | row y=0 | random clears level 1 every time | 1.0 / 9 |
| su15 | click | 6,7 | 9 | 361 | row y=63 | 3128 states explored for level 2 | 1.0 / – |
| tn36 | click | 6 | 7 | 317 | row y=1 | level 1 in 11 actions after lattice fix | 0 (1.0 after lattice fix) |
| tr87 | keyboard | 1-4 | 6 | 414 | row y=63 | small mover; 2779 states, 0 levels | 0 / 0 |
| tu93 | keyboard_click | 1-4 | 9 | 462 | row y=63 | A1-A3 only tick at start; A4 8-layer move | 2.0-4 / 9 |
| vc33 | click | 6 | 7 | 447 | row y=0 | level 1 solvable in 3 actions (baseline 7) | 2.0 / – |
| wa30 | keyboard | 1-5 | 9 | 1843 | – | | 0 / 1 |

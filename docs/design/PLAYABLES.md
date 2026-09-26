# Design: available actions set the focus; things are playable, indicators or scenery

Owner's idea (2026-09-26), written down so the code can follow it. Status: **step 1 measured and rejected
(N=30: per-kind clicks no effect, keys-first in mixed games −0.064 score); `GameMode` kept. Steps 2-4 not built.**

## 1. The idea

The buttons a game offers say what kind of game it is, and that decides what to look at and what
"shortest path" means. A game with arrow keys is about *what moves and what blocks it*; a game with
only clicking is about *which objects react to a click and how*. Everything on screen falls into one
of three classes, and the classes are learned by watching what each action changes.

## 2. A measured fact this design must respect

Every action is counted by the scorecard, whether or not it does anything
(`.claude/scratch/design/count_check.py`, all 25 public games, 5 tries each):

- a legal click on background: screen unchanged (0/5 in ft09, lp85, sb26, sc25, sk48, bp35, ar25), **5 actions counted**
- a button the game does not offer: often ignored by the game itself (ls20, tr87, tu93, wa30, re86: even
  the step bar did not move), **5 actions counted**

So there is no free exploration: a click on scenery costs exactly what a real move costs. A movement key
is almost always a real move; a click is only worth it on something believed to be playable.

## 3. Three classes of things (per item kind = colour + exact shape)

| class | what it is | evidence that makes something this class |
|---|---|---|
| **playable** | what actions act on: the avatar, pushed blocks, clickable tiles, switches | it moves when a key is pressed, or it changes when *it* is clicked (the change is at the clicked cells) |
| **indicator** | a *product* of what playables do: step bars, lives, a lamp that lights up, a selection frame, a colour change that marks "selected" | it changes when you act, but not where you acted: on almost every action (a counter), or together with a playable change elsewhere |
| **scenery** | background, walls, textures, decoration | it never changed (a wall is scenery that blocks: `colour_roles` "wall") |
| **?** | not enough evidence yet | the starting state of every kind |

Rules of thumb that come with the classes:

- **A new kind in a new level has a purpose.** The first look compares this level's kinds with all earlier
  levels; new kinds are tried first and named to the LLM.
- **One click per kind before repeats.** If clicking one tile of a kind did nothing, its twins probably do
  nothing too; if it did something, its twins are the next best clicks.
- A kind whose click changed only *itself* (colour swap) is probably a toggle or a selection; a kind whose
  click changed something *else* is a switch; the thing that changed elsewhere is its target or an indicator.

## 4. Available actions → game mode → what the planner optimises

| offered | mode | the state that matters | explore first | "shortest path" is over |
|---|---|---|---|---|
| 1-4 (± 5, 7) | **mover** | positions of playables (avatar, pushables) | each key once per new position; learn what blocks | avatar positions / object configurations (the screen map we already have) |
| 6 only (± 7) | **clicker** | state (colour/shape) of each playable | one click per ? kind, never on scenery | which objects to change, in which order (often order-free) |
| 1-4 and 6 | **mixed** | what is selected + positions | clicks on ? / playable kinds (selection), then keys | (selection, positions) |
| 5 present | **use** | what the avatar is on / next to | ACTION5 next to new kinds | as mover, plus where to use |

The mode is read from the first frame and re-read if the offered actions change.

## 5. What already exists (don't rebuild it)

- indicators, partly: `world.counters` learns and ignores HUD counters and blinkers (a special case of
  "changes on every action, not where you acted")
- playables, partly: `world.avatar` (+ `parts`), `learn_what_touching_does` (wall / hazard / goal / pushable per colour)
- the per-level inventory: `look.first_look` → `world.kinds` with purpose "?"
- click priors per object size, and `world.actions["click_colour_changed_screen"]` (per colour, not per kind)

## 6. What would change in code (small, one step at a time)

1. **Mode + one click per kind** (`brain.py`, `decide.candidate_actions`): the mode from the offered actions;
   clicks ranked so every ? kind gets one representative click before any twin; scenery kinds get a
   near-zero prior. Cheapest step and the most directly measurable (click games).
2. **Classes** (`learn.py`): after each step, attribute every changed blob to the kind that was there and
   to *how* it changed (at the clicked cells / with the avatar / elsewhere) → `world.kinds[k]["class"]` with
   evidence counts. `counters` stays as the indicator special case.
3. **New kinds first** (`look.first_look`): mark kinds not seen in earlier levels; boost their click/touch priors.
4. **The LLM sees objects, not only pixels** (`llm.describe`): a short table of kinds with class, colour,
   size, position and what each action did to them — where the model's pattern sense (your "house shape")
   is most useful.

## 7. How we will know

Each step is its own paired run vs the champion (N≥30, AGENT.md rule 3) and is deleted if it measures
nothing (the ablation rule). The click games (ft09, lp85, r11l, s5i5, sb26, su15, tn36, vc33) are where
step 1 should show up first, as actions per completed level.

"""Regression check for counter masking (learn.update_counters): after 400 actions, each public game's fully-masked
lines must equal its HUD counter track (docs/GAME_GUIDE.md).

Per-game expectations are fine HERE (a test), never in the agent. Motivation: the
line-masking rule was rewritten three times in one session (2026-09-25) and two of the
intermediate versions silently masked most of the board on some games (ls20: 2980 cells;
m0r0: everything) — invisible in aggregate scores until levels quietly vanished.

    .venv/bin/python eval/mask_check.py            # all games with an expectation
    .venv/bin/python eval/mask_check.py ls20,vc33
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

# game -> (fully masked rows, fully masked cols). Games whose counter is not a clean
# line (or not yet characterised) are omitted rather than guessed.
EXPECTED = {
    "ls20": ([61, 62], []), "sb26": ([53], []), "sk48": ([53], []), "ar25": ([], [63]),
    "m0r0": ([0, 63], []), "tn36": ([1], []), "r11l": ([], [0]), "vc33": ([0], []),
    "tu93": ([63], []),
}
ACTIONS = 400


def masked_lines(game: str) -> tuple[list[int], list[int]]:
    import numpy as np
    import run

    run._bootstrap()
    os.environ["ARC3_SEED"] = "1"
    from agent.brain.kaggle_agent import MyAgent

    arc = run._arcade()
    card = arc.open_scorecard(tags=["maskcheck"])
    a = MyAgent(card_id=card, game_id=game, agent_name="maskcheck", ROOT_URL="x", record=False,
                arc_env=arc.make(game, scorecard_id=card))
    a.MAX_ACTIONS = ACTIONS
    a.main()
    arc.close_scorecard(card)
    m = a.brain.world.counters.ignore
    return np.flatnonzero(m.all(axis=1)).tolist(), np.flatnonzero(m.all(axis=0)).tolist()


def main() -> int:
    games = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(EXPECTED)
    with ProcessPoolExecutor(max_workers=min(8, len(games))) as ex:
        got = dict(zip(games, ex.map(masked_lines, games), strict=True))
    bad = 0
    for g in games:
        ok = tuple(got[g]) == EXPECTED[g]
        bad += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {g}: rows={got[g][0]} cols={got[g][1]}"
              + ("" if ok else f"  expected rows={EXPECTED[g][0]} cols={EXPECTED[g][1]}"))
    print(f"\n{len(games) - bad}/{len(games)} games mask exactly their counter track")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify the action interface against the real engine instead of assuming it.

Each check prints PASS/FAIL plus the observed behaviour. These are facts the
agent's design depends on (see AGENT.md "Verified interface facts"); re-run
after any `make setup` that bumps arc-agi / arcengine / the framework.

    .venv/bin/python eval/interface_checks.py
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
logging.disable(logging.CRITICAL)

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402
from pydantic import ValidationError  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def grid(f) -> np.ndarray:
    return np.asarray(f.frame[-1])


def main() -> int:
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)

    # 1. Bare GameAction(6): enum values are (id, type) tuples, so int lookup fails.
    #    The only safe int->action path is GameAction.from_id().
    try:
        GameAction(6)
        check("bare GameAction(6) raises", False, "constructed silently (interface changed - re-audit)")
    except ValueError as e:
        check("bare GameAction(6) raises", True, f"ValueError: {e} -> always use GameAction.from_id()")
    check("GameAction.from_id(6) works", GameAction.from_id(6) is GameAction.ACTION6, "returns ACTION6 singleton")

    # 2. Out-of-range coordinates are rejected at set_data time (in choose_action -> kills the thread).
    try:
        GameAction.ACTION6.set_data({"x": 64, "y": 0})
        check("ACTION6 x=64 rejected", False, "accepted silently")
    except ValidationError:
        check("ACTION6 x=64 rejected", True, "set_data raises pydantic ValidationError (uncaught => agent thread dies)")
    GameAction.ACTION6.set_data({"x": 0, "y": 0})

    # 3. Illegal (not in available_actions) actions: exception? counted? state change?
    env = arc.make("ls20")  # keyboard-only: available [1,2,3,4]
    f0 = env.step(GameAction.RESET)
    before = grid(f0).copy()
    f1 = env.step(GameAction.ACTION6, data={"x": 10, "y": 10})
    ok = f1 is not None
    changed = ok and not np.array_equal(before, grid(f1))
    check("illegal ACTION6 on ls20 does not raise", ok,
          f"returned frame={ok}, grid changed={changed}, state={getattr(f1, 'state', None)}")
    act_count = env._game._action_count if hasattr(env, "_game") else "?"
    check("illegal action still counts as an action (engine)", act_count == 1,
          f"engine per-level _action_count after 1 illegal action = {act_count}")

    # 4. Double RESET -> full reset (new scored play, back to level 1).
    env = arc.make("ls20")
    env.step(GameAction.RESET)
    env.step(GameAction.ACTION1)
    fa = env.step(GameAction.RESET)  # level reset (action_count > 0)
    fb = env.step(GameAction.RESET)  # action_count == 0 now -> full reset
    check("RESET after an action = level reset", not fa.full_reset, f"full_reset={fa.full_reset}")
    check("RESET with 0 actions in level = FULL reset", fb.full_reset,
          f"full_reset={fb.full_reset} (same path fires right after a level-up: never RESET at level start)")

    # 5. Determinism: initial grid identical across fresh envs.
    g1 = grid(arc.make("vc33").step(GameAction.RESET))
    g2 = grid(arc.make("vc33").step(GameAction.RESET))
    check("initial frame deterministic across envs", np.array_equal(g1, g2), "vc33 fresh env x2")

    # 6. GameAction singletons are shared: set_data in one thread is visible in another.
    seen: list[dict] = []
    barrier = threading.Barrier(2)

    def worker(x: int) -> None:
        GameAction.ACTION6.set_data({"x": x, "y": x})
        barrier.wait()
        seen.append(GameAction.ACTION6.action_data.model_dump())

    ts = [threading.Thread(target=worker, args=(x,)) for x in (5, 50)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    same = seen[0] == seen[1]
    check("ACTION6 data is NOT thread-local (shared enum)", same,
          f"two threads set x=5 / x=50, both then read {seen} -> Swarm threads can send each other's clicks")

    # 7. Frame layers: can one action return >1 grid (animation frames)?
    multi = {}
    for gid in ("lf52", "bp35"):
        e = arc.make(gid)
        fr = e.step(GameAction.RESET)
        multi[gid] = len(fr.frame)
    check("RESET can return multiple frame layers", any(v > 1 for v in multi.values()),
          f"layers per RESET frame: {multi} -> use frame[-1] as the settled state")

    # 8. After GAME_OVER, non-RESET actions return an empty frame.
    found = None
    rng = np.random.default_rng(0)
    for gid in ("ls20", "g50t", "tr87", "wa30"):
        e = arc.make(gid)
        fr = e.step(GameAction.RESET)
        for _ in range(400):
            acts = [a for a in fr.available_actions if a != 6] or fr.available_actions
            fr = e.step(GameAction.from_id(int(rng.choice(acts))))
            if fr is not None and fr.state == GameState.GAME_OVER:
                after = e.step(GameAction.ACTION1)
                found = (gid, len(after.frame) if after is not None else None,
                         after.state if after is not None else None)
                break
        if found:
            break
    if found:
        check("post-GAME_OVER action returns empty frame", found[1] == 0,
              f"game={found[0]} frame layers={found[1]} state={found[2]} -> must RESET")
    else:
        check("post-GAME_OVER action returns empty frame", False, "could not reach GAME_OVER randomly (inconclusive)")

    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks match expectations")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

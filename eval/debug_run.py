"""Debug run: play ONE game and print every decision - what the agent saw, did, why, what it weighed, what happened.

    .venv/bin/python eval/debug_run.py ar25                  # brain only, 150 steps
    .venv/bin/python eval/debug_run.py ar25 --steps 400 --every 10
    .venv/bin/python eval/debug_run.py ar25 --llm            # with the local LLM (start it first: make llm-local)
    .venv/bin/python eval/debug_run.py ar25 --screen         # also print the screen (hex) at every event
    .venv/bin/python eval/debug_run.py ar25 --trace          # also record it for the Agent Replay page
    make debug GAME=ar25 STEPS=150 LLM=1

Each step prints one block:
    #  12  L0  life 12  ACTION3            explore: best untried action here (chance 0.55)
         weighed  ACTION3 here c=0.55 | ACTION6 12 30 +2 steps c=0.41 | ...
         result   50 cells changed within x=34..38, y=40..49
         (it chose this on a screen it had never seen: 14 seen so far)
Lost lives and finished levels are marked; --every N prints only every Nth ordinary step (events always print).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents")]
logging.disable(logging.CRITICAL)

BOLD, DIM, RED, GREEN, PINK, RESET = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[35m", "\033[0m"


def action_name(action) -> str:
    action_id, x, y = action
    return "RESET" if action_id == 0 else f"ACTION6 {x} {y}" if action_id == 6 else f"ACTION{action_id}"


def screen_text(grid: np.ndarray) -> str:
    return "\n".join("         " + "".join(f"{v:x}" for v in row) for row in grid.tolist())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--every", type=int, default=1, help="print every Nth ordinary step (events always print)")
    ap.add_argument("--llm", action="store_true", help="use the LLM advisor at ARC3_LLM_URL (default: local server)")
    ap.add_argument("--screen", action="store_true", help="print the screen at the start and at every event")
    ap.add_argument("--trace", action="store_true",
                    help="record for the Agent Replay page (.claude/scratch/trace/debug)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    global BOLD, DIM, RED, GREEN, PINK, RESET
    if args.no_color or not sys.stdout.isatty():
        BOLD = DIM = RED = GREEN = PINK = RESET = ""

    if args.llm:
        os.environ.setdefault("ARC3_LLM_URL", "http://127.0.0.1:8000/v1")
    else:
        os.environ.pop("ARC3_LLM_URL", None)
    if args.trace:
        os.environ["ARC3_TRACE_DIR"] = str(ROOT / ".claude" / "scratch" / "trace" / "debug")

    import arc_agi
    from arcengine import GameAction

    from agent.brain.brain import Brain
    from agent.brain.llm import PromptWriter

    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                         environments_dir=str(ROOT / "environment_files"))
    card = arc.open_scorecard(tags=["debug"])
    env = arc.make(args.game, scorecard_id=card)
    frame = env.reset()
    brain = Brain(seed=args.seed, name=args.game)
    m = brain.memory
    print(f"{BOLD}{args.game}{RESET}  offered actions {sorted(frame.available_actions)}  "
          f"LLM {'ON (' + os.environ['ARC3_LLM_URL'] + ')' if args.llm else 'off'}  seed {args.seed}")
    if args.screen:
        print(screen_text(np.asarray(frame.frame[-1])))
    last_theory, started = "", time.time()
    for i in range(args.steps):
        level_before, screens_before = frame.levels_completed, brain.stats["screens"]
        grid_before = np.asarray(frame.frame[-1], dtype=np.uint8) if frame.frame else None
        brain.explorer.considered = []
        t = time.time()
        action = brain.next_action(frame.state.value, frame.frame, frame.levels_completed,
                                   list(frame.available_actions or []), bool(frame.full_reset))
        think_s = time.time() - t
        weighed = sorted(brain.explorer.considered)[:4]
        data = {"x": action[1], "y": action[2]} if action[0] == 6 else {}
        frame = env.step(GameAction.from_id(action[0]), data=data)
        grid_after = np.asarray(frame.frame[-1], dtype=np.uint8) if frame.frame else None
        hide = brain.world.counters.hide  # step counters hidden, as the brain compares screens
        result = PromptWriter.describe_step(action, hide(grid_before) if grid_before is not None else None,
                                            hide(grid_after) if grid_after is not None else None, None)
        result = result.split(" -> ", 1)[1]
        died = frame.state.value == "GAME_OVER"
        leveled = frame.levels_completed > level_before
        new_screen = brain.stats["screens"] > screens_before
        event = died or leveled
        if not event and i % args.every and not (m.theory and m.theory != last_theory):
            continue
        who = PINK if m.why.startswith("LLM") else ""
        slow = f"  {DIM}({think_s:.1f} s){RESET}" if think_s > 0.5 else ""
        print(f"{BOLD}#{i:4d}{RESET}  L{level_before}  life {m.life_length:3d}  "
              f"{BOLD}{action_name(action):16}{RESET} {who}{m.why}{RESET}{slow}")
        if weighed and not m.why.startswith("LLM"):
            options = " | ".join(f"{action_name(a)} {'here' if d == 0 else f'+{d} steps'} c={c:.2f}"
                                 for _cost, d, c, a in weighed)
            print(f"{DIM}      weighed  {options}{RESET}")
        print(f"      result   {result}")
        if m.theory and m.theory != last_theory:
            print(f"      {PINK}LLM thinks: {m.theory}{RESET}")
            if m.plan:
                print(f"      {PINK}LLM plan left: {' '.join(action_name(a) for a in m.plan)}{RESET}")
            last_theory = m.theory
        if new_screen:
            seen = brain.stats["screens"]
            print(f"{DIM}      (it chose this on a screen it had never seen: {seen} seen so far){RESET}")
        if died:
            print(f"      {RED}{BOLD}*** LIFE LOST after this action "
                  f"(lives lost so far: {brain.stats['deaths'] + 1}) ***{RESET}")
        if leveled:
            print(f"      {GREEN}{BOLD}*** LEVEL COMPLETE -> level {frame.levels_completed + 1} ***{RESET}")
        if event and args.screen and frame.frame:
            print(screen_text(np.asarray(frame.frame[-1])))
    arc.close_scorecard(card)
    s = brain.stats
    print(f"\n{BOLD}done{RESET}: {args.steps} steps in {time.time() - started:.0f} s, levels {frame.levels_completed}, "
          f"lives lost {s['deaths']}, screens {s['screens']}, LLM actions {s['llm_actions']} "
          f"(questions {s['llm_questions']}), random fallbacks {s['random_fallbacks']}")
    if args.trace:
        print(f"trace: .claude/scratch/trace/debug/{args.game}.jsonl  (build the page: "
              f".venv/bin/python eval/trace_viewer.py \"debug=.claude/scratch/trace/debug\")")
    return 0


if __name__ == "__main__":
    sys.exit(main())

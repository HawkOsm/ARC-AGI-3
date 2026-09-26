"""Debug run: play ONE game and print every decision - what the agent did, why, what it weighed, what happened.

    .venv/bin/python eval/debug_run.py ar25                  # brain only, 150 steps
    .venv/bin/python eval/debug_run.py ar25 --steps 400 --every 10
    .venv/bin/python eval/debug_run.py ar25 --llm            # LLM plans actions (start it first: make llm-local)
    .venv/bin/python eval/debug_run.py ar25 --hypothesis     # LLM explains objects + one test action per question
    .venv/bin/python eval/debug_run.py ar25 --hypothesis --show-question   # also print each question sent
    .venv/bin/python eval/debug_run.py ar25 --screen         # also print the screen (hex) at every event
    .venv/bin/python eval/debug_run.py ar25 --trace          # also record it for the Agent Replay page
    make debug GAME=ar25 STEPS=150 [LLM=1 | HYP=1]

Each step prints one block:
    #  12  L0  life 12  ACTION3            explore: best untried action here (chance 0.55)
          weighed  ACTION3 here c=0.55 | ACTION6 12 30 +2 steps c=0.41 | ...
          result   50 cells changed within x=34..38, y=40..49
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


class Colours:
    def __init__(self, on: bool) -> None:
        codes = {"bold": "\033[1m", "dim": "\033[2m", "red": "\033[31m", "green": "\033[32m", "pink": "\033[35m",
                 "end": "\033[0m"}
        for name, code in codes.items():
            setattr(self, name, code if on else "")


def action_name(action) -> str:
    action_id, x, y = action
    return "RESET" if action_id == 0 else f"ACTION6 {x} {y}" if action_id == 6 else f"ACTION{action_id}"


def screen_text(grid: np.ndarray) -> str:
    return "\n".join("         " + "".join(f"{v:x}" for v in row) for row in grid.tolist())


class StepPrinter:
    """Prints one block per decision, and the LLM's explanations and test verdicts when they are new."""

    def __init__(self, brain, c: Colours, every: int, show_screen: bool) -> None:
        self.brain, self.c, self.every, self.show_screen = brain, c, every, show_screen
        self.last_theory = ""
        self.last_hypothesis = None
        self.verdicts_seen = 0

    def step(self, i: int, level: int, action, result: str, weighed: list, think_s: float, died: bool,
             leveled: bool, new_screen: bool, after: np.ndarray | None) -> None:
        brain, c, m = self.brain, self.c, self.brain.memory
        news = self._llm_news()
        if not (died or leveled or news) and i % self.every:
            return
        who = c.pink if m.why.startswith("LLM") else ""
        slow = f"  {c.dim}({think_s:.1f} s){c.end}" if think_s > 0.5 else ""
        print(f"{c.bold}#{i:4d}{c.end}  L{level}  life {m.life_length:3d}  "
              f"{c.bold}{action_name(action):16}{c.end} {who}{m.why}{c.end}{slow}")
        if weighed and not m.why.startswith("LLM"):
            options = " | ".join(f"{action_name(a)} {'here' if d == 0 else f'+{d} steps'} c={p:.2f}"
                                 for _cost, d, p, a in weighed)
            print(f"{c.dim}      weighed  {options}{c.end}")
        print(f"      result   {result}")
        chosen = brain.goal_board.chosen
        if chosen is not None and brain.scene is not None and brain.memory.level is not None:
            from agent.brain.goals import Goals
            progress = Goals.progress_text(brain.scene.scene(brain.memory.level), chosen)
            print(f"{c.dim}      goal     {chosen.describe()} -> {progress}{c.end}")
        for line in news:
            print(line)
        if new_screen:
            seen = brain.stats["screens"]
            print(f"{c.dim}      (it chose this on a screen it had never seen: {seen} seen so far){c.end}")
        if died:
            print(f"      {c.red}{c.bold}*** LIFE LOST after this action "
                  f"(lives lost so far: {brain.stats['deaths'] + 1}) ***{c.end}")
        if leveled:
            print(f"      {c.green}{c.bold}*** LEVEL COMPLETE -> level {level + 2} ***{c.end}")
        if (died or leveled) and self.show_screen and after is not None:
            print(screen_text(after))

    def _llm_news(self) -> list[str]:
        """New LLM explanation, new test verdict, or new plan-mode theory since the last printed block."""
        c, lab, m = self.c, self.brain.lab, self.brain.memory
        lines = []
        if lab.verdicts > self.verdicts_seen:
            self.verdicts_seen = lab.verdicts
            colour = c.green if lab.last_verdict.startswith("CORRECT") else c.red
            lines.append(f"      {colour}test verdict: {lab.last_verdict}{c.end}")
        h = lab.latest
        if h is not None and h is not self.last_hypothesis:
            roles = ", ".join(f"#{i} {r}" for i, r in h.roles.items())
            board = self.brain.goal_list
            lines.append(f"      {c.dim}possible goals (brain's ranking):{c.end}")
            lines += [f"      {c.dim}  {i}. {g.describe()}  [score {g.score:.2f}]{c.end}"
                      for i, g in enumerate(board[:6], 1)]
            picked = f"#{h.goal_choice} = {board[h.goal_choice - 1].describe()}" \
                if 1 <= h.goal_choice <= len(board) else "none of the list"
            lines += [f"      {c.pink}LLM rule: {h.rule}{c.end}",
                      f"      {c.pink}    controlled {h.controlled} · picked goal {picked}{c.end}",
                      f"      {c.pink}    in its words: {h.goal}{c.end}",
                      f"      {c.pink}    roles: {roles}{c.end}"]
            self.last_hypothesis, self.last_theory = h, m.theory
        elif m.theory and m.theory != self.last_theory:
            lines.append(f"      {c.pink}LLM thinks: {m.theory}{c.end}")
            if m.plan:
                lines.append(f"      {c.pink}LLM plan left: {' '.join(action_name(a) for a in m.plan)}{c.end}")
            self.last_theory = m.theory
        return lines


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--every", type=int, default=1, help="print every Nth ordinary step (events always print)")
    ap.add_argument("--llm", action="store_true", help="LLM plans actions, at ARC3_LLM_URL (default: local server)")
    ap.add_argument("--hypothesis", action="store_true",
                    help="LLM in hypothesis mode: object questions + one test action each (implies --llm)")
    ap.add_argument("--show-question", action="store_true", help="print the full question sent to the LLM")
    ap.add_argument("--screen", action="store_true", help="print the screen at the start and at every event")
    ap.add_argument("--trace", action="store_true",
                    help="record for the Agent Replay page (.claude/scratch/trace/debug)")
    ap.add_argument("--no-color", action="store_true")
    return ap.parse_args()


def configure(args: argparse.Namespace) -> None:
    if args.hypothesis:
        args.llm = True
        os.environ["ARC3_LLM_MODE"] = "hypothesis"
    if args.llm:
        os.environ.setdefault("ARC3_LLM_URL", "http://127.0.0.1:8000/v1")
    else:
        os.environ.pop("ARC3_LLM_URL", None)
    if args.trace:
        os.environ["ARC3_TRACE_DIR"] = str(ROOT / ".claude" / "scratch" / "trace" / "debug")
    if args.show_question:
        from agent.brain import llm

        original = llm.Advisor.hypothesise

        def showing(self, question, available, tag=""):
            print("      ── question ──\n" + "\n".join("      | " + line for line in question.splitlines()))
            return original(self, question, available, tag)

        llm.Advisor.hypothesise = showing


def main() -> int:
    args = parse_args()
    c = Colours(not args.no_color and sys.stdout.isatty())
    configure(args)
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
    mode = "hypothesis" if args.hypothesis else "plans" if args.llm else "off"
    offered = sorted(frame.available_actions)
    print(f"{c.bold}{args.game}{c.end}  offered actions {offered}  LLM {mode}  seed {args.seed}")
    if args.screen:
        print(screen_text(np.asarray(frame.frame[-1])))
    printer = StepPrinter(brain, c, args.every, args.screen)
    started = time.time()
    for i in range(args.steps):
        level, screens = frame.levels_completed, brain.stats["screens"]
        before = np.asarray(frame.frame[-1], dtype=np.uint8) if frame.frame else None
        brain.explorer.considered = []
        t = time.time()
        action = brain.next_action(frame.state.value, frame.frame, frame.levels_completed,
                                   list(frame.available_actions or []), bool(frame.full_reset))
        think_s = time.time() - t
        weighed = sorted(brain.explorer.considered)[:4]
        data = {"x": action[1], "y": action[2]} if action[0] == 6 else {}
        frame = env.step(GameAction.from_id(action[0]), data=data)
        after = np.asarray(frame.frame[-1], dtype=np.uint8) if frame.frame else None
        hide = brain.world.counters.hide  # step counters hidden, as the brain compares screens
        result = PromptWriter.describe_step(action, hide(before) if before is not None else None,
                                            hide(after) if after is not None else None, None).split(" -> ", 1)[1]
        printer.step(i, level, action, result, weighed, think_s, frame.state.value == "GAME_OVER",
                     frame.levels_completed > level, brain.stats["screens"] > screens, after)
    arc.close_scorecard(card)
    s = brain.stats
    tests = f", tests correct {s['llm_tests_correct']} / wrong {s['llm_tests_wrong']}" if args.hypothesis else ""
    print(f"\n{c.bold}done{c.end}: {args.steps} steps in {time.time() - started:.0f} s, "
          f"levels {frame.levels_completed}, lives lost {s['deaths']}, screens {s['screens']}, "
          f"LLM actions {s['llm_actions']} (questions {s['llm_questions']}), "
          f"random fallbacks {s['random_fallbacks']}{tests}")
    if args.trace:
        print(f"trace: .claude/scratch/trace/debug/{args.game}.jsonl  (build the page: "
              f".venv/bin/python eval/trace_viewer.py \"debug=.claude/scratch/trace/debug\")")
    return 0


if __name__ == "__main__":
    sys.exit(main())

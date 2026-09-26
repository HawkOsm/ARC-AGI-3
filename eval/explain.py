"""Show what the agent believes about a game after playing it for a while - the World, in words.

    .venv/bin/python eval/explain.py ls20 [ACTIONS] [SEED]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))


def main() -> None:
    import run

    run._bootstrap()
    game = sys.argv[1]
    actions = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    os.environ["ARC3_SEED"] = sys.argv[3] if len(sys.argv) > 3 else "1"
    from agent.brain.kaggle_agent import MyAgent
    from agent.brain.learn import role_of, usual_step

    arc = run._arcade()
    card = arc.open_scorecard(tags=["explain"])
    agent = MyAgent(card_id=card, game_id=game, agent_name="explain", ROOT_URL="x", record=False,
                    arc_env=arc.make(game, scorecard_id=card))
    agent.MAX_ACTIONS = actions
    agent.main()
    arc.close_scorecard(card)
    brain, world = agent.brain, agent.brain.world
    ignore = world.counters["ignore"]

    print(f"== {game} after {brain.stats['actions']} actions: {brain.stats['level_ups']} level(s) finished")
    print("\nCOUNTERS (ignored): rows", np.flatnonzero(ignore.all(axis=1)).tolist(),
          "cols", np.flatnonzero(ignore.all(axis=0)).tolist(), f"({int(ignore.sum())} cells)")
    av = world.avatar
    steps = {a: usual_step(world, a) for a in sorted(av["steps"])}
    print(f"\nAVATAR: colour {av['colour']} + parts {sorted(av['parts'])}; usual step per key: {steps}")
    print("\nCOLOUR ROLES (what touching each colour did):")
    for colour, roles in sorted(world.colour_roles.items()):
        print(f"  colour {colour:2}: {role_of(world, colour) or '?':12} evidence {dict(roles)}")
    print("\nKINDS seen at first look (purpose from colour role when known):")
    for kind in sorted(world.kinds.values(), key=lambda k: -k["size"])[:25]:
        learned = role_of(world, kind["colour"])
        print(f"  {kind['name']:24} size {kind['size']:5}  purpose {learned or kind['purpose']:12} "
              f"levels {sorted(kind['seen_in_levels'])}")
    if len(world.kinds) > 25:
        print(f"  ... {len(world.kinds) - 25} more kinds")
    print("\nPATTERNS noticed:")
    for p in world.patterns[:12]:
        print(f"  level {p['level']}: {p['type']:10} x{p['count']}  {', '.join(p['kinds'][:4])}")
    from agent.brain.goals import confidence

    print("\nGOALS (decided by the game: what changed when levels were won):")
    for g in sorted(world.goals, key=lambda g: -confidence(g)):
        arrow = "up" if g["direction"] > 0 else "down"
        print(f"  {str(g['feature']):22} {arrow:4} confidence {confidence(g):.2f} "
              f"(support {g['support']}, against {g['against']})  from: {g['source']}")
    if not world.goals:
        print("  (none yet - no level won and no first-look cue)")
    print("\nRULES:", {k: (dict(v) if isinstance(v, dict) else v) for k, v in world.rules.items()})
    a = world.actions
    print("\nACTIONS (changed the screen / tries):",
          {k: f"{t.hits}/{t.tries}" for k, t in sorted(a["changed_screen"].items())},
          "| known no-op spots", len(a["no_op_at"]), "| deadly spots", len(a["died_at"]))
    for lv, m in sorted(world.levels.items()):
        print(f"LEVEL {lv}: {len(m.items)} items at first look, {len(m.screens)} screens mapped, "
              f"solved={m.has_solution}, contradictions={m.contradictions}")


if __name__ == "__main__":
    main()

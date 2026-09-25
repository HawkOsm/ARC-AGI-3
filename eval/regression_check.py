"""CI regression gate on the fixed 2-game smoke set (== `make verify-local`'s games).

Runs agent/my_agent.py with fixed seeds, in both execution models:
  * process-pool (one game per process), REPS reps per game;
  * swarm (both games as threads in one process, like Kaggle).
Fails on any crash, or if the mean score / mean levels drop below the values
logged in eval/champion/smoke_baseline.json (minus a small tolerance).

This is a tripwire for breakage, NOT a promotion signal — 2 games say nothing
about the hidden set. Promotion needs `make eval-paired` on all games (AGENT.md).

    .venv/bin/python eval/regression_check.py            # check
    .venv/bin/python eval/regression_check.py --update   # re-baseline after a promotion
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "eval" / "champion" / "smoke_baseline.json"
AGENT = "agent/my_agent.py"
GAMES = run.SMOKE_GAMES
REPS, BUDGET, SEED = 3, 400, 12345
TOL = 0.02  # relative; the agent is seeded, so reruns should reproduce exactly


def measure() -> dict:
    tasks = [(AGENT, g, rep, run.run_seed(g, rep, SEED), BUDGET, "ci") for rep in range(REPS) for g in GAMES]
    with ProcessPoolExecutor(max_workers=min(len(tasks), 6)) as ex:
        rows = list(ex.map(run.play_one, *zip(*tasks, strict=True)))
    swarm = run.run_swarm(AGENT, GAMES, BUDGET, SEED, "ci-swarm")
    return dict(
        score=statistics.fmean(r["score"] for r in rows),
        levels=statistics.fmean(r["levels"] for r in rows),
        per_game={g: statistics.fmean(r["score"] for r in rows if r["game"] == g) for g in GAMES},
        crashes=sum(1 for r in rows + swarm if r["error"]),
        swarm_levels=sum(r["levels"] for r in swarm),
        first_error=next((r["error"] for r in rows + swarm if r["error"]), None),
        config=dict(games=GAMES, reps=REPS, budget=BUDGET, seed=SEED),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()
    m = measure()
    print(json.dumps({k: v for k, v in m.items() if k != "first_error"}, indent=1))
    if m["crashes"]:
        print(f"FAIL: {m['crashes']} crashed run(s):\n{m['first_error']}")
        return 1
    if args.update:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(m, indent=1) + "\n")
        print(f"baseline written to {BASELINE.relative_to(ROOT)}")
        return 0
    if not BASELINE.exists():
        print("FAIL: no baseline; run with --update after promoting a champion")
        return 1
    base = json.loads(BASELINE.read_text())
    bad = []
    for k in ("score", "levels"):
        if m[k] < base[k] * (1 - TOL) - 1e-9:
            bad.append(f"{k}: {m[k]:.4f} < baseline {base[k]:.4f}")
    if bad:
        print("FAIL: regression vs logged champion baseline:\n  " + "\n  ".join(bad))
        return 1
    print(f"PASS: score {m['score']:.4f} (baseline {base['score']:.4f}), "
          f"levels {m['levels']:.2f} (baseline {base['levels']:.2f}), 0 crashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Local evaluation harness: N-repeat runs per game, official scorecard scoring.

Every run goes through the real framework loop (`Agent.main()`) against the
real engine, scored by the real `arc_agi` scorecard (OFFLINE mode, cached
games in environment_files/) — no reimplementation of the scoring formula.

Records per run: official score (0-100), levels completed, actions per level,
total actions, wall time, and whether the agent crashed. Results land in
eval/results/<stamp>_<label>/ (gitignored) as runs.jsonl + summary.md.

Modes
  single   : one agent, N reps per game
      .venv/bin/python eval/run.py --agent agent/my_agent.py --reps 30
  paired   : candidate vs champion on identical (game, rep, seed) triples
      .venv/bin/python eval/run.py --agent agent/my_agent.py \
          --champion eval/champion/my_agent.py --reps 30
  swarm    : one process, all games as concurrent threads (what Kaggle does);
             catches thread-safety bugs the process-pool mode cannot see
      .venv/bin/python eval/run.py --agent agent/my_agent.py --swarm

Seeding: ARC3_SEED is set per run (agents that honour it are reproducible);
the global `random`/`numpy` RNGs are also re-seeded after agent construction,
which covers the starter random agent.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import random
import statistics
import sys
import threading
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
RESULTS_DIR = ROOT / "eval" / "results"
SMOKE_GAMES = ["ls20", "vc33"]  # == `make verify-local`

_ARC = None  # one Arcade per worker process


def _bootstrap() -> None:
    for p in (str(VENDOR), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("OPERATION_MODE", "offline")
    logging.disable(logging.CRITICAL)


def _arcade():
    global _ARC
    if _ARC is None:
        _bootstrap()
        import arc_agi
        from arc_agi import OperationMode

        _ARC = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    return _ARC


def load_agent_class(path: str):
    _bootstrap()
    p = Path(path).resolve()
    name = "evalagent_" + format(zlib.crc32(str(p).encode()), "x")
    if name in sys.modules:
        return sys.modules[name].MyAgent
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.MyAgent


def all_game_ids() -> list[str]:
    return sorted(e.game_id.split("-")[0] for e in _arcade().get_environments())


def run_seed(game: str, rep: int, base_seed: int) -> int:
    return (base_seed * 1_000_003 + rep * 7919 + zlib.crc32(game.encode())) % (2**31)


def _score_from_card(sc, game: str) -> dict:
    envs = [e for e in sc.environments if e.id.split("-")[0] == game]
    if not envs:
        return dict(score=0.0, levels=0, actions=0, level_actions=[], baselines=[], plays=0)
    e = envs[0]
    best = max(e.runs, key=lambda r: r.score)
    return dict(
        score=float(e.score),
        levels=int(e.levels_completed),
        actions=int(e.actions),  # summed over all plays (full resets start a new play)
        level_actions=list(best.level_actions or []),
        baselines=list(best.level_baseline_actions or []),
        plays=len(e.runs),
        state=str(best.state.value if best.state else None),
    )


def play_one(agent_path: str, game: str, rep: int, seed: int, max_actions: int, label: str) -> dict:
    """Play one game once with a fresh scorecard. Runs inside a worker process."""
    arc = _arcade()
    import numpy as np

    Cls = load_agent_class(agent_path)
    os.environ["ARC3_SEED"] = str(seed)
    card = arc.open_scorecard(tags=[label])
    env = arc.make(game, scorecard_id=card)
    t0 = time.time()
    err = None
    agent = None
    try:
        agent = Cls(card_id=card, game_id=game, agent_name=f"eval.{label}", ROOT_URL="http://localhost",
                    record=False, arc_env=env, tags=[label])
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        agent.MAX_ACTIONS = max_actions
        agent.main()
    except Exception as e:  # a crash is a result, not a harness failure
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
    wall = time.time() - t0
    sc = arc.close_scorecard(card)
    out = dict(label=label, agent=agent_path, game=game, rep=rep, seed=seed, max_actions=max_actions,
               wall_s=round(wall, 2), error=err, framework_actions=getattr(agent, "action_counter", None))
    out.update(_score_from_card(sc, game))
    stats = getattr(agent, "eval_stats", None)  # optional agent-provided diagnostics
    if callable(stats):
        try:
            out["agent_stats"] = stats()
        except Exception as e:  # diagnostics must never fail a run
            out["agent_stats"] = {"error": repr(e)}
    return out


# ───────────────────────────── swarm mode ─────────────────────────────

def run_swarm(agent_path: str, games: list[str], max_actions: int, seed: int, label: str) -> list[dict]:
    """All games concurrently as threads in ONE process, like agents/swarm.py on Kaggle."""
    arc = _arcade()
    Cls = load_agent_class(agent_path)
    os.environ["ARC3_SEED"] = str(seed)
    card = arc.open_scorecard(tags=[label])
    agents, errors = [], {}

    def target(a):
        try:
            a.main()
        except Exception as e:
            errors[a.game_id] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"

    for g in games:
        a = Cls(card_id=card, game_id=g, agent_name=f"swarm.{label}", ROOT_URL="http://localhost",
                record=False, arc_env=arc.make(g, scorecard_id=card), tags=[label])
        a.MAX_ACTIONS = max_actions
        agents.append(a)
    t0 = time.time()
    threads = [threading.Thread(target=target, args=(a,), daemon=True) for a in agents]
    [t.start() for t in threads]
    [t.join() for t in threads]
    wall = time.time() - t0
    sc = arc.close_scorecard(card)
    rows = []
    for a in agents:
        r = dict(label=label, agent=agent_path, game=a.game_id, rep=0, seed=seed, max_actions=max_actions,
                 wall_s=round(wall, 2), error=errors.get(a.game_id), framework_actions=a.action_counter)
        r.update(_score_from_card(sc, a.game_id))
        rows.append(r)
    return rows


# ───────────────────────────── statistics ─────────────────────────────

def mean_ci(xs: list[float]) -> tuple[float, float]:
    """Mean and 95% half-width (normal approx; t would be marginally wider at n<30)."""
    if not xs:
        return 0.0, 0.0
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def aggregate_by_rep(rows: list[dict], label: str, games: list[str]) -> dict[int, float]:
    """Competition-style aggregate (mean over games) for each rep."""
    by = {}
    for r in rows:
        if r["label"] == label:
            by.setdefault(r["rep"], {})[r["game"]] = r["score"]
    return {rep: statistics.fmean(d.get(g, 0.0) for g in games) for rep, d in by.items()}


def summarize(rows: list[dict], labels: list[str], games: list[str]) -> str:
    lines = []
    for lab in labels:
        rs = [r for r in rows if r["label"] == lab]
        agg = list(aggregate_by_rep(rows, lab, games).values())
        m, h = mean_ci(agg)
        errs = sum(1 for r in rs if r["error"])
        lines.append(f"### {lab}  ({len(rs)} runs, {errs} crashed)")
        lines.append(f"Aggregate score (mean over games, per rep): **{m:.3f} ± {h:.3f}** (95% CI, n={len(agg)} reps)")
        lines.append("")
        lines.append("| game | score mean ± ci | levels mean | max levels | actions mean "
                     "| acts/level (completed) | crashes |")
        lines.append("|---|---|---|---|---|---|---|")
        for g in games:
            gr = [r for r in rs if r["game"] == g]
            if not gr:
                continue
            sm, sh = mean_ci([r["score"] for r in gr])
            lv = statistics.fmean(r["levels"] for r in gr)
            comp = [a for r in gr for a in r["level_actions"][: r["levels"]]]
            apl = f"{statistics.fmean(comp):.1f}" if comp else "-"
            lines.append(f"| {g} | {sm:.2f} ± {sh:.2f} | {lv:.2f} | {max(r['levels'] for r in gr)} | "
                         f"{statistics.fmean(r['actions'] for r in gr):.0f} | {apl} | "
                         f"{sum(1 for r in gr if r['error'])} |")
        lines.append("")
    if len(labels) == 2:
        a, b = labels
        A, B = aggregate_by_rep(rows, a, games), aggregate_by_rep(rows, b, games)
        diffs = [A[k] - B[k] for k in sorted(set(A) & set(B))]
        m, h = mean_ci(diffs)
        lines.append(f"### Paired: {a} − {b}")
        lines.append(f"Aggregate diff per rep: **{m:+.3f} ± {h:.3f}** (95% CI, n={len(diffs)})  "
                     f"→ {'SIGNIFICANT' if abs(m) > h else 'WITHIN NOISE'}")
        lines.append("")
        lines.append("| game | Δscore mean ± ci | Δlevels mean | wins/losses/ties (by score) |")
        lines.append("|---|---|---|---|")
        for g in games:
            ga = {r["rep"]: r for r in rows if r["label"] == a and r["game"] == g}
            gb = {r["rep"]: r for r in rows if r["label"] == b and r["game"] == g}
            ks = sorted(set(ga) & set(gb))
            if not ks:
                continue
            ds = [ga[k]["score"] - gb[k]["score"] for k in ks]
            dl = statistics.fmean(ga[k]["levels"] - gb[k]["levels"] for k in ks)
            dm, dh = mean_ci(ds)
            w, lo = sum(d > 0 for d in ds), sum(d < 0 for d in ds)
            lines.append(f"| {g} | {dm:+.2f} ± {dh:.2f} | {dl:+.2f} | {w}/{lo}/{len(ds) - w - lo} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="agent/my_agent.py", help="path to a file defining MyAgent")
    ap.add_argument("--champion", default=None, help="paired mode: path to the champion's MyAgent file")
    ap.add_argument("--games", default="all", help="'all', 'smoke', or comma-separated short ids")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--max-actions", type=int, default=1000, help="per-game action cap (framework MAX_ACTIONS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--swarm", action="store_true", help="run all games as threads in one process, once")
    ap.add_argument("--label", default=None, help="results folder suffix")
    args = ap.parse_args()

    _bootstrap()
    games = all_game_ids() if args.games == "all" else SMOKE_GAMES if args.games == "smoke" else \
        [g.strip() for g in args.games.split(",")]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "swarm" if args.swarm else "paired" if args.champion else "single"
    out_dir = RESULTS_DIR / f"{stamp}_{args.label or mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.swarm:
        rows = run_swarm(args.agent, games, args.max_actions, args.seed, "candidate")
        labels = ["candidate"]
    else:
        entrants = [("candidate", args.agent)] + ([("champion", args.champion)] if args.champion else [])
        labels = [lab for lab, _ in entrants]
        tasks = [(path, g, rep, run_seed(g, rep, args.seed), args.max_actions, lab)
                 for rep in range(args.reps) for g in games for lab, path in entrants]
        rows = []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(play_one, *t) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if i % max(1, len(tasks) // 10) == 0:
                    print(f"  {i}/{len(tasks)} runs ({time.time() - t0:.0f}s)", flush=True)

    with open(out_dir / "runs.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta = dict(cmd=" ".join(sys.argv), games=games, reps=args.reps, max_actions=args.max_actions,
                seed=args.seed, swarm=args.swarm, agent=args.agent, champion=args.champion)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    summary = summarize(rows, labels, games)
    (out_dir / "summary.md").write_text(f"# Eval {out_dir.name}\n\n`{meta['cmd']}`\n\n{summary}\n")
    print(summary)
    print(f"\nWrote {out_dir.relative_to(ROOT)}/")
    crashed = [r for r in rows if r["error"]]
    if crashed:
        print(f"\n{len(crashed)} run(s) crashed; first error:\n{crashed[0]['error']}")
    return 1 if crashed else 0


if __name__ == "__main__":
    raise SystemExit(main())

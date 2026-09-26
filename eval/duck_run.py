"""Run the Duck harness (vendor/duck, Tufa Labs, MIT) locally against our vLLM server — the same solver the top
public Kaggle notebooks run, pointed at `make llm-local LLM_VISION=1 LLM_TOOLS=1`.

    .venv/bin/python eval/duck_run.py --games ar25,tn36 --minutes 10 --concurrency 2
    .venv/bin/python eval/duck_run.py --games all --minutes 8 --concurrency 4 --label duck_local

Writes the harness's own run folder (per-game JSON, transcripts) under .claude/scratch/duck/runs/<label>/ and prints
levels and actions per game. The local 4B model is far weaker than Kaggle's 27B+: this checks the plumbing, not the
score.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUCK = ROOT / "vendor" / "duck"


def configure_environment(base_url: str, model: str, context: int, thinking: bool) -> None:
    """The harness reads these at import time (see vendor/duck/kaggle_setup_commands.original.json)."""
    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base_url, "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_PROVIDER": "vllm", "OPENAI_PROVIDER": "vllm",
        "LOCAL_ANALYZER_MODEL_ID": model, "INFERENCE_ANALYZER_MODEL": model,
        "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
        "LOCAL_ANALYZER_CONTEXT_WINDOW": str(context),
        "LOCAL_ANALYZER_MAX_OUTPUT": "0", "LOCAL_ANALYZER_TOOL_STEPS": "0",
        "LOCAL_ANALYZER_TOOL_TIMEOUT": "30", "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
        "LOCAL_ANALYZER_YIELD_SECONDS": "60",
        "LOCAL_ANALYZER_TEMPERATURE": "0.6", "LOCAL_ANALYZER_TOP_P": "0.95", "LOCAL_ANALYZER_TOP_K": "20",
        "LOCAL_ANALYZER_ENABLE_THINKING": "true" if thinking else "false",
        "MULTIMODAL_CONTEXT": "current_grid", "MULTIMODAL_UPSCALE": "4",
        "TAAF_MINIMAL_DIAGNOSTICS": "1", "ONLY_RESET_LEVELS": "true",
    })
    for entry in (DUCK / "ARC3-Inference", DUCK / "tufa-arc-agi-framework" / "src"):
        sys.path.insert(0, str(entry))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default="ar25", help="'all' or comma-separated short ids")
    ap.add_argument("--minutes", type=float, default=10.0, help="wall-clock budget per game")
    ap.add_argument("--max-actions", type=int, default=0, help="action cap per game (0 = none, as on Kaggle)")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--label", default="duck_local")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen", help="served model name")
    ap.add_argument("--context", type=int, default=16384, help="the harness's context window (evicts beyond it)")
    ap.add_argument("--no-thinking", action="store_true")
    return ap.parse_args()


async def run(args: argparse.Namespace) -> Path:
    import arc_agi
    import taaf.benchmark
    import taaf.game_api
    from inference.framework.solver import HarnessSolver

    env_dir = str(ROOT / "environment_files")
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    ids = sorted(e.game_id for e in arcade.available_environments)
    if args.games != "all":
        wanted = args.games.split(",")
        ids = [i for i in ids if i.split("-")[0] in wanted]
    games = [taaf.game_api.GameAPI(env_name=i, arcade_spec=spec) for i in ids]
    job_dir = ROOT / ".claude" / "scratch" / "duck" / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{args.label}"
    job_dir.mkdir(parents=True, exist_ok=True)
    solver = HarnessSolver(label="duck-harness", model="local", analyzer_timeout=900.0,
                           max_actions_per_game=args.max_actions or None,
                           max_runtime_s_per_game=args.minutes * 60, concurrency=args.concurrency)
    bm = taaf.benchmark.Benchmark(label=args.label, games=games, solver=solver, n_passes=1, job_dir=job_dir)
    rounds = -(-len(games) // max(1, args.concurrency))
    soft_end = datetime.now() + timedelta(minutes=args.minutes * rounds + 5)
    await bm.run(soft_end_time=soft_end, runtime_environment=None, minimal_diagnostics=True)
    bm._save_json()
    return job_dir


def summarize(job_dir: Path) -> None:
    """Levels and actions per game, from the harness's saved benchmark JSON."""
    files = sorted(job_dir.rglob("*.json"))
    found = False
    for path in files:
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        runs = data.get("game_runs") if isinstance(data, dict) else None
        if not runs:
            continue
        found = True
        for run in runs:
            print(json.dumps({k: run.get(k) for k in ("game", "env_name", "levels_completed", "score", "n_actions",
                                                        "actions", "state", "error") if k in run})[:300])
    if not found:
        print(f"no game_runs JSON found under {job_dir}; files: {[p.name for p in files][:20]}")


def main() -> int:
    args = parse_args()
    configure_environment(args.base_url, args.model, args.context, not args.no_thinking)
    started = time.time()
    job_dir = asyncio.run(run(args))
    print(f"done in {time.time() - started:.0f} s; run folder {job_dir}")
    summarize(job_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

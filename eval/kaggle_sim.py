"""Local emulation of Kaggle's competition rerun — runs the ACTUAL generated notebook.

What Kaggle does (reconstructed from notebooks/submission.ipynb + arc_agi's server code):
a gateway sidecar serves the hidden games over HTTP (arc_agi's Flask server, most likely with
competition_mode=True); the notebook copies the framework, registers MyAgent, writes a .env
pointing at http://gateway:8001 and runs `python main.py --agent myagent`, i.e. agents/swarm.py
with every game as a thread in one process, stepping over HTTP (RemoteEnvironmentWrapper).

This script:
  1. builds the notebook for --agent (scripts/build_notebook.build());
  2. starts an arc_agi gateway subprocess on localhost serving the chosen games, with
     competition mode on by default (--no-competition to compare);
  3. executes the notebook's run cell with /kaggle paths and the `gateway` host rewritten,
     KAGGLE_IS_COMPETITION_RERUN=1, `!` shell lines run via subprocess;
  4. parses the FINAL SCORECARD REPORT the framework logs and prints per-game results.

Differences from real Kaggle that remain: the pip-install cell is skipped (the local venv is
pinned to the same wheels), and competition_mode=True on Kaggle's gateway is inferred, not seen.

    .venv/bin/python eval/kaggle_sim.py --agent agent/my_agent.py --games ls20,vc33 --max-actions 300
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / ".claude" / "scratch" / "kaggle_sim"
VENV_PY = ROOT / ".venv" / "bin" / "python"
KAGGLE_INPUT = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def prepare_envs(games: list[str] | None) -> Path:
    """A games dir the gateway serves: a copy of the chosen cached environment_files/."""
    src = ROOT / "environment_files"
    dst = SIM / "environment_files"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for d in sorted(src.iterdir()):
        if d.is_dir() and (games is None or d.name in games):
            shutil.copytree(d, dst / d.name)  # copy: 3.12 rglob does not follow dir symlinks
    return dst


def start_gateway(port: int, envs: Path, competition: bool) -> subprocess.Popen:
    code = (
        "import logging, arc_agi; logging.disable(logging.CRITICAL);"
        f"a = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir={str(envs)!r});"
        f"a.listen_and_serve(host='127.0.0.1', port={port}, competition_mode={competition})"
    )
    log = open(SIM / "gateway.log", "w")
    proc = subprocess.Popen([str(VENV_PY), "-c", code], cwd=SIM, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/games", timeout=1)
            return proc
        except Exception:
            time.sleep(0.2)
    proc.kill()
    raise SystemExit(f"gateway did not come up; see {SIM / 'gateway.log'}")


def notebook_cells(agent: Path) -> list[str]:
    os.environ["ARC3_AGENT_SRC"] = str(agent)
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_notebook  # noqa: E402  (reads ARC3_AGENT_SRC at import)

    build_notebook.AGENT_SRC = agent.resolve()
    nb = build_notebook.build()
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in nb["cells"] if c["cell_type"] == "code"]


def rewrite(text: str, port: int) -> str:
    work = SIM / "working"
    text = text.replace(f"{KAGGLE_INPUT}/ARC-AGI-3-Agents", str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
    text = text.replace("/kaggle/working", str(work))
    text = text.replace("/tmp/my_agent.py", str(SIM / "my_agent.py"))
    return text.replace("http://gateway:8001", f"http://127.0.0.1:{port}").replace("HOST=gateway", "HOST=127.0.0.1") \
        .replace("PORT=8001", f"PORT={port}")


def ipython_to_python(cell: str) -> str:
    """Turn `!shell` lines (with backslash continuations) into __sh() calls."""
    out, lines, i = [], cell.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)!(.*)$", line)
        if not m:
            out.append(line)
            i += 1
            continue
        indent, cmd = m.group(1), m.group(2)
        while cmd.rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            cmd = cmd.rstrip()[:-1] + " " + lines[i].strip()
        out.append(f"{indent}__sh({cmd!r})")
        i += 1
    return "\n".join(out)


def run_notebook(cells: list[str], port: int, env_extra: dict) -> None:
    work = SIM / "working"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    # cell 1: pip install (skipped: local venv pinned to the same wheels)
    # cell 2: %%writefile /tmp/my_agent.py
    body = cells[1].split("\n", 1)[1]
    (SIM / "my_agent.py").write_text(body)
    env = dict(os.environ, KAGGLE_IS_COMPETITION_RERUN="1", **env_extra)
    env["PATH"] = f"{VENV_PY.parent}:{env['PATH']}"

    def __sh(cmd: str) -> None:
        cmd = cmd.replace("python main.py", f"{VENV_PY} main.py")
        r = subprocess.run(cmd, shell=True, env=env, cwd=work)
        if r.returncode:
            print(f"[kaggle_sim] shell step exited {r.returncode}: {cmd[:120]}")

    os.environ.update({k: v for k, v in env.items() if k.startswith(("KAGGLE_", "ARC3_"))})
    for cell in cells[2:]:
        exec(compile(ipython_to_python(rewrite(cell, port)), "<notebook>", "exec"), {"__sh": __sh})


def parse_scorecard() -> dict | None:
    log = SIM / "working" / "ARC-AGI-3-Agents" / "logs.log"
    if not log.exists():
        return None
    text = log.read_text()
    i = text.rfind("--- FINAL SCORECARD REPORT ---")
    if i < 0:
        return None
    j = text.find("{", i)
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return json.loads(text[j:k + 1])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="agent/my_agent.py")
    ap.add_argument("--games", default="ls20,vc33", help="comma list, or 'all'")
    ap.add_argument("--max-actions", type=int, default=300, help="ARC3_MAX_ACTIONS for agents that honour it")
    ap.add_argument("--no-competition", action="store_true", help="gateway with competition_mode=False")
    args = ap.parse_args()

    SIM.mkdir(parents=True, exist_ok=True)
    games = None if args.games == "all" else args.games.split(",")
    port = free_port()
    gw = start_gateway(port, prepare_envs(games), competition=not args.no_competition)
    t0 = time.time()
    try:
        run_notebook(notebook_cells(Path(args.agent)), port, {"ARC3_MAX_ACTIONS": str(args.max_actions)})
    finally:
        gw.terminate()
    card = parse_scorecard()
    if card is None:
        print(f"FAIL: no FINAL SCORECARD REPORT in {SIM}/working/ARC-AGI-3-Agents/logs.log")
        return 1
    print(f"\ncompetition_mode={not args.no_competition}  wall={time.time() - t0:.0f}s  "
          f"aggregate score={card.get('score', 0):.3f}")
    for e in card.get("environments", []):
        runs = e.get("runs", [])
        best = max(runs, key=lambda r: r.get("score", 0)) if runs else {}
        print(f"  {e['id']:16} score={e.get('score', 0):6.2f} levels={e.get('levels_completed')} "
              f"actions={e.get('actions')} plays={len(runs)} level_actions={best.get('level_actions')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

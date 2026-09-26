"""ARC-AGI-3 LLM dev run (Kaggle GPU notebook: RTX PRO 6000, internet OFF, attached to the competition).

Measures, it does not submit anything:
  1. the GPU/driver, and whether the offline vLLM wheelhouse installs and serves Qwen3.8-27B-FP8;
  2. the 25 public games played by agent/my_agent.py with the LLM advisor ON, all games as threads
     in one process (Kaggle's execution model), under a wall-clock budget and an action cap;
  3. every LLM request's latency and token counts (ARC3_LLM_LOG), for throughput planning.

The advisor-OFF comparison needs no GPU: run the brain locally on the same games with the same
per-game action counts (eval/run.py).

Built by notebooks/llm_dev/build.py, which embeds agent/my_agent.py below.
Outputs in /kaggle/working: dev_report.json, llm_requests.jsonl, vllm.log
"""
import base64
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zlib
from pathlib import Path

T0 = time.time()
AGENT_SOURCE = zlib.decompress(base64.b64decode("__AGENT_B64__")).decode()
MAX_ACTIONS = int("__MAX_ACTIONS__")  # per game
ARMS = json.loads("""__ARMS__""")  # [{"name", "thinking": "0"/"1", "budget_s"}], played one after another
SERVER_WAIT_S = 45 * 60
PORT = 8000
MODEL_NAME = "qwen"

# local dry run (eval/mock_llm.py instead of vLLM): DEV_INPUT, DEV_OUT, DEV_LLM_URL, DEV_SKIP_PIP
INPUT = os.environ.get("DEV_INPUT", "/kaggle/input")
OUT = Path(os.environ.get("DEV_OUT", "/kaggle/working"))
EXTERNAL_LLM = os.environ.get("DEV_LLM_URL")
SITE = Path("/tmp/vllm-site")
FRAMEWORK = Path("/tmp/fw")
report: dict = {"arms_config": ARMS, "max_actions": MAX_ACTIONS}


def save():
    report["elapsed_s"] = round(time.time() - T0)
    (OUT / "dev_report.json").write_text(json.dumps(report, indent=1, default=str))


def sh(cmd, env=None, timeout=3600):
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    print(f"$ {' '.join(map(str, cmd))[:160]} -> rc={p.returncode} ({time.time() - t:.0f}s)", flush=True)
    return p


def one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


# ── inputs and machine ───────────────────────────────────────────────────────
comp = one(f"{INPUT}/**/arc_agi_3_wheels")
wheel_lock = one(f"{INPUT}/**/wheelhouse/requirements.lock")
model_file = one(f"{INPUT}/**/layers-0.safetensors")
games_hit = one(f"{INPUT}/**/ls20/*/ls20.py")
report["inputs"] = {"competition_wheels": comp, "wheelhouse_lock": wheel_lock, "model": model_file, "games": games_hit}
report["nvidia_smi"] = sh(["nvidia-smi"]).stdout[-2500:]
report["python"] = sys.version
print(json.dumps(report["inputs"], default=str), flush=True)
if not (comp and wheel_lock and model_file and games_hit):
    report["fatal"] = "missing input"
    save()
    sys.exit(0)
MODEL_DIR = model_file.parent
GAMES_DIR = games_hit.parent.parent.parent
WHEELHOUSE = wheel_lock.parent

# ── installs ─────────────────────────────────────────────────────────────────
PIP = [sys.executable, "-m", "pip", "install", "--no-index", "--find-links"]
p = sh(["true"]) if os.environ.get("DEV_SKIP_PIP") else sh(PIP + [str(comp), "arc-agi", "python-dotenv"])
report["arc_install_rc"] = p.returncode
p = sh(["true"]) if EXTERNAL_LLM else sh(PIP + [str(WHEELHOUSE), "--target", str(SITE), "-r", str(wheel_lock)])
report["vllm_install_rc"] = p.returncode
report["vllm_install_tail"] = (p.stdout + p.stderr)[-2000:]

# A --target dir on PYTHONPATH is not a "site" dir, so its .pth files are ignored - but some wheels
# need them (nvidia-cutlass-dsl puts `cutlass` on the path that way; v1 of this run died on it).
# Let site.addsitedir() run them once and pass the paths they add to every server process.
p = sh([sys.executable, "-c", "import site, sys; before = set(sys.path); site.addsitedir(sys.argv[1]); "
        "print('\\n'.join(x for x in sys.path if x not in before and x != sys.argv[1]))", str(SITE)],
       env=dict(os.environ, PYTHONPATH=str(SITE)))
PTH_PATHS = [x for x in p.stdout.splitlines() if x.strip()]
SERVER_PYTHONPATH = os.pathsep.join([str(SITE)] + PTH_PATHS)
report["pth_paths"] = PTH_PATHS
# The image's own CUDA toolkit is too old to compile for Blackwell (sm_120 needs >= 12.9; v2 died in
# flashinfer's JIT sampler on "SM 12.x requires CUDA >= 12.9"). Use vLLM's non-JIT sampler, and point any
# other JIT at the CUDA 13 compiler that ships in the wheelhouse (nvidia-cuda-nvcc).
SERVER_ENV = dict(os.environ, PYTHONPATH=SERVER_PYTHONPATH, VLLM_USE_FLASHINFER_SAMPLER="0")
nvcc = one(f"{SITE}/nvidia/**/bin/nvcc")
if nvcc:
    SERVER_ENV["CUDA_HOME"] = str(nvcc.parent.parent)
    SERVER_ENV["PATH"] = f"{nvcc.parent}{os.pathsep}{os.environ.get('PATH', '')}"
report["cuda_home"] = SERVER_ENV.get("CUDA_HOME")
report["nvcc_versions"] = {"system": sh(["bash", "-c", "nvcc --version | tail -2"]).stdout.strip(),
                           "server": sh(["bash", "-c", "nvcc --version | tail -2"], env=SERVER_ENV).stdout.strip()}
p = sh([sys.executable, "-c", "import vllm, torch, flashinfer, cutlass; "
        "from flashinfer.gdn_prefill import chunk_gated_delta_rule; print(vllm.__version__, torch.__version__)"],
       env=SERVER_ENV)
report["server_import_check"] = {"rc": p.returncode, "out": p.stdout.strip(), "err": p.stderr[-1500:]}
save()
if p.returncode != 0 and not EXTERNAL_LLM:
    report["fatal"] = "server environment cannot import its kernels (see server_import_check)"
    save()
    sys.exit(0)

# ── the LLM server ───────────────────────────────────────────────────────────
server = None
LLM_URL = EXTERNAL_LLM or f"http://127.0.0.1:{PORT}/v1"
if not EXTERNAL_LLM:
    server_env = dict(SERVER_ENV, VLLM_NO_USAGE_STATS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    server_log = open(OUT / "vllm.log", "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(MODEL_DIR),
         "--served-model-name", MODEL_NAME, "--host", "127.0.0.1", "--port", str(PORT),
         "--max-model-len", "32768", "--gpu-memory-utilization", "0.90", "--enable-prefix-caching",
         "--reasoning-parser", "qwen3", "--trust-remote-code", "--language-model-only",
         # compact JSON only: with free whitespace the model can loop on newlines until max_tokens
         "--structured-outputs-config",
         '{"backend": "xgrammar", "disable_any_whitespace": true, "reasoning_parser": "qwen3"}'],
        env=server_env, stdout=server_log, stderr=subprocess.STDOUT)
    t = time.time()
    up = False
    while time.time() - t < SERVER_WAIT_S and server.poll() is None:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
            up = True
            break
        except Exception:
            time.sleep(10)
    report["server_up"] = up
    report["server_start_s"] = round(time.time() - t)
    report["server_exit_code"] = server.poll()
    print("server up:", up, "after", report["server_start_s"], "s", flush=True)
    if not up:
        server_log.flush()
        report["vllm_log_tail"] = (OUT / "vllm.log").read_text()[-6000:]
        save()
        server.kill()
        sys.exit(0)
    save()

# ── the games: one arm after another, same server ────────────────────────────
shutil.rmtree(FRAMEWORK, ignore_errors=True)
shutil.copytree(comp.parent / "ARC-AGI-3-Agents", FRAMEWORK, ignore=shutil.ignore_patterns(".git"))
(FRAMEWORK / "agents" / "__init__.py").write_text(
    "from .agent import Agent, Playback\nfrom .swarm import Swarm\n")
AGENT_FILE = FRAMEWORK / "agents" / "templates" / "my_agent.py"
AGENT_FILE.write_text(AGENT_SOURCE)
sys.path.insert(0, str(FRAMEWORK))

import importlib.util  # noqa: E402
import logging  # noqa: E402

import arc_agi  # noqa: E402

logging.disable(logging.CRITICAL)
arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(GAMES_DIR))
games = sorted(e.game_id.split("-")[0] for e in arc.get_environments())
report["arms"] = {}


def load_agent_class(arm_name: str):
    """A fresh copy of the agent module per arm: its clock, advisor and settings start over."""
    spec = importlib.util.spec_from_file_location(f"agent_{arm_name}", AGENT_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MyAgent


def run_arm(arm: dict) -> dict:
    os.environ.update(ARC3_LLM_URL=LLM_URL, ARC3_LLM_MODEL=MODEL_NAME, OPERATION_MODE="offline",
                      ARC3_LLM_LOG=str(OUT / f"llm_requests_{arm['name']}.jsonl"),
                      ARC3_TIME_BUDGET_S=str(arm["budget_s"]), ARC3_SEED="0",
                      ARC3_LLM_THINKING=arm["thinking"])
    agent_class = load_agent_class(arm["name"])  # the budget clock starts here
    card = arc.open_scorecard(tags=[arm["name"]])
    agents, errors = [], {}
    for g in games:
        a = agent_class(card_id=card, game_id=g, agent_name=arm["name"], ROOT_URL="http://localhost",
                        record=False, arc_env=arc.make(g, scorecard_id=card), tags=[arm["name"]])
        a.MAX_ACTIONS = MAX_ACTIONS
        agents.append(a)

    def play(a):
        try:
            a.main()
        except Exception as e:
            errors[a.game_id] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"

    t = time.time()
    threads = [threading.Thread(target=play, args=(a,), daemon=True) for a in agents]
    [th.start() for th in threads]
    [th.join() for th in threads]
    wall = round(time.time() - t)
    sc = arc.close_scorecard(card)
    rows = []
    for a in agents:
        env = [e for e in sc.environments if e.id.split("-")[0] == a.game_id.split("-")[0]]
        e = env[0] if env else None
        best = max(e.runs, key=lambda r: r.score) if e and e.runs else None
        rows.append({
            "game": a.game_id.split("-")[0],
            "score": float(e.score) if e else 0.0,
            "levels": int(e.levels_completed) if e else 0,
            "actions": int(e.actions) if e else 0,
            "level_actions": list(best.level_actions or []) if best else [],
            "baselines": list(best.level_baseline_actions or []) if best else [],
            "framework_actions": a.action_counter,
            "error": errors.get(a.game_id),
            "stats": a.eval_stats(),
        })
    result = {"config": arm, "games_wall_s": wall, "games": rows,
              "aggregate_score": sum(r["score"] for r in rows) / max(1, len(rows)),
              "total_levels": sum(r["levels"] for r in rows),
              "llm_actions": sum(r["stats"].get("llm_actions", 0) for r in rows),
              "all_actions": sum(r["framework_actions"] or 0 for r in rows)}
    print(json.dumps({k: v for k, v in result.items() if k != "games"}), flush=True)
    for r in rows:
        st = r["stats"]
        print(f"  {r['game']}: levels={r['levels']} actions={r['framework_actions']} score={r['score']:.2f} "
              f"llm_actions={st.get('llm_actions', 0)} questions={st.get('llm_questions', 0)} "
              f"err={bool(r['error'])}", flush=True)
    return result


for arm in ARMS:
    report["arms"][arm["name"]] = run_arm(arm)
    save()
if server is not None:
    server.terminate()

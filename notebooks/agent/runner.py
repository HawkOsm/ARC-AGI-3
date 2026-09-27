# %% [markdown]
# # ARC-AGI-3 — our agent (`agent/model.py`): an LLM that plays by writing Python
#
# Built by `notebooks/agent/build.py`, which fills in SETTINGS and embeds `agent/model.py` below. Serves the model
# on this GPU, then plays every game at once. In a scored run (KAGGLE_IS_COMPETITION_RERUN) the games come from
# Kaggle's game server; otherwise the 25 public games are played offline and scored locally (results.json).
#
# Models: `flashnext` = Qwen3.8-Flash-Next NVFP4 through the public Duck fork's pinned serving
# (keithtyser datasets; its tuning knobs are the TAAF_VLLM_* variables); `qwen27b` = Qwen3.8-27B-FP8 through our
# own offline vLLM wheelhouse (hawkosm/arc3-llm-prep).

# %%
import base64
import glob
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path

NOTEBOOK_START = time.time()
SETTINGS = json.loads("""__SETTINGS__""")
AGENT_SOURCE = zlib.decompress(base64.b64decode("__AGENT_B64__")).decode()
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
WORKING_DIR = Path("/kaggle/working")
NOTEBOOK_LIMIT_S = 9 * 3600  # Kaggle's run-time limit
RESERVE_S = 20 * 60  # left for closing the scorecard and writing outputs
print(f"settings {json.dumps(SETTINGS)}; submission={TRUE_SUBMISSION}", flush=True)


def sh(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    started = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(f"$ {' '.join(map(str, cmd))[:150]} -> rc={p.returncode} ({time.time() - started:.0f}s)", flush=True)
    return p


def first(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


def wait_for(url: str, timeout_s: float, process: subprocess.Popen | None = None) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"the model server exited with code {process.returncode}")
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except Exception:
            time.sleep(10)
    raise TimeoutError(f"{url} did not answer within {timeout_s:.0f} s")


# %%
# The game runtime (arc-agi) from the competition's offline wheels.
COMPETITION = first("/kaggle/input/**/arc_agi_3_wheels")
assert COMPETITION is not None, "the competition data is not attached"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--find-links",
                       str(COMPETITION), "arc-agi"], stdout=subprocess.DEVNULL)


# %%
# The model server.
def serve_flashnext() -> tuple[str, str]:
    """The Duck fork's pinned serving (its setup script, unchanged), with our tuning in TAAF_VLLM_*."""
    marker = first("/kaggle/input/**/taaf-kaggle-bundle.json")
    assert marker is not None, "the Flash-Next serving bundle (keithtyser dataset) is not attached"
    bundle = marker.parent
    inputs = {}
    for ref in SETTINGS["serving_datasets"]:
        owner, slug = ref.split("/", 1)
        found = [p for p in (Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug) if p.exists()]
        inputs[ref] = str(bundle if ref == SETTINGS["serving_datasets"][0] else (found[0] if found else ""))
    setup_env_path = WORKING_DIR / "taaf_setup_env.json"
    setup_env_path.write_text("{}")
    env = dict(os.environ, PYTHON=sys.executable, TAAF_KAGGLE_BUNDLE_DIR=str(bundle),
               TAAF_KAGGLE_WORKING_DIR=str(WORKING_DIR), TAAF_KAGGLE_SETUP_ENV=str(setup_env_path),
               TAAF_KAGGLE_INPUT_PATHS=json.dumps(inputs), TAAF_KAGGLE_DATASET_SOURCES=json.dumps(list(inputs)),
               TAAF_VLLM_ENABLE_PREFIX_CACHING="1" if SETTINGS["prefix_cache"] else "0",
               TAAF_VLLM_MAX_NUM_SEQS=str(SETTINGS["max_seqs"]),
               TAAF_VLLM_KV_CACHE_MEMORY_BYTES=str(SETTINGS["kv_cache_bytes"]),
               TAAF_VLLM_KV_CACHE_DTYPE=SETTINGS["kv_cache_dtype"],
               TAAF_VLLM_MAX_NUM_BATCHED_TOKENS="8192", TAAF_VLLM_MTP_TOKENS="3",
               TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE="32", TAAF_VLLM_OMP_THREADS="1")
    for command in json.loads((bundle / "setup_commands.json").read_text()):
        print(f"setup: {command}", flush=True)
        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
    setup = json.loads(setup_env_path.read_text())
    return setup["LOCAL_ANALYZER_BASE_URL"], setup["LOCAL_ANALYZER_MODEL_ID"]


def serve_qwen27b() -> tuple[str, str]:
    """Qwen3.8-27B-FP8 through our offline vLLM wheelhouse (the fixes from our earlier Kaggle runs kept)."""
    lock = first("/kaggle/input/**/wheelhouse/requirements.lock")
    weights = first("/kaggle/input/**/layers-0.safetensors")
    assert lock is not None and weights is not None, "the vLLM wheelhouse or the 27B model is not attached"
    site = Path("/tmp/vllm-site")
    p = sh([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(lock.parent),
            "--target", str(site), "-r", str(lock)])
    assert p.returncode == 0, p.stderr[-2000:]
    # a --target dir is not a "site" dir: collect the paths its .pth files add, for the server process
    p = sh([sys.executable, "-c", "import site, sys; before = set(sys.path); site.addsitedir(sys.argv[1]); "
            "print('\\n'.join(x for x in sys.path if x not in before and x != sys.argv[1]))", str(site)],
           env=dict(os.environ, PYTHONPATH=str(site)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(site)] + p.stdout.split()),
               VLLM_USE_FLASHINFER_SAMPLER="0", VLLM_NO_USAGE_STATS="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1")
    nvcc = first(f"{site}/nvidia/**/bin/nvcc")  # Blackwell JIT needs the CUDA 13 compiler from the wheelhouse
    if nvcc:
        env["CUDA_HOME"], env["PATH"] = str(nvcc.parent.parent), f"{nvcc.parent}{os.pathsep}{env.get('PATH', '')}"
    port = 8000
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(weights.parent),
               "--served-model-name", "qwen", "--host", "127.0.0.1", "--port", str(port),
               "--max-model-len", "32768", "--gpu-memory-utilization", "0.92", "--trust-remote-code",
               "--max-num-seqs", str(SETTINGS["max_seqs"]), "--kv-cache-dtype", SETTINGS["kv_cache_dtype"],
               "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
               "--default-chat-template-kwargs", '{"preserve_thinking": true}',
               "--limit-mm-per-prompt", '{"image": 4, "video": 0}',
               "--enable-prefix-caching" if SETTINGS["prefix_cache"] else "--no-enable-prefix-caching"]
    log = open(WORKING_DIR / "vllm.log", "w")
    server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    wait_for(f"http://127.0.0.1:{port}/v1/models", 45 * 60, server)
    return f"http://127.0.0.1:{port}/v1", "qwen"


serve_started = time.time()
BASE_URL, MODEL_NAME = serve_flashnext() if SETTINGS["model"] == "flashnext" else serve_qwen27b()
SERVER_START_S = round(time.time() - serve_started)
print(f"model server ready: {MODEL_NAME} at {BASE_URL} after {SERVER_START_S} s", flush=True)

# %%
# The agent: agent/model.py, embedded by build.py; its settings are AGENT_* environment variables.
(WORKING_DIR / "agent").mkdir(exist_ok=True)
(WORKING_DIR / "agent" / "__init__.py").write_text("")
(WORKING_DIR / "agent" / "model.py").write_text(AGENT_SOURCE)
sys.path.insert(0, str(WORKING_DIR))
os.environ.update(AGENT_BASE_URL=BASE_URL, AGENT_MODEL=MODEL_NAME, AGENT_CONTEXT=str(SETTINGS["context"]),
                  AGENT_SCREEN=SETTINGS["screen"], AGENT_MAX_TOKENS=str(SETTINGS["max_tokens"]),
                  AGENT_THINKING="1" if SETTINGS["thinking"] else "0",
                  AGENT_TRACE_DIR="" if TRUE_SUBMISSION else str(WORKING_DIR / "traces"))
import agent.model as agent_model  # noqa: E402

# %%
# Play. Every game at once (one thread each); in a scored run with more games than MAX_PARALLEL_GAMES they go in
# rounds, and each game's time shrinks so that everything ends inside Kaggle's 9 h.
if TRUE_SUBMISSION:
    os.environ.setdefault("ARC_API_KEY", "test-key-123")
    os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
    wait_for(f"{os.environ['ARC_BASE_URL']}api/games", 600)
    arcade = agent_model.open_arcade("competition")
else:
    games_dir = first("/kaggle/input/**/environment_files")
    assert games_dir is not None, "the public game files are not attached"
    arcade = agent_model.open_arcade("offline", str(games_dir))
game_ids = sorted({e.game_id for e in arcade.available_environments})
if not game_ids:
    raise RuntimeError("the arcade lists no games (competition gateway down, or no game files found)")
parallel = min(len(game_ids), SETTINGS["max_parallel_games"])
rounds = math.ceil(len(game_ids) / parallel)
left_s = NOTEBOOK_LIMIT_S - (time.time() - NOTEBOOK_START) - RESERVE_S
# a submission's own Save & Run only has to finish cleanly: it may use a short play time (dev_minutes_per_game)
wanted = SETTINGS["minutes_per_game"] if TRUE_SUBMISSION else (SETTINGS["dev_minutes_per_game"]
                                                               or SETTINGS["minutes_per_game"])
minutes = min(wanted, left_s / rounds / 60)
print(f"playing {len(game_ids)} games, {parallel} at once, {minutes:.0f} min each", flush=True)
results = agent_model.play_games(arcade, game_ids, minutes, parallel, agent_model.Config())
mean = sum(r["score"] for r in results.values()) / max(1, len(results))
summary = {"settings": SETTINGS, "mean_score": mean, "total_levels": sum(r["levels"] for r in results.values()),
           "minutes_per_game": minutes, "server_start_s": SERVER_START_S,
           "session_s": round(time.time() - NOTEBOOK_START), "games": results}
(WORKING_DIR / "results.json").write_text(json.dumps(summary, indent=1))
print(f"mean score {mean:.2f}, levels {summary['total_levels']}", flush=True)
if not TRUE_SUBMISSION:  # Save & Run expects a valid file; a scored run is scored by the game server instead
    import pandas as pd
    pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
        WORKING_DIR / "submission.parquet", index=False)

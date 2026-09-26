"""Build the pushable dev kernel: embeds agent/my_agent.py into dev.py and writes kernel-metadata.json.

    .venv/bin/python notebooks/llm_dev/build.py --out .claude/scratch/llm/kaggle/dev_push \
        --arm nothink:0:5400 --arm think:1:2700 --max-actions 1000
"""
import argparse
import base64
import json
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--arm", action="append", default=[], metavar="NAME:THINKING:BUDGET_S",
                help="e.g. nothink:0:5400 think:1:2700 (played in this order)")
ap.add_argument("--max-actions", type=int, default=1000)
args = ap.parse_args()
arms = [dict(zip(("name", "thinking", "budget_s"), a.split(":"), strict=True)) for a in args.arm or ["nothink:0:5400"]]
for a in arms:
    a["budget_s"] = float(a["budget_s"])
agent = (ROOT / "agent" / "my_agent.py").read_bytes()
src = (HERE / "dev.py").read_text()
src = (src.replace("__AGENT_B64__", base64.b64encode(zlib.compress(agent, 9)).decode())
       .replace("__ARMS__", json.dumps(arms)).replace("__MAX_ACTIONS__", str(args.max_actions)))
compile(src, "dev.py", "exec")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "dev.py").write_text(src)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "hawkosm/arc3-llm-dev", "title": "arc3-llm-dev", "code_file": "dev.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": False, "machine_shape": "NvidiaRtxPro6000", "keywords": [],
    "dataset_sources": ["hawkosm/arc3-public-games"], "kernel_sources": ["hawkosm/arc3-llm-prep"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": ["michaelpoluektov/qwen3-8-27b-fp8/transformers/default/1"]}, indent=1))
print(f"wrote {out}/dev.py ({len(src)} bytes) arms={arms} max_actions={args.max_actions}")

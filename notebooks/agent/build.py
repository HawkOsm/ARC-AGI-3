"""Build the Kaggle notebook for our agent: notebooks/agent/runner.py + agent/model.py + settings -> an .ipynb and
its kernel-metadata.json, ready to push (nothing is pushed here).

    .venv/bin/python notebooks/agent/build.py --model flashnext --context 12000 --prefix-cache
    .venv/bin/kaggle kernels push -p notebooks/agent/build            # the owner pushes

The three throughput levers from the run-A analysis (docs/tests/LOG.md 2026-09-27) are settings:
    --context       conversation budget per call in tokens (run A: 32768, ~20k per call in practice)
    --prefix-cache  keep the server's prefix cache on (run A: off)
    --model         flashnext (Qwen3.8-Flash-Next NVFP4, run A's model) or qwen27b (Qwen3.8-27B-FP8: smaller weights,
                    so a much bigger cache and every game at once)
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {  # what each model needs attached
    "flashnext": {"dataset_sources": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                      "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                  "kernel_sources": [],
                  "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                  # the image the fork's serving runtime was built and tested on (from its kernel-metadata.json)
                  "docker_image": "gcr.io/kaggle-private-byod/python@sha256:"
                                  "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c"},
    "qwen27b": {"dataset_sources": [], "kernel_sources": ["hawkosm/arc3-llm-prep"],
                "model_sources": ["michaelpoluektov/qwen3-8-27b-fp8/transformers/default/1"]},
}


def cells(source: str) -> list[dict]:
    """Split on `# %%` / `# %% [markdown]` markers into notebook cells."""
    out = []
    for block in re.split(r"^# %%", source, flags=re.M)[1:]:
        markdown = block.startswith(" [markdown]")
        body = block.split("\n", 1)[1] if "\n" in block else ""
        if markdown:
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            out.append({"cell_type": "markdown", "metadata": {}, "source": body.strip().splitlines(keepends=True)})
        else:
            out.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                        "source": body.strip().splitlines(keepends=True)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(SOURCES), default="flashnext")
    ap.add_argument("--context", type=int, default=12000, help="AGENT_CONTEXT: tokens per call")
    ap.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-seqs", type=int, default=28, help="requests the server runs at once")
    ap.add_argument("--kv-cache-bytes", type=int, default=5 * 1024**3,
                    help="flashnext: fixed cache size (run A: 5 GiB; 0 = let vLLM size it, risk of OOM)")
    ap.add_argument("--kv-cache-dtype", choices=["auto", "fp8"], default="auto", help="fp8 ~doubles the cache")
    ap.add_argument("--minutes-per-game", type=float, default=130.0)
    ap.add_argument("--dev-minutes-per-game", type=float, default=0.0,
                    help="play time in a normal Save & Run only (0 = same as the scored run); a submission's Save & "
                         "Run just has to finish, e.g. 5 saves ~2 GPU hours")
    ap.add_argument("--max-parallel-games", type=int, default=28)
    ap.add_argument("--screen", choices=["image", "text", "both", "none"], default="image")
    ap.add_argument("--max-tokens", type=int, default=0, help="per reply (0 = no cap, as run A)")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--kernel", default="hawkosm/arc3-agent", help="Kaggle kernel id to push as")
    ap.add_argument("--out", default=str(HERE / "build"))
    args = ap.parse_args()
    settings = {"model": args.model, "context": args.context, "prefix_cache": args.prefix_cache,
                "max_seqs": args.max_seqs, "kv_cache_bytes": args.kv_cache_bytes,
                "kv_cache_dtype": args.kv_cache_dtype, "minutes_per_game": args.minutes_per_game,
                "dev_minutes_per_game": args.dev_minutes_per_game,
                "max_parallel_games": args.max_parallel_games, "screen": args.screen,
                "max_tokens": args.max_tokens, "thinking": not args.no_thinking,
                "serving_datasets": SOURCES[args.model]["dataset_sources"]}
    agent = (ROOT / "agent" / "model.py").read_bytes()
    source = (HERE / "runner.py").read_text()
    source = source.replace("__SETTINGS__", json.dumps(settings)).replace(
        "__AGENT_B64__", base64.b64encode(zlib.compress(agent, 9)).decode())
    compile(source, "runner.py", "exec")
    notebook = {"cells": cells(source), "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                      "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "arc3-agent.ipynb").write_text(json.dumps(notebook, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": args.kernel, "title": args.kernel.split("/", 1)[1], "code_file": "arc3-agent.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaRtxPro6000", "keywords": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"], **SOURCES[args.model]}, indent=1))
    print(f"wrote {out}/arc3-agent.ipynb ({len(source) // 1024} KB) and kernel-metadata.json\n"
          f"settings {json.dumps({k: v for k, v in settings.items() if k != 'serving_datasets'})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

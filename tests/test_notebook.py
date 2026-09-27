"""notebooks/agent/build.py: both model variants build into a notebook with the right inputs and settings."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _build(tmp_path, *args):
    subprocess.run([sys.executable, str(ROOT / "notebooks/agent/build.py"), "--out", str(tmp_path), *args],
                   check=True, capture_output=True)
    notebook = json.loads((tmp_path / "arc3-agent.ipynb").read_text())
    return notebook, json.loads((tmp_path / "kernel-metadata.json").read_text())


def test_flashnext_notebook_carries_the_throughput_settings(tmp_path):
    notebook, meta = _build(tmp_path, "--model", "flashnext", "--context", "12000", "--no-prefix-cache")
    code = "".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    settings = json.loads(code.split('SETTINGS = json.loads("""', 1)[1].split('"""', 1)[0])
    assert settings["context"] == 12000 and settings["prefix_cache"] is False and settings["model"] == "flashnext"
    assert meta["model_sources"][0].startswith("keithtyser/qwen3-8-flash-next-nvfp4")
    assert meta["docker_image"] and meta["enable_internet"] is False and meta["is_private"] is True
    compile(code, "notebook", "exec")


def test_qwen27b_notebook_attaches_our_wheelhouse_and_the_27b_model(tmp_path):
    _notebook, meta = _build(tmp_path, "--model", "qwen27b")
    assert meta["kernel_sources"] == ["hawkosm/arc3-llm-prep"] and "docker_image" not in meta
    assert meta["model_sources"][0].startswith("michaelpoluektov/qwen3-8-27b-fp8")

"""ARC-AGI-3 LLM prep (Kaggle CPU notebook, internet ON, NOT attached to the competition).

Two jobs, both checked by measurement rather than trust:
  1. VERIFY the attached Kaggle mirror of Qwen/Qwen3.8-27B-FP8 file-by-file against the Hugging
     Face manifest for revision 017b9c7af6 (sha256 for LFS files, git blob sha1 for the rest).
  2. BUILD an offline vLLM wheelhouse (every wheel vLLM needs, for this Kaggle image's Python),
     then prove it installs with --no-index into a fresh --target directory and imports.

Outputs in /kaggle/working:
  wheelhouse/            the wheels + requirements.lock (attach this notebook's output later)
  prep_report.json       every measurement below
"""
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VLLM_VERSION = "0.30.0"
OUT = Path("/kaggle/working")
WHEELHOUSE = OUT / "wheelhouse"
TEST_SITE = Path("/tmp/vllm-site")
MANIFEST = json.loads(r"""__MANIFEST__""")

report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "vllm_version": VLLM_VERSION}


def run(cmd, env=None, timeout=3600):
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    print(f"$ {' '.join(map(str, cmd))[:200]}  -> rc={p.returncode} in {time.time() - t:.0f}s", flush=True)
    return p


# ── environment ──────────────────────────────────────────────────────────────
report["python"] = sys.version
report["platform"] = platform.platform()
report["glibc"] = platform.libc_ver()
report["cpu_count"] = os.cpu_count()
report["disk_working"] = run(["df", "-h", "/kaggle/working", "/tmp"]).stdout
print(json.dumps({k: report[k] for k in ("python", "platform", "glibc", "cpu_count")}, indent=1), flush=True)


# ── 1. verify the model mirror ───────────────────────────────────────────────
def file_digest(path: Path, lfs: bool) -> str:
    h = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


hits = glob.glob("/kaggle/input/**/layers-0.safetensors", recursive=True)
report["model_candidates"] = hits
if not hits:
    report["model_verify"] = {"ok": False, "error": "model not mounted"}
else:
    root = Path(hits[0]).parent
    report["model_root"] = str(root)
    present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    expected = MANIFEST["files"]
    missing = sorted(set(expected) - present)
    extra = sorted(present - set(expected))
    t = time.time()

    def check(name):
        meta = expected[name]
        path = root / name
        lfs = meta["sha256"] is not None
        size_ok = path.stat().st_size == meta["size"]
        got = file_digest(path, lfs)
        want = meta["sha256"] if lfs else meta["git_blob"]
        return name, size_ok and got == want, got

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(check, sorted(set(expected) & present)))
    bad = [(n, g) for n, ok, g in results if not ok]
    report["model_verify"] = {
        "ok": not missing and not bad,
        "checked": len(results),
        "expected": len(expected),
        "missing": missing,
        "extra_files": extra,
        "mismatched": bad,
        "seconds": round(time.time() - t),
        "revision": MANIFEST["revision"],
    }
print("MODEL VERIFY:", json.dumps(report["model_verify"], indent=1)[:3000], flush=True)


# ── 2. build and prove the wheelhouse ────────────────────────────────────────
WHEELHOUSE.mkdir(parents=True, exist_ok=True)
p = run([sys.executable, "-m", "pip", "wheel", f"vllm=={VLLM_VERSION}", "-w", str(WHEELHOUSE)], timeout=5400)
report["pip_wheel_rc"] = p.returncode
report["pip_wheel_tail"] = (p.stdout + p.stderr)[-3000:]

p = run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(WHEELHOUSE),
         "--target", str(TEST_SITE), f"vllm=={VLLM_VERSION}"], timeout=3600)
report["offline_install_rc"] = p.returncode
report["offline_install_tail"] = (p.stdout + p.stderr)[-3000:]

env = dict(os.environ, PYTHONPATH=str(TEST_SITE), VLLM_NO_USAGE_STATS="1")
p = run([sys.executable, "-c",
         "import torch, vllm, transformers; print(vllm.__version__, torch.__version__, torch.version.cuda, "
         "transformers.__version__, torch.cuda.get_arch_list() if hasattr(torch.cuda,'get_arch_list') else '')"],
        env=env)
report["import_check"] = {"rc": p.returncode, "stdout": p.stdout.strip(), "stderr_tail": p.stderr[-2000:]}

p = run([sys.executable, "-m", "pip", "freeze", "--path", str(TEST_SITE)])
(WHEELHOUSE / "requirements.lock").write_text(p.stdout)
report["locked_packages"] = len(p.stdout.splitlines())

wheels = sorted(WHEELHOUSE.glob("*.whl"))
report["wheel_count"] = len(wheels)
report["wheelhouse_gb"] = round(sum(w.stat().st_size for w in wheels) / 1e9, 2)
report["key_wheels"] = [w.name for w in wheels if w.name.split("-")[0].lower().replace("_", "-")
                        in ("vllm", "torch", "flashinfer-python", "transformers", "triton", "xformers")]
report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

(OUT / "prep_report.json").write_text(json.dumps(report, indent=1))
print("REPORT:", json.dumps({k: v for k, v in report.items() if not k.endswith("_tail")}, indent=1), flush=True)

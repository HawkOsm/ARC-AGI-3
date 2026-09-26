"""Build the Kaggle dataset `hawkosm/arc3-duck-hints`: our addition to the Duck harness, for notebooks/duck_hints.

    .venv/bin/python scripts/build_duck_hints_dataset.py      # -> kaggle_datasets/duck_hints/ (gitignored build output)

Contents (no model weights, no third-party serving code):
    ARC3-Inference/     the Duck harness from vendor/duck (Tufa Labs, MIT) with our 28-line `brain_hints` addition
    arc3-brain/agent/   our agent package (agent/brain + agent/duck/hints.py), imported by the harness when
                        DUCK_BRAIN_HINTS=1
    NOTICE.md           credits and licence of the harness
The notebook attaches it next to the public Flash-Next bundle and puts these two folders first on sys.path, so our
harness copy replaces the bundle's one and everything else (serving, model) stays exactly as in the public notebook.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle_datasets" / "duck_hints"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "viewer", "uv.lock")


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT / "vendor" / "duck" / "ARC3-Inference", OUT / "ARC3-Inference", ignore=IGNORE)
    brain = OUT / "arc3-brain" / "agent"
    brain.mkdir(parents=True)
    (brain / "__init__.py").write_text("")
    for package in ("brain", "duck"):
        shutil.copytree(ROOT / "agent" / package, brain / package, ignore=IGNORE)
    shutil.copy(ROOT / "vendor" / "duck" / "NOTICE.md", OUT / "NOTICE.md")
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "arc3-duck-hints", "id": "hawkosm/arc3-duck-hints",
        "licenses": [{"name": "MIT"}],
    }, indent=2) + "\n")
    files = sum(1 for p in OUT.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT} ({files} files, {size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

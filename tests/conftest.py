"""Shared test setup: import agent.brain from the repo and the framework from vendor/ (make setup)."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents")):
    if p not in sys.path:
        sys.path.insert(0, p)
logging.disable(logging.CRITICAL)

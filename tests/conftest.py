"""Shared test setup: import `agent` from the repo root; keep the game engine's logging quiet."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)

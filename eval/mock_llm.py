"""A fake OpenAI-compatible LLM server for testing the advisor plumbing (agent/brain/llm.py) locally.

It answers /v1/chat/completions with a random plan of LEGAL actions (read from the prompt's
"Available actions:" line), and can be told to misbehave: answer slowly, fail, or answer garbage.
It measures nothing about play quality - a real model is needed for that (Kaggle GPU notebook).

    .venv/bin/python eval/mock_llm.py --port 8765 --delay 0.05 --fail-rate 0.1 --garbage-rate 0.1
    ARC3_LLM_URL=http://127.0.0.1:8765/v1 .venv/bin/python eval/run.py ...

On exit (Ctrl-C / SIGTERM) it prints request counts and prompt sizes (characters, and a rough
token estimate) - the prompt size is what the real server's context and throughput must handle.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import signal
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATS = {"requests": 0, "failed_on_purpose": 0, "garbage_on_purpose": 0, "prompt_chars": []}
LOCK = threading.Lock()


def make_handler(args, rng: random.Random):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._send(200, {"data": [{"id": "mock"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            prompt = body["messages"][-1]["content"]
            with LOCK:
                STATS["requests"] += 1
                STATS["prompt_chars"].append(len(prompt) + len(body["messages"][0]["content"]))
                roll = rng.random()
            time.sleep(args.delay)
            if roll < args.fail_rate:
                with LOCK:
                    STATS["failed_on_purpose"] += 1
                self._send(500, {"error": "mock failure"})
                return
            if roll < args.fail_rate + args.garbage_rate:
                with LOCK:
                    STATS["garbage_on_purpose"] += 1
                content = "I think we should move up. {plan: ACTION1"
            else:
                content = "Theory: exploring.\n" + json.dumps({"plan": _random_plan(prompt, rng)})
            self._send(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _random_plan(prompt: str, rng: random.Random) -> list[str]:
    m = re.search(r"Available actions: ([^\n]*)", prompt)
    ids = [int(a) for a in re.findall(r"ACTION(\d)", m.group(1))] if m else [1, 2, 3, 4]
    plan = []
    for _ in range(rng.randint(1, 8)):
        a = rng.choice(ids)
        # include a few out-of-range and illegal steps: the parser must drop them
        plan.append(f"ACTION6 {rng.randrange(70)} {rng.randrange(64)}" if a == 6 else f"ACTION{a}")
    if rng.random() < 0.1:
        plan.append("RESET")
    return plan


def report() -> None:
    chars = STATS["prompt_chars"]
    out = {k: v for k, v in STATS.items() if k != "prompt_chars"}
    if chars:
        out.update(prompt_chars_median=int(statistics.median(chars)), prompt_chars_max=max(chars),
                   rough_tokens_median=int(statistics.median(chars) / 2.5))
    print(json.dumps(out), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds per answer")
    ap.add_argument("--fail-rate", type=float, default=0.0)
    ap.add_argument("--garbage-rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args, random.Random(args.seed)))

    def stop(*_):
        report()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    print(f"mock LLM on http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

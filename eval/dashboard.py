"""Run board: one self-contained HTML page with every run we have data for (make dashboard).

Reads only files already on disk - it runs nothing:
  .claude/scratch/llm/kaggle/dev_out_v*/     Kaggle GPU dev runs (dev_report.json, llm_requests*.jsonl)
  eval/results/*_brainref_<game>/            brain-only replays at the LLM arm's per-game action counts
  eval/results/*_step1_ablation_n30/         PLAYABLES step 1, N=30
  eval/results/*_llm_local, *_brain_swarm_200  local 4B runs and their brain reference
  .claude/scratch/llm/local/requests.jsonl  the latest local run's LLM requests (answers tagged by game)
  environment_files/                         the 25 public games (first screens, offered actions)
Writes .claude/scratch/dashboard/arc3-run-board.html (publish it as an artifact, or open it in a browser).
"""
from __future__ import annotations

import base64
import html
import io
import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / ".claude" / "scratch" / "llm" / "kaggle"
RESULTS = ROOT / "eval" / "results"
OUT = ROOT / ".claude" / "scratch" / "dashboard" / "arc3-run-board.html"
# The ARC-AGI-3 palette (framework agents/templates/reasoning_agent.py), colour index -> hex
PALETTE = ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000", "#E53AA3", "#FF7BCC",
           "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00", "#FF851B", "#921231", "#4FCC30", "#A356D6"]

# What happened on each Kaggle GPU dev run (facts from docs/tests/LOG.md; numbers are filled from the files).
KAGGLE_STORY = [
    ("v1", "failed", "Server never started", "flashinfer could not import `cutlass`: the wheel exposes it only via a "
     ".pth file, which a --target install on PYTHONPATH ignores.", "Collect the .pth paths into the server's PYTHONPATH; "
     "pre-flight import check."),
    ("v2", "failed", "Server died on its first token", "flashinfer JIT-compiled its sampler with the image's CUDA 12.8 "
     "(sm_120 needs ≥ 12.9).", "VLLM_USE_FLASHINFER_SAMPLER=0; CUDA_HOME → the wheelhouse's CUDA 13 nvcc."),
    ("v3", "measured", "Server up; the LLM took 0 actions", "Thinking mode: every answer was still thinking at 6,144 "
     "tokens; timeouts paused the advisor and the brain played.", "Thinking off by default; Qwen's own sampling "
     "settings; runs with several arms."),
    ("v4", "measured", "Thinking vs no thinking", "Thinking: 9/10 answers never finished even at 16,384 tokens. "
     "No thinking: 58% of answers were prose that hit 1,024 tokens.", "Structured JSON output + compact JSON "
     "(found on the local 4B model)."),
    ("v5", "measured", "No thinking + structured JSON: clean answers, far worse play", "Every answer parsed (9,292/9,292, "
     "12.7 s median) and the LLM chose 98% of actions: 1 level vs the brain's 17 at the same actions. Plans are mostly "
     "one key repeated; no theory mentions a mirror.", "Next step: owner's choice after watching replays."),
]


# ── data ─────────────────────────────────────────────────────────────────────


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def request_health(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    ok = [r for r in rows if "usage" in r and r["usage"]]
    finish = defaultdict(int)
    for r in ok:
        finish[r.get("finish") or "?"] += 1
    secs = sorted(r["seconds"] for r in ok) or [0.0]
    out_tokens = [r["usage"].get("completion_tokens", 0) for r in ok]
    return {"requests": len(rows), "errors": len(rows) - len(ok), "stop": finish.get("stop", 0),
            "length": finish.get("length", 0), "p50": secs[len(secs) // 2], "p90": secs[int(0.9 * (len(secs) - 1))],
            "tokens_p50": statistics.median(out_tokens) if out_tokens else 0}


def kaggle_runs() -> list[dict]:
    runs = []
    for version, status, headline, cause, fix in KAGGLE_STORY:
        folder = KAGGLE / f"dev_out_{version}"
        report = json.loads((folder / "dev_report.json").read_text()) if (folder / "dev_report.json").exists() else {}
        arms = []
        if "arms" in report:
            for name, arm in report["arms"].items():
                arms.append({"name": name, "levels": arm["total_levels"], "score": arm["aggregate_score"],
                             "llm_actions": arm["llm_actions"], "all_actions": arm["all_actions"],
                             "minutes": round(arm["games_wall_s"] / 60),
                             "health": request_health(read_jsonl(folder / f"llm_requests_{name}.jsonl"))})
        elif "games" in report:
            arms.append({"name": "think (6k)", "levels": report["total_levels"], "score": report["aggregate_score"],
                         "llm_actions": sum(g["stats"].get("llm_actions", 0) for g in report["games"]),
                         "all_actions": sum(g["framework_actions"] or 0 for g in report["games"]),
                         "minutes": round(report.get("games_wall_s", 0) / 60),
                         "health": request_health(read_jsonl(folder / "llm_requests.jsonl"))})
        runs.append({"version": version, "status": status, "headline": headline, "cause": cause, "fix": fix,
                     "server_s": report.get("server_start_s") if report.get("server_up") else None, "arms": arms})
    return runs


def latest(pattern: str) -> Path | None:
    hits = sorted(RESULTS.glob(pattern))
    return hits[-1] if hits else None


def v4_versus_brain() -> list[dict]:
    """Per game: the 27B no-thinking arm vs the brain replayed at the same action count (seed 0)."""
    path = KAGGLE / "dev_out_v4" / "dev_report.json"
    if not path.exists():
        return []
    rows = []
    for g in json.loads(path.read_text())["arms"]["nothink"]["games"]:
        ref = latest(f"*_brainref_{g['game']}")
        brain = read_jsonl(ref / "runs.jsonl")[0] if ref else None
        if brain is None:
            continue
        rows.append({"game": g["game"], "actions": g["framework_actions"], "llm_levels": g["levels"],
                     "llm_score": g["score"], "brain_levels": brain["levels"], "brain_score": brain["score"]})
    return rows


def step1_ablation() -> tuple[list[dict], dict[str, float]]:
    """Paired per-rep deltas vs the champion, and the champion's mean levels per game."""
    folder = latest("*_step1_ablation_n30")
    if folder is None:
        return [], {}
    runs = read_jsonl(folder / "runs.jsonl")
    games = sorted({r["game"] for r in runs})
    per = defaultdict(lambda: defaultdict(dict))  # label -> rep -> game -> run
    for r in runs:
        per[r["label"]][r["rep"]][r["game"]] = r
    ref = "candidate"
    out = []
    for label in ("kind", "mixed", "both"):
        if label not in per:
            continue
        ds, dl = [], []
        for rep, by_game in per[label].items():
            base = per[ref].get(rep)
            if not base:
                continue
            ds.append(sum(by_game[g]["score"] for g in games) / len(games) - sum(base[g]["score"] for g in games) / len(games))
            dl.append(sum(by_game[g]["levels"] for g in games) - sum(base[g]["levels"] for g in games))
        out.append({"label": label, "dscore": statistics.fmean(ds), "dscore_ci": 1.96 * statistics.stdev(ds) / math.sqrt(len(ds)),
                    "dlevels": statistics.fmean(dl), "dlevels_ci": 1.96 * statistics.stdev(dl) / math.sqrt(len(dl)), "n": len(ds)})
    champion_levels = {g: statistics.fmean(per[ref][rep][g]["levels"] for rep in per[ref]) for g in games}
    return out, champion_levels


def local_run() -> dict | None:
    llm, brain = latest("*_llm_local"), latest("*_brain_swarm_200")
    if llm is None or not (llm / "runs.jsonl").exists() or brain is None:
        return None
    L = {r["game"]: r for r in read_jsonl(llm / "runs.jsonl")}
    B = {r["game"]: r for r in read_jsonl(brain / "runs.jsonl")}
    requests = read_jsonl(ROOT / ".claude" / "scratch" / "llm" / "local" / "requests.jsonl")
    return {"llm_levels": sum(r["levels"] for r in L.values()), "brain_levels": sum(r["levels"] for r in B.values()),
            "llm_score": statistics.fmean(r["score"] for r in L.values()),
            "brain_score": statistics.fmean(r["score"] for r in B.values()),
            "health": request_health(requests), "folder": llm.name}


def latest_answers() -> dict[str, dict]:
    """The most recent parsed answer per game from the local request log (needs the game tag)."""
    answers = {}
    for r in read_jsonl(ROOT / ".claude" / "scratch" / "llm" / "local" / "requests.jsonl"):
        if not r.get("game") or "answer" not in r:
            continue
        try:
            answers[r["game"].split("-")[0]] = json.loads(r["answer"])
        except (ValueError, TypeError):
            continue
    return answers


def first_screens() -> list[dict]:
    sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
    logging.disable(logging.CRITICAL)
    import arc_agi

    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files"))
    games = sorted(e.game_id.split("-")[0] for e in arc.get_environments())
    screens = []
    for game in games:
        card = arc.open_scorecard(tags=["dashboard"])
        frame = arc.make(game, scorecard_id=card).reset()
        arc.close_scorecard(card)
        screens.append({"game": game, "png": png_uri(np.asarray(frame.frame[-1], dtype=np.uint8)),
                        "actions": sorted(frame.available_actions)})
    return screens


def png_uri(grid: np.ndarray) -> str:
    from PIL import Image

    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in PALETTE], dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb[np.clip(grid, 0, 15)], "RGB").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def mode_of(actions: list[int]) -> str:
    keys, click = any(a in actions for a in (1, 2, 3, 4)), 6 in actions
    return "mixed" if keys and click else "mover" if keys else "clicker" if click else "other"


# ── drawing ──────────────────────────────────────────────────────────────────

E = html.escape


def health_bar(h: dict | None) -> str:
    """Stacked bar: usable answers / cut off at the token limit / failed requests."""
    if not h:
        return '<span class="muted">no request log</span>'
    total = max(1, h["requests"])
    parts = [("ok", h["stop"], "answered"), ("cut", h["length"], "cut off at token limit"), ("err", h["errors"], "failed / timed out")]
    segs = "".join(f'<span class="seg {c}" style="flex:{n}" title="{n} {t}"></span>' for c, n, t in parts if n)
    legend = " · ".join(f'<span class="dot {c}"></span>{n:,} {t}' for c, n, t in parts if n)
    return (f'<div class="health"><div class="bar">{segs}</div><div class="legend">{legend} '
            f'<span class="muted">of {total:,} · median {h["p50"]:.0f} s · p90 {h["p90"]:.0f} s · '
            f'{h["tokens_p50"]:.0f} tokens</span></div></div>')


def levels_chart(rows: list[dict]) -> str:
    """Grouped bars: levels per game, LLM vs brain, only games where either finished a level."""
    rows = [r for r in rows if r["llm_levels"] or r["brain_levels"]]
    if not rows:
        return ""
    w, bar, gap, h = 44, 14, 16, 120
    width = len(rows) * (w + gap) + 40
    top = max(max(r["llm_levels"], r["brain_levels"]) for r in rows)
    scale = (h - 20) / top
    out = [f'<svg viewBox="0 0 {width} {h + 40}" role="img" aria-label="Levels per game, LLM vs brain">']
    for lv in range(top + 1):
        y = h - lv * scale
        out.append(f'<line x1="30" x2="{width}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="24" y="{y + 4:.1f}" class="tick" text-anchor="end">{lv}</text>')
    for i, r in enumerate(rows):
        x = 40 + i * (w + gap)
        for j, (key, cls) in enumerate((("llm_levels", "llm"), ("brain_levels", "brain"))):
            v = r[key]
            out.append(f'<rect x="{x + j * (bar + 2)}" y="{h - v * scale:.1f}" width="{bar}" height="{v * scale:.1f}" class="{cls}"><title>{r["game"]} {cls}: {v} level(s)</title></rect>')
        out.append(f'<text x="{x + bar}" y="{h + 16}" class="tick" text-anchor="middle">{r["game"]}</text>')
    out.append("</svg>")
    return "".join(out)


def delta_chart(rows: list[dict]) -> str:
    """Δscore with 95% CI whiskers against zero."""
    if not rows:
        return ""
    width, row_h, left = 520, 34, 70
    lo = min(r["dscore"] - r["dscore_ci"] for r in rows)
    hi = max(r["dscore"] + r["dscore_ci"] for r in rows)
    lo, hi = min(lo, -0.02) * 1.15, max(hi, 0.02) * 1.15
    x = lambda v: left + (v - lo) / (hi - lo) * (width - left - 20)  # noqa: E731
    h = len(rows) * row_h + 30
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="Score change vs champion, 95% CI">',
           f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="4" y2="{h - 22}" class="zero"/>',
           f'<text x="{x(0):.1f}" y="{h - 6}" class="tick" text-anchor="middle">0</text>',
           f'<text x="{x(lo):.1f}" y="{h - 6}" class="tick">{lo:+.2f}</text>',
           f'<text x="{x(hi):.1f}" y="{h - 6}" class="tick" text-anchor="end">{hi:+.2f}</text>']
    for i, r in enumerate(rows):
        y = 18 + i * row_h
        worse = r["dscore"] + r["dscore_ci"] < 0
        cls = "bad" if worse else "neutral"
        out.append(f'<text x="{left - 10}" y="{y + 4}" class="label" text-anchor="end">{E(r["label"])}</text>'
                   f'<line x1="{x(r["dscore"] - r["dscore_ci"]):.1f}" x2="{x(r["dscore"] + r["dscore_ci"]):.1f}" y1="{y}" y2="{y}" class="ci {cls}"/>'
                   f'<circle cx="{x(r["dscore"]):.1f}" cy="{y}" r="5" class="pt {cls}"><title>{r["dscore"]:+.3f} ± {r["dscore_ci"]:.3f}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def build() -> str:
    runs = kaggle_runs()
    versus = v4_versus_brain()
    ablation, champion_levels = step1_ablation()
    local = local_run()
    answers = latest_answers()
    screens = first_screens()
    v4_by_game = {r["game"]: r for r in versus}

    run_rows = []
    for r in runs:
        arms = "".join(
            f'<div class="arm"><div class="arm-head"><b>{E(a["name"])}</b> · {a["minutes"]} min · '
            f'{a["levels"]} levels · score {a["score"]:.3f} · LLM chose {a["llm_actions"]:,} of {a["all_actions"]:,} actions</div>'
            f'{health_bar(a["health"])}</div>' for a in r["arms"])
        server = f'server up in {r["server_s"]} s' if r["server_s"] else ""
        run_rows.append(
            f'<article class="run {r["status"]}"><div class="run-id">{r["version"]}</div><div class="run-body">'
            f'<div class="run-title"><span class="pill {r["status"]}">{r["status"]}</span> {E(r["headline"])} '
            f'<span class="muted">{server}</span></div>'
            f'<p><span class="k">What happened</span> {E(r["cause"])}</p><p><span class="k">Fix</span> {E(r["fix"])}</p>{arms}</div></article>')

    tot_llm = sum(r["llm_levels"] for r in versus)
    tot_brain = sum(r["brain_levels"] for r in versus)
    s_llm = statistics.fmean(r["llm_score"] for r in versus) if versus else 0
    s_brain = statistics.fmean(r["brain_score"] for r in versus) if versus else 0

    cards = []
    for s in screens:
        g = s["game"]
        ans = answers.get(g)
        v4 = v4_by_game.get(g)
        facts = [f'<span class="chip">{mode_of(s["actions"])}</span>',
                 f'<span class="chip ghost">{" ".join("A" + str(a) for a in s["actions"])}</span>']
        stats = []
        if g in champion_levels:
            stats.append(f'brain (N=30, 1000 actions): <b>{champion_levels[g]:.2f}</b> levels')
        if v4:
            stats.append(f'Kaggle v4 at {v4["actions"]} actions: LLM <b>{v4["llm_levels"]}</b> · brain <b>{v4["brain_levels"]}</b>')
        said = (f'<blockquote><span class="k">LLM (local 4B) thinks</span> {E(ans.get("theory", ""))}'
                f'<code>{E(" ".join(ans.get("plan", [])))}</code></blockquote>') if ans else ""
        cards.append(f'<figure class="game"><img src="{s["png"]}" alt="first screen of {g}" width="128" height="128">'
                     f'<figcaption><div class="gname">{g}</div><div class="chips">{"".join(facts)}</div>'
                     f'<div class="gstats">{"<br>".join(stats)}</div>{said}</figcaption></figure>')

    local_html = ""
    if local:
        local_html = (f'<p>Latest local run <span class="muted">({E(local["folder"])}, 200 actions per game, all 25 games at once)</span>: '
                      f'LLM <b>{local["llm_levels"]}</b> levels / score {local["llm_score"]:.3f} vs brain '
                      f'<b>{local["brain_levels"]}</b> / {local["brain_score"]:.3f}.</p>{health_bar(local["health"])}')

    return TEMPLATE.format(
        runs="".join(run_rows),
        versus_chart=levels_chart(versus),
        tot_llm=tot_llm, tot_brain=tot_brain, s_llm=s_llm, s_brain=s_brain,
        ablation_chart=delta_chart(ablation),
        ablation_rows="".join(f'<tr><td>{E(a["label"])}</td><td class="num">{a["dscore"]:+.3f} ± {a["dscore_ci"]:.3f}</td>'
                              f'<td class="num">{a["dlevels"]:+.2f} ± {a["dlevels_ci"]:.2f}</td><td class="num">{a["n"]}</td></tr>'
                              for a in ablation),
        local=local_html or '<p class="muted">No local LLM run with results yet.</p>',
        cards="".join(cards),
    )


TEMPLATE = """<title>ARC-AGI-3 Run Board</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Pixelify+Sans:wght@500;700&family=Figtree:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root {{
  --ground: #F4F6F9; --panel: #FFFFFF; --ink: #1B2130; --muted: #5F6878; --line: #DDE2EA;
  --accent: #1E93FF; --llm: #1E93FF; --brain: #A356D6; --ok: #2E9E1C; --cut: #E0A800; --err: #D42C22;
  --shadow: 0 1px 2px rgba(20, 30, 50, .06);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground: #0E121B; --panel: #161B27; --ink: #E6EAF2; --muted: #98A2B5; --line: #283042;
    --accent: #4AA8FF; --llm: #4AA8FF; --brain: #B77BE6; --ok: #5BD24A; --cut: #FFDC00; --err: #FF5A50;
    --shadow: none; color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --ground: #0E121B; --panel: #161B27; --ink: #E6EAF2; --muted: #98A2B5; --line: #283042;
  --accent: #4AA8FF; --llm: #4AA8FF; --brain: #B77BE6; --ok: #5BD24A; --cut: #FFDC00; --err: #FF5A50;
  --shadow: none; color-scheme: dark;
}}
body {{ background: var(--ground); color: var(--ink); font: 15px/1.55 Figtree, system-ui, sans-serif; padding-inline: 16px; padding-block: 24px 64px; }}
main {{ max-width: 1120px; margin: 0 auto; display: grid; gap: 40px; }}
h1 {{ font: 700 clamp(28px, 4vw, 40px)/1.1 "Pixelify Sans", Figtree, sans-serif; margin: 0; letter-spacing: .01em; color: var(--ink); }}
h2 {{ font: 700 20px/1.2 Figtree, sans-serif; margin: 0 0 4px; text-wrap: balance; color: var(--ink); }}
.lede {{ color: var(--muted); max-width: 70ch; margin: 8px 0 0; }}
section > p.note {{ color: var(--muted); margin: 0 0 16px; max-width: 75ch; }}
.muted {{ color: var(--muted); }}
.k {{ font: 600 11px/1 Figtree, sans-serif; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-right: 6px; }}
.num, code, .legend, .arm-head {{ font-family: "JetBrains Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
.runs {{ display: grid; gap: 12px; }}
.run {{ display: grid; grid-template-columns: 56px 1fr; gap: 16px; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; box-shadow: var(--shadow); }}
.run-id {{ font: 700 22px/1 "Pixelify Sans", sans-serif; color: var(--muted); }}
.run-title {{ font-weight: 600; color: var(--ink); display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; }}
.run p {{ margin: 6px 0 0; }}
.pill {{ font: 600 11px/1 Figtree, sans-serif; text-transform: uppercase; letter-spacing: .06em; padding: 4px 8px; border-radius: 99px; border: 1px solid currentColor; }}
.pill.failed {{ color: var(--err); }} .pill.measured {{ color: var(--ok); }} .pill.running {{ color: var(--cut); }}
.arm {{ margin-top: 12px; padding-top: 12px; border-top: 1px dashed var(--line); }}
.arm-head {{ font-size: 13px; }}
.health {{ margin-top: 8px; }}
.bar {{ display: flex; height: 12px; border-radius: 3px; overflow: hidden; background: var(--line); }}
.seg.ok, .dot.ok {{ background: var(--ok); }} .seg.cut, .dot.cut {{ background: var(--cut); }} .seg.err, .dot.err {{ background: var(--err); }}
.legend {{ font-size: 12px; margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; }}
.dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; box-shadow: var(--shadow); }}
.chart {{ overflow-x: auto; }}
svg {{ display: block; min-width: 520px; width: 100%; height: auto; }}
svg .grid {{ stroke: var(--line); }} svg .zero {{ stroke: var(--muted); stroke-dasharray: 3 3; }}
svg .tick, svg .label {{ fill: var(--muted); font: 11px "JetBrains Mono", monospace; }}
svg .label {{ fill: var(--ink); font-size: 12px; }}
svg rect.llm {{ fill: var(--llm); }} svg rect.brain {{ fill: var(--brain); }}
svg .ci {{ stroke-width: 3; stroke-linecap: round; }} svg .ci.bad {{ stroke: var(--err); }} svg .ci.neutral {{ stroke: var(--muted); }}
svg .pt.bad {{ fill: var(--err); }} svg .pt.neutral {{ fill: var(--ink); }}
.key {{ display: flex; gap: 16px; font-size: 13px; margin-bottom: 8px; flex-wrap: wrap; }}
.key i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
.totals {{ display: flex; gap: 24px; flex-wrap: wrap; margin-top: 12px; }}
.totals div {{ min-width: 150px; }}
.totals b {{ font: 600 26px/1.1 "JetBrains Mono", monospace; display: block; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
th {{ font: 600 11px/1 Figtree, sans-serif; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }}
td.num {{ text-align: right; }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
.game {{ margin: 0; display: grid; grid-template-columns: 128px 1fr; gap: 12px; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; align-items: start; }}
.game img {{ image-rendering: pixelated; width: 128px; height: 128px; border-radius: 4px; border: 1px solid var(--line); }}
.gname {{ font: 700 18px/1 "Pixelify Sans", sans-serif; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }}
.chip {{ font: 600 11px/1 Figtree, sans-serif; padding: 3px 7px; border-radius: 4px; background: var(--accent); color: #fff; }}
.chip.ghost {{ background: transparent; color: var(--muted); border: 1px solid var(--line); font-family: "JetBrains Mono", monospace; font-weight: 400; }}
.gstats {{ font-size: 12px; color: var(--muted); }}
.gstats b {{ color: var(--ink); }}
blockquote {{ grid-column: 1 / -1; margin: 8px 0 0; font-size: 13px; }}
blockquote code {{ display: block; margin-top: 4px; font-size: 11px; color: var(--muted); overflow-wrap: anywhere; }}
@media (max-width: 480px) {{ .run {{ grid-template-columns: 1fr; gap: 6px; }} .game {{ grid-template-columns: 96px 1fr; }} .game img {{ width: 96px; height: 96px; }} }}
</style>
<main>
<header>
  <h1>ARC-AGI-3 Run Board</h1>
  <p class="lede">Every run so far with the local LLM advisor (Qwen3.8-27B on Kaggle's RTX PRO 6000, Qwen3.5-4B on the laptop) against
  the rule-based brain. Built by <code>eval/dashboard.py</code> from the run files; nothing here is typed in by hand except the one-line
  run stories.</p>
</header>

<section>
  <h2>Kaggle GPU runs</h2>
  <p class="note">Each bar is one run's LLM answers: green = a usable answer, yellow = the model was still writing when it hit its token limit (no plan),
  red = request failed or timed out.</p>
  <div class="runs">{runs}</div>
</section>

<section>
  <h2>LLM vs brain, same number of actions (Kaggle v4, no thinking)</h2>
  <p class="note">Each game replayed by the brain alone with the same seed and exactly the actions the LLM run used (337–691). One seed: a sign, not a result.
  Games where neither finished a level are left out.</p>
  <div class="panel">
    <div class="key"><span><i style="background:var(--llm)"></i>LLM (27B, thinking off)</span><span><i style="background:var(--brain)"></i>brain only</span></div>
    <div class="chart">{versus_chart}</div>
    <div class="totals"><div><span class="k">LLM</span><b>{tot_llm} levels</b><span class="muted num">score {s_llm:.3f}</span></div>
    <div><span class="k">Brain</span><b>{tot_brain} levels</b><span class="muted num">score {s_brain:.3f}</span></div></div>
  </div>
</section>

<section>
  <h2>Local 4B model (RTX 4070 laptop)</h2>
  <p class="note">For development only: the 4B model plays worse than the 27B one; these runs check plumbing, speed and answer health.</p>
  <div class="panel">{local}</div>
</section>

<section>
  <h2>PLAYABLES step 1 (rejected)</h2>
  <p class="note">Score change vs the champion, N=30 paired reps, all 25 games, 1000 actions; bars are 95% intervals. Red = measurably worse. Both behaviours were deleted.</p>
  <div class="panel">
    <div class="chart">{ablation_chart}</div>
    <table><thead><tr><th>variant</th><th class="num">Δ score</th><th class="num">Δ levels</th><th class="num">reps</th></tr></thead><tbody>{ablation_rows}</tbody></table>
  </div>
</section>

<section>
  <h2>The 25 public games, as the agent first sees them</h2>
  <p class="note">First screen of each game, the buttons it offers (A1–A7) and the mode they imply, how the brain does on it, and — when the local run logged it — what the LLM thought and planned.</p>
  <div class="gallery">{cards}</div>
</section>
</main>
"""


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

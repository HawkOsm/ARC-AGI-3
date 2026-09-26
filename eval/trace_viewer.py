"""Agent Replay: a page that plays back recorded games step by step (what the agent saw, did, and why).

Record traces first (the agent writes them when ARC3_TRACE_DIR is set), e.g.
    ARC3_TRACE_DIR=.claude/scratch/trace/brain_300 .venv/bin/python eval/run.py --swarm --max-actions 300
then build the page from one or more trace folders (LABEL=FOLDER):
    .venv/bin/python eval/trace_viewer.py "brain, 300 actions=.claude/scratch/trace/brain_300"
Writes .claude/scratch/dashboard/agent-replay.html (self-contained; frames decode in the browser).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".claude" / "scratch" / "dashboard" / "agent-replay.html"
PALETTE = ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000", "#E53AA3", "#FF7BCC",
           "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00", "#FF851B", "#921231", "#4FCC30", "#A356D6"]


def load(folder: Path) -> dict[str, list[dict]]:
    games = {}
    for path in sorted(folder.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if rows:
            games[path.stem] = rows
    return games


def build(runs: list[tuple[str, Path]]) -> str:
    data = [{"label": label, "games": load(folder)} for label, folder in runs]
    payload = json.dumps({"palette": PALETTE, "runs": data}, separators=(",", ":"))
    return TEMPLATE.replace("__DATA__", payload.replace("</", "<\\/"))


TEMPLATE = r"""<meta charset="utf-8">
<title>Agent Replay</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Pixelify+Sans:wght@600&family=Figtree:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root {
  --ground: #EEF1F5; --panel: #FFFFFF; --ink: #1B2130; --muted: #5E6778; --line: #D9DFE8; --accent: #1E93FF;
  --explore: #1E93FF; --walk: #88D8F1; --replay: #4FCC30; --llm: #E53AA3; --reset: #FF851B; --random: #999999; --other: #CCCCCC;
  --death: #F93C31; --levelup: #2E9E1C; --changed: rgba(229, 58, 163, .9);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0D1119; --panel: #161B26; --ink: #E6EAF2; --muted: #98A2B5; --line: #283042; --accent: #4AA8FF;
    --other: #3A4254; --random: #6B7385; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --ground: #0D1119; --panel: #161B26; --ink: #E6EAF2; --muted: #98A2B5; --line: #283042; --accent: #4AA8FF;
  --other: #3A4254; --random: #6B7385; color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--ground); color: var(--ink); font: 14px/1.5 Figtree, system-ui, sans-serif; padding-inline: 16px; padding-block: 16px 40px; }
.app { max-width: 1180px; margin: 0 auto; display: grid; gap: 14px; }
header { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: center; }
h1 { font: 600 28px/1 "Pixelify Sans", Figtree, sans-serif; margin: 0; color: var(--ink); }
select, button { font: inherit; color: var(--ink); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px; }
button { cursor: pointer; min-width: 38px; }
button:hover, select:hover { border-color: var(--accent); }
button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
.games { display: flex; flex-wrap: wrap; gap: 6px; }
.games button { font: 600 13px/1 "JetBrains Mono", monospace; padding: 6px 8px; display: inline-flex; gap: 6px; align-items: center; }
.games button small { font-weight: 400; color: inherit; opacity: .7; }
.main { display: grid; grid-template-columns: minmax(0, 560px) minmax(0, 1fr); gap: 14px; align-items: start; }
.stage { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; display: grid; gap: 10px; }
.screen { position: relative; width: 100%; aspect-ratio: 1; max-width: 100%; }
.screen canvas { position: absolute; inset: 0; width: 100%; height: 100%; image-rendering: pixelated; border-radius: 4px; }
.controls { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.controls input[type=range] { flex: 1 1 160px; accent-color: var(--accent); }
.controls label { color: var(--muted); font-size: 12px; display: inline-flex; gap: 4px; align-items: center; }
.timeline { width: 100%; height: 34px; cursor: pointer; border-radius: 4px; display: block; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 12px; color: var(--muted); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.side { display: grid; gap: 10px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.k { font: 600 11px/1 Figtree, sans-serif; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); display: block; margin-bottom: 6px; }
.facts { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
.facts b { display: block; font: 600 20px/1.1 "JetBrains Mono", monospace; font-variant-numeric: tabular-nums; }
.facts span { color: var(--muted); font-size: 12px; }
.act { font: 600 18px/1.2 "JetBrains Mono", monospace; }
.chip { display: inline-block; font: 600 11px/1 Figtree, sans-serif; padding: 4px 7px; border-radius: 4px; color: #fff; margin-right: 6px; vertical-align: 2px; }
.why { margin-top: 6px; }
.result { font-family: "JetBrains Mono", monospace; font-size: 13px; }
.result.bad { color: var(--death); } .result.good { color: var(--levelup); font-weight: 600; } .result.none { color: var(--muted); }
.theory { font-size: 14px; }
.theory small { color: var(--muted); display: block; margin-top: 4px; font-size: 12px; }
.muted { color: var(--muted); }
.swatch { display: inline-block; width: 12px; height: 12px; border: 1px solid var(--line); vertical-align: -1px; margin-right: 4px; }
@media (max-width: 820px) { .main { grid-template-columns: 1fr; } .facts { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }
</style>
<div class="app">
  <header>
    <h1>Agent Replay</h1>
    <label class="muted" for="run">Run</label><select id="run"></select>
    <span class="muted" id="summary"></span>
  </header>
  <div class="games" id="games" role="tablist" aria-label="Games"></div>
  <div class="main">
    <div class="stage">
      <div class="screen"><canvas id="frame" width="64" height="64" aria-label="game screen"></canvas><canvas id="overlay" width="640" height="640"></canvas></div>
      <div class="controls">
        <button id="first" title="First step (Home)">⏮</button>
        <button id="prev" title="Previous step (←)">◀</button>
        <button id="play" title="Play / pause (space)">▶</button>
        <button id="next" title="Next step (→)">▶|</button>
        <button id="event" title="Jump to the next death, level-up or new LLM plan (N)">next event</button>
        <input id="slider" type="range" min="0" max="0" value="0" aria-label="step">
        <label for="speed">speed <select id="speed"><option value="2">2/s</option><option value="5" selected>5/s</option><option value="12">12/s</option><option value="30">30/s</option></select></label>
        <label for="diff"><input id="diff" type="checkbox" checked> show what changed</label>
      </div>
      <canvas class="timeline" id="timeline" height="34" aria-label="timeline: who chose each action"></canvas>
      <div class="legend" id="legend"></div>
    </div>
    <div class="side">
      <div class="card"><div class="facts">
        <div><b id="f-step">0</b><span>step</span></div>
        <div><b id="f-level">0</b><span>levels done</span></div>
        <div><b id="f-life">0</b><span>steps this life</span></div>
        <div><b id="f-deaths">0</b><span>lives lost</span></div>
      </div></div>
      <div class="card"><span class="k">Action taken</span><div class="act" id="action"></div><div class="why" id="why"></div></div>
      <div class="card"><span class="k">What it did</span><div class="result" id="result"></div></div>
      <div class="card"><span class="k">What the brain knows</span><div id="brain"></div></div>
      <div class="card"><span class="k">LLM theory</span><div class="theory" id="theory"></div></div>
      <div class="card" id="check-card" hidden><span class="k">Last test of the LLM's prediction</span><div class="result" id="check"></div></div>
    </div>
  </div>
</div>
<script>
const DATA = __DATA__;
const SOURCES = [
  ["llm", "LLM plan", r => r.why.startsWith("LLM")],
  ["replay", "replaying a solved level", r => r.why.startsWith("replay")],
  ["walk", "walking to a frontier", r => r.why.startsWith("explore: walk")],
  ["explore", "trying something new here", r => r.why.startsWith("explore")],
  ["reset", "reset / game over", r => /^(reset|game over|start|the whole game)/.test(r.why)],
  ["random", "random fallback", r => r.why.startsWith("random")],
  ["other", "other", r => true],
];
const source = r => SOURCES.find(s => s[2](r))[0];
const css = name => getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim();
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
let run = 0, game = null, rows = [], frames = [], step = 0, timer = null;

async function inflate(b64) {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
async function loadGame(name) {
  stop();
  game = name; rows = DATA.runs[run].games[name]; frames = [];
  let last = new Uint8Array(4096);
  for (const r of rows) { if (r.grid) last = await inflate(r.grid); frames.push(last); }
  $("slider").max = rows.length - 1; step = 0;
  document.querySelectorAll("#games button").forEach(b => { b.classList.toggle("on", b.dataset.game === name); b.setAttribute("aria-selected", b.dataset.game === name); });
  drawTimeline(); show(0);
}
function isDeath(r) { return (r.outcome || "").includes("lost a life"); }
function isLevelUp(r) { return (r.outcome || "").includes("LEVEL COMPLETE"); }
function show(i) {
  step = Math.max(0, Math.min(rows.length - 1, i));
  const r = rows[step], nextRow = rows[step + 1];
  $("slider").value = step;
  drawFrame(frames[step], nextRow ? frames[step + 1] : null, r.action);
  $("f-step").textContent = `${step}/${rows.length - 1}`;
  $("f-level").textContent = r.level ?? 0; $("f-life").textContent = r.life; $("f-deaths").textContent = r.deaths;
  const [id, x, y] = r.action;
  $("action").textContent = id === 0 ? "RESET" : id === 6 ? `ACTION6  click (x ${x}, y ${y})` : `ACTION${id}`;
  const s = source(r); const label = SOURCES.find(t => t[0] === s)[1];
  $("why").innerHTML = `<span class="chip" style="background:${css(s)}">${esc(label)}</span>${esc(r.why || "—")}`;
  const res = nextRow ? (nextRow.outcome || "") : "(end of recording)";
  const text = res.replace(/^\S+( \d+ \d+)? -> /, "");
  const cls = /lost a life/.test(res) ? "bad" : /LEVEL COMPLETE/.test(res) ? "good" : /nothing changed/.test(res) ? "none" : "";
  $("result").className = "result " + cls; $("result").textContent = text || "—";
  const colour = r.avatar;
  $("brain").innerHTML = `game mode <b>${esc(r.mode || "?")}</b> · screens mapped on this level <b>${r.screens}</b><br>` +
    `it thinks it controls: ${colour == null ? "<span class='muted'>not known yet</span>" : `<span class="swatch" style="background:${DATA.palette[colour]}"></span>colour ${colour}`}`;
  let t = null, at = null;
  for (let j = step; j >= 0; j--) if (rows[j].theory) { t = rows[j].theory; at = j; break; }
  $("theory").innerHTML = t ? `${esc(t)}<small>received at step ${at}</small>` : "<span class='muted'>No LLM in this run, or it has not answered yet.</span>";
  let ck = null, ckAt = null;
  for (let j = step; j >= 0; j--) if (rows[j].check) { ck = rows[j].check; ckAt = j; break; }
  $("check-card").hidden = !rows.some(r => r.check);
  $("check").className = "result " + (ck && ck.startsWith("CORRECT") ? "good" : ck ? "bad" : "none");
  $("check").textContent = ck ? `${ck}  (judged at step ${ckAt})` : "no test judged yet";
  drawCursor();
}
function drawFrame(f, next, action) {
  const c = $("frame").getContext("2d"), img = c.createImageData(64, 64);
  const rgb = DATA.palette.map(h => [1, 3, 5].map(k => parseInt(h.slice(k, k + 2), 16)));
  for (let k = 0; k < 4096; k++) { const p = rgb[f[k] & 15]; img.data.set([p[0], p[1], p[2], 255], k * 4); }
  c.putImageData(img, 0, 0);
  const o = $("overlay").getContext("2d"), s = 10;
  o.clearRect(0, 0, 640, 640);
  if (next && $("diff").checked) {
    o.strokeStyle = css("changed"); o.lineWidth = 2;
    for (let k = 0; k < 4096; k++) if (f[k] !== next[k]) o.strokeRect((k % 64) * s + 1, Math.floor(k / 64) * s + 1, s - 2, s - 2);
  }
  const [id, x, y] = action;
  o.font = "600 28px 'JetBrains Mono', monospace"; o.textBaseline = "top";
  if (id === 6) {
    const cx = x * s + s / 2, cy = y * s + s / 2;
    o.lineWidth = 4; o.strokeStyle = "#000"; o.beginPath(); o.arc(cx, cy, 16, 0, Math.PI * 2); o.stroke();
    o.lineWidth = 2; o.strokeStyle = "#fff"; o.beginPath(); o.arc(cx, cy, 16, 0, Math.PI * 2); o.stroke();
    o.beginPath(); o.moveTo(cx - 26, cy); o.lineTo(cx + 26, cy); o.moveTo(cx, cy - 26); o.lineTo(cx, cy + 26); o.stroke();
  } else {
    const label = id === 0 ? "RESET" : "A" + id;
    const w = o.measureText(label).width + 20;
    o.fillStyle = "rgba(0,0,0,.72)"; o.fillRect(10, 10, w, 42); o.fillStyle = "#fff"; o.fillText(label, 20, 17);
  }
}
function drawTimeline() {
  const cv = $("timeline"), w = cv.clientWidth * devicePixelRatio, h = 34 * devicePixelRatio;
  cv.width = w; cv.height = h;
  const c = cv.getContext("2d"), n = rows.length, bw = w / n;
  c.clearRect(0, 0, w, h);
  rows.forEach((r, i) => { c.fillStyle = css(source(r)); c.fillRect(i * bw, h * .3, Math.max(1, bw), h * .7); });
  rows.forEach((r, i) => {
    if (isDeath(r)) { c.fillStyle = css("death"); c.fillRect(i * bw - 1, 0, Math.max(2, bw), h * .3); }
    if (isLevelUp(r)) { c.fillStyle = css("levelup"); c.fillRect(i * bw - 1, 0, Math.max(3, bw), h); }
  });
  cv._base = c.getImageData(0, 0, w, h);
}
function drawCursor() {
  const cv = $("timeline"), c = cv.getContext("2d");
  if (!cv._base) return;
  c.putImageData(cv._base, 0, 0);
  const x = (step + .5) * cv.width / rows.length;
  c.fillStyle = css("ink"); c.fillRect(x - 1.5 * devicePixelRatio, 0, 3 * devicePixelRatio, cv.height);
}
function play() {
  if (timer) return stop();
  $("play").textContent = "⏸";
  timer = setInterval(() => { if (step >= rows.length - 1) return stop(); show(step + 1); }, 1000 / +$("speed").value);
}
function stop() { clearInterval(timer); timer = null; $("play").textContent = "▶"; }
function nextEvent() {
  for (let j = step + 1; j < rows.length; j++) {
    const r = rows[j];
    if (isDeath(r) || isLevelUp(r) || r.theory) return show(Math.max(0, j - 1));
  }
  show(rows.length - 1);
}
function buildGames() {
  const box = $("games"); box.innerHTML = "";
  const games = DATA.runs[run].games;
  let total = 0, llm = 0, levels = 0;
  for (const [name, rs] of Object.entries(games)) {
    const last = rs[rs.length - 1], lv = Math.max(...rs.map(r => r.level || 0));
    const deaths = last.deaths, share = rs.filter(r => source(r) === "llm").length;
    total += rs.length; llm += share; levels += lv;
    const b = document.createElement("button");
    b.dataset.game = name; b.setAttribute("role", "tab");
    b.innerHTML = `${name}<small>${lv}L ${deaths}✝</small>`;
    b.title = `${rs.length} steps, ${lv} level(s) done, ${deaths} lives lost, ${share} LLM steps`;
    b.onclick = () => loadGame(name);
    box.appendChild(b);
  }
  $("summary").textContent = `${Object.keys(games).length} games · ${levels} levels finished · ${total.toLocaleString()} steps` + (llm ? ` · ${Math.round(100 * llm / total)}% chosen by the LLM` : "");
  loadGame(Object.keys(games)[0]);
}
$("legend").innerHTML = SOURCES.filter(s => s[0] !== "other").map(s => `<span><i style="background:var(--${s[0]})"></i>${s[1]}</span>`).join("")
  + `<span><i style="background:var(--death)"></i>life lost</span><span><i style="background:var(--levelup)"></i>level finished</span>`;
DATA.runs.forEach((r, i) => { const o = document.createElement("option"); o.value = i; o.textContent = r.label; $("run").appendChild(o); });
$("run").onchange = e => { run = +e.target.value; buildGames(); };
$("slider").oninput = e => { stop(); show(+e.target.value); };
$("first").onclick = () => { stop(); show(0); };
$("prev").onclick = () => { stop(); show(step - 1); };
$("next").onclick = () => { stop(); show(step + 1); };
$("play").onclick = play;
$("event").onclick = () => { stop(); nextEvent(); };
$("speed").onchange = () => { if (timer) { stop(); play(); } };
$("diff").onchange = () => show(step);
$("timeline").onclick = e => { stop(); const r = e.target.getBoundingClientRect(); show(Math.floor((e.clientX - r.left) / r.width * rows.length)); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT" && e.target.type !== "range") return;
  if (e.key === "ArrowRight") { stop(); show(step + 1); e.preventDefault(); }
  else if (e.key === "ArrowLeft") { stop(); show(step - 1); e.preventDefault(); }
  else if (e.key === " ") { play(); e.preventDefault(); }
  else if (e.key === "Home") { stop(); show(0); }
  else if (e.key === "End") { stop(); show(rows.length - 1); }
  else if (e.key === "n" || e.key === "N") { stop(); nextEvent(); }
});
addEventListener("resize", () => { if (rows.length) { drawTimeline(); drawCursor(); } });
buildGames();
</script>
"""


def main() -> int:
    runs = []
    for arg in sys.argv[1:] or ["brain=.claude/scratch/trace/brain_300"]:
        label, _, folder = arg.rpartition("=")
        runs.append((label or Path(folder).name, ROOT / folder))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(runs))
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

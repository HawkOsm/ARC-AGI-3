# Test Log Convention

Every local test outcome — an eval run, a correctness check, a before/after comparison — gets an
entry in [`LOG.md`](LOG.md), newest first (right after the `---` at the top). Project rule (see
`../../AGENT.md`), carried over from Kaggriculture: it is what separates "we measured this" from
"we assume this" weeks later, when the numbers decide what gets a daily Kaggle submission.

**Before trying a strategic idea, check [`IDEAS_TRIED.md`](IDEAS_TRIED.md)** — the concept-indexed
"have we already tried this?" ledger. Add a row there the moment an idea gets a real verdict.

## Run artifacts

`eval/run.py` writes `eval/results/<stamp>_<label>/` (gitignored): `runs.jsonl` (one row per
game×rep: official score, levels, per-level actions, human baselines, crash text, agent
diagnostics), `meta.json` (exact command), `summary.md` (the tables). Quote numbers from
`summary.md`; for anything else, **script a summary from `runs.jsonl`** — never read a raw trace
into context.

## Entry format

```markdown
## YYYY-MM-DD — short title
- Change: what changed (file/flag), or "none — measurement only"
- Command: `...`
- Result: raw numbers (aggregate ± 95% CI, n; per-game deltas that matter; crashes)
- Verdict: PROMOTED / REJECTED / NOISE / INFO — one or two sentences on what this rules in or out
```

## Statistics rules

- **N ≥ 30 reps** before trusting a close result. Aggregate = mean over games of the official
  per-game score, computed per rep; CI = 1.96·sd/√n over reps.
- Paired mode (`--champion`) runs both agents on identical (game, rep, seed) triples; the paired
  diff CI is the promotion statistic. A diff inside its CI is logged as **NOISE**, not as a win.
- The public set is 25 games; the scored set is ~4× larger and hidden. A gain concentrated in
  one or two games is a tuning result until shown not to be (check the per-game table: how many
  games moved, and did any regress?).
- Budget matters: note `--max-actions` in every entry. Results at different budgets are not
  comparable.

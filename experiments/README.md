# Experiments

One dated subfolder per approach: `YYYY-MM-DD_<idea>/` with a short `README.md` (hypothesis,
how the variant is built, result + LOG.md pointer) and, where applicable, the frozen
single-file agent (`my_agent.py`) that `eval/run.py --agent` / `--champion` can load directly.

Build a settings variant without touching `agent/brain/`:

    .venv/bin/python scripts/build_submission.py --set NAME=VALUE --out experiments/<dir>/my_agent.py

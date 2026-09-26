# duck-flashnext-hints

`notebooks/duck_repro` (the public Qwen3.8-Flash-Next Duck notebook, unchanged) plus ONE added cell: it puts the
dataset `hawkosm/arc3-duck-hints` (built by `scripts/build_duck_hints_dataset.py`) first on the import path and sets
`DUCK_BRAIN_HINTS=1`, so the model's Python tool also gets `brain_hints` (agent/duck/hints.py). Same model, serving,
budget and games as the repro run: the A/B test of our addition.

Push (dataset first, then the notebook):
    .venv/bin/kaggle datasets create -p kaggle_datasets/duck_hints            # later versions: datasets version -m ...
    .venv/bin/kaggle kernels push -p notebooks/duck_hints

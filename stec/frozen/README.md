# Frozen originals

Everything under this directory is a byte-identical copy of a pre-rebuild `src/` script
(verified by checksum at relocation time — see the git log for the commit that moved them).
**Do not port, refactor, reformat, or "modernise" anything here.** That is not an oversight
to fix; it is the property that makes these two files worth anything.

## Why these two specifically stay unported

- `analysis/repair_gim_baseline.py` — the regression check for the GIM day-lookup repair
  (CLAUDE.md, R1.4). It recomputes the IGS GIM baseline independently and asserts that
  unaffected days reproduce the store's stored `gim_stec` column to ~1e-5 TECU. If this
  script were rewritten to share code with `stec/baselines/gim.py` or
  `stec/inference/prediction_store.py` — the ported, actively-maintained implementations
  that produced the values being checked — a bug reintroduced in the shared code would move
  both sides of the comparison together and the check would silently stop checking anything.
  Its independence is the entire point, permanently, not just until porting catches up.
- `analysis/hyperparameter_search_summary.py` — not excluded for independence reasons, only
  a data one: it reads local `wandb/run-*/files/{config.yaml,wandb-summary.json}` pairs
  (~606 MB, gitignored), which do not exist in this worktree, a fresh clone, or anywhere
  except the original training host. Porting it would be mechanical (it already has zero
  `src/` dependencies beyond its own arguments), but there is nothing here to port it
  *against* — it stays frozen because its input is unreachable, not because the check
  reasoning above applies to it too.

## `evaluation/` — why it came along

`repair_gim_baseline.py` does `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
and then `from evaluation import prediction_store` / `from evaluation.gim_mapper import
GIMMapper` — it expects a sibling `evaluation/` package two directories up, exactly the
`src/analysis/` + `src/evaluation/` layout it was written against. This directory mirrors
that layout (`frozen/analysis/` + `frozen/evaluation/`) so the script needed zero source
changes to run from its new home. `evaluation/__init__.py`, `gim_mapper.py` and
`prediction_store.py` are the pre-rebuild originals, copied verbatim for the same
independence reason as the script itself: swapping in the ported `stec/baselines/gim.py` or
`stec/inference/prediction_store.py` here would reintroduce exactly the shared-implementation
problem this whole freeze exists to avoid. Only these three files were copied (not the rest
of `src/evaluation/` — `plotter.py`, `publication_plots.py`, `utils.py`,
`madrigal_builder.py`, `madrigal_loader.py`) because `repair_gim_baseline.py` never imports
them; `evaluation/__init__.py`'s lazy `__getattr__` only reaches `plotter`/`utils` when an
attribute like `STECPlotter` is actually accessed, which this script never does.

## How these run

Both are invoked as plain scripts (`python stec/frozen/analysis/<name>.py ...`), the same
way they always were — never as `-m` modules, never `import`ed by anything else in `stec/`.
`stec/pipeline/stages.py` declares the `repair_gim_baseline` and `hyperparameter_search`
stages with these paths. Neither file has an `__init__.py` above it inside `frozen/`, on the
same convention as the pure-data `stec/data/splits/` directory: this tree is not part of the
`stec` package's import namespace, it just lives inside `stec/` so the whole repository is
one checkout with no dependency left outside it.

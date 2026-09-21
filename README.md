# PNN_STEC

Probabilistic neural network modelling of **slant total electron content** (STEC) from GNSS
observations, with per-observation uncertainty.

The model predicts the integrated electron density along a receiver-to-satellite line of sight
**directly**, rather than estimating a vertical map and projecting it through a mapping
function. Every prediction carries an uncertainty split into aleatoric and epistemic parts, and
that uncertainty is used — not merely reported — as an observation weight in precise point
positioning.

This repository backs the paper *Probabilistic Machine Learning for Slant Total Electron
Content Modelling based on GNSS* (Rüegg, Mao, Pan, Orús Pérez, Soja).

## What it produces

Four approaches are evaluated on the same observations over a continuous multi-month test
period in 2024, held out by both station and date:

- **Direct STEC** — the model, fine-tuned per day.
- **Pretrained STEC** — the same architecture without daily fine-tuning.
- **VTEC + mapping** — a vertical-TEC neural baseline projected with a mapping function.
- **IGS GIM + mapping** — the operational global ionosphere maps, projected the same way.

Each is scored as a STEC prediction problem and, separately, by the positioning accuracy its
corrections achieve. Results are CSVs produced by declared pipeline stages; see
[Canonical results](#canonical-results) for where each lives.

## Repository map

| Path | Contents |
|---|---|
| `stec/` | The implementation: config, data, models, training, inference, baselines, analysis, viz, positioning, and a 36-stage pipeline |
| `cli.py` | Entry point for training, inference, comparison, maps and multi-day evaluation |
| `positioning/` | PPPx-based positioning evaluation and the geometry recovery tools |
| `config/` | Run configurations, one per model variant |
| `tests/` | 1024 tests mirroring the `stec/` layout |
| `verification/` | Equivalence gates and independent correctness checks |
| `docs/` | Manual, architecture and reproduction guide |
| `multiday_results/` | Analysis outputs, bucketed by the question they answer |
| `.pipeline/` | Provenance record: what produced each result, and from which inputs |

## Installation

Python 3.10 or newer.

```bash
git clone <repository-url> PNN_STEC
cd PNN_STEC
python -m venv env
source env/bin/activate
pip install -e ".[dev]"
```

`requirements.txt` is a `pip freeze` of the development host, not an install specification —
use `pyproject.toml` via the command above.

On any machine other than the original host, point the code at your data:

```bash
export STEC_DATA_ROOT=/path/to/iono/data     # external datasets
export STEC_REPO_DATA=/path/to/aggregated    # training splits, space-weather indices
```

Positioning evaluation additionally needs the external PPPx binary and a one-time runtime
library step; see the manual, chapter 2.

## Quick start

Two commands work with no external data at all, and together they are the fastest way to
confirm the delivery is intact:

```bash
python -m stec.pipeline status   # every declared stage, and whether it is up to date
pytest                           # 1024 tests against checked-in fixtures
```

`status` reads only declared inputs and prints why each stage would or would not run. It also
validates the stage registry at startup, so it fails loudly if two stages claim the same output
or the same canonical role.

Everything beyond that needs the datasets described in the manual, chapter 3.

```bash
python -m stec.pipeline run --only daily_metrics   # regenerate one result
python -m stec.cli metrics --dataset own           # a standalone read
python cli.py train --config config/config_BNN.yaml
```

## Canonical results

| What | Path under `multiday_results/` |
|---|---|
| STEC metrics, four approaches | `analyses/daily_metrics/pre_rebuild/summary.csv` |
| Positioning, full population | `analyses/positioning_summary/rebuilt/overall.csv` |
| Positioning, matched common set | `analyses/common_set_positioning/rebuilt/table5_common_set.csv` |
| Provenance index | `analyses/results_manifest/rebuilt/manifest.csv` |

Every one of these is written by a pipeline stage that records its inputs, its command and its
output digests in `.pipeline/<stage>.json`. That file, not this table, is the authority on what
produced a given number.

Result trees that have been superseded are marked with a `.superseded.json` sidecar rather than
deleted, so earlier configurations remain on the record.

## Documentation

| Document | Read it for |
|---|---|
| [docs/USER_MANUAL.md](docs/USER_MANUAL.md) | How the system is operated, end to end. **Start here** |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Internal design: layer boundaries, the stage contract, what each gate proves |
| [docs/REPRODUCING.md](docs/REPRODUCING.md) | Reproducing results from a clean clone, and what is verifiable without the data |
| [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) | Subcommand-by-subcommand reference for `cli.py` |
| [docs/MULTIDAY_FILE_STRUCTURE.md](docs/MULTIDAY_FILE_STRUCTURE.md) | Directory layout a multi-day evaluation produces |

## What is not in this delivery

| Not included | Why, and how to obtain it |
|---|---|
| Raw GNSS STEC database | Institutional dataset, several terabytes. Not redistributable here |
| Madrigal reference STEC | External archive, obtained from Madrigal directly |
| IGS global ionosphere maps | External IGS product, publicly available from IGS data centres |
| Trained checkpoints | Large binary artifacts; produced by the training workflow in the manual, chapter 5 |
| PPPx binary | External precise point positioning software, licensed separately |

The station and date splits **are** included, at `stec/data/splits/`, so the exact evaluation
population is part of the delivery even though the observations are not.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

**Keywords**: ionosphere, slant TEC, Bayesian neural networks, uncertainty quantification,
space weather, GNSS, precise point positioning.

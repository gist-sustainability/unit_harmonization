# Unit Harmonization

This repository contains the unit harmonization component of a retrieval-augmented
pipeline for extracting CO2 emissions from corporate sustainability reports. The
project was developed by the LMU SODA Lab and the Deutsche Bundesbank as part of
[Climatextract](https://github.com/gist-sustainability/climatextract/tree/main).

## Project Structure

```text
data/
	unit_normalization_dict.csv   Training and evaluation data
src/
	preprocessing.py              Feature engineering and data splitting
	build_and_fit_pipeline.py     Model construction, training, and search
	evaluation.py                 Metrics, reports, and visualizations
notebooks/
	01_main_results.ipynb         Experiments reported in the paper
	02_additional_experiments.ipynb  Model comparisons and searches
```

The implementation is intentionally kept close to the experimental notebook:
the notebook imports the reusable functions from the Python modules, while its
remaining cells contain the reported experiments and exploratory analyses.

## Setup

Python 3.11 or newer is required. Install the project and development tools with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Alternatively, install the dependencies listed in `pyproject.toml` with pip.

## Reproduce the Main Experiment

Open `notebooks/01_main_results.ipynb` in a Jupyter-compatible development
environment, select the configured Python 3.11-or-newer environment as the
notebook kernel, and run the cells from the project root. This can be done in
VS Code, Jupyter Notebook, JupyterLab, or another compatible IDE.

The main notebook reproduces the selected voting ensemble, the majority-class
baseline, ten-fold cross-validation, random-split error analysis, and rare-unit
robustness evaluation. Additional model comparisons and hyperparameter searches
are available in `notebooks/02_additional_experiments.ipynb`.

The main programmatic entry point is:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

from build_and_fit_pipeline import full_pipeline

results = full_pipeline(ml_model="voting_classifier")
```

`full_pipeline` returns the fitted pipeline, the label encoder, the train/test
split, and the complete evaluation dictionary used by the notebook.

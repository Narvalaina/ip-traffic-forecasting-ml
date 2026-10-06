# Reproducibility

## Scope

This repository is reproducible **up to the data-distribution boundary**.
Source code, selected metrics, manifests, reports, figures and environment
information are public; the UGR'16 dataset and aligned prediction payloads are
not.

Portable validation during repository reconstruction was performed without
training, reselection or metric regeneration.

## Reference portable environment

The public baseline is **CPython 3.12**. The historical WSL scientific
environment used as the primary version anchor was **Python 3.12.13**.

The reviewed direct dependencies are:

```text
# Reference portable environment for the public TFM repository.
# Baseline: CPython 3.12 (historical WSL reference: Python 3.12.13).
#
# The scientific campaigns also used Google Colab. In particular, GPU runs
# observed PyTorch 2.11.0+cu128 with CUDA 12.8 and cuDNN 91900. The public
# requirements keep the portable PyPI package name/version; runtime-specific GPU
# installation details are documented separately during PHASE 1F.

lightgbm==4.7.0
matplotlib==3.11.1
numpy==2.5.2
pandas==3.0.5
pyarrow==25.0.1
scikit-learn==1.9.0
scipy==1.18.0
statsmodels==0.14.6
torch==2.11.0
tqdm==4.68.4
xgboost-cpu==3.3.0
```

Create a local environment with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository does not publish a full historical `pip freeze`: only direct,
scientifically relevant dependencies are pinned.

## Historical WSL2 evidence

The main development workflow used Ubuntu under **WSL2**. Historical
environment inspection was performed read-only while reconstructing the public
repository.

The portable reference uses the reviewed package versions in
`requirements.txt`; it is not intended to clone every transient package from
the historical development environment.

## Historical Google Colab GPU evidence

Selected recurrent, Transformer and Phase G executions used Google Colab.
Observed runtime evidence includes:

```text
Python       3.13.15
PyTorch      2.11.0+cu128
CUDA         12.8
cuDNN        91900
NumPy        2.1.3
pandas       2.2.3
statsmodels  0.14.6
```

Observed Colab interpreter: Python 3.13.15. The general `requirements.txt` uses portable `torch==2.11.0` rather than forcing the Colab-specific `+cu128` build on every platform.

## Data setup

Follow [`../data/README.md`](../data/README.md). Dataset files must be acquired
separately and placed/generated locally.

The expected public state before data acquisition is
**`EXPECTED_DATA_ABSENCE`**, not a repository failure.

## Path portability

Historical absolute WSL/Colab paths were removed from executable code during
repository reconstruction. Runtime resolution is repository-relative, with
optional overrides through:

```text
TFM_PROJECT_ROOT
TFM_COLAB_WORKDIR
```

Sanitized placeholders may still appear in historical manifests/provenance,
where they document origin without exposing a personal machine path.

## Validation boundary

The public candidate was checked for:

- Python syntax;
- notebook structure and cleared outputs;
- direct dependency imports in the historical WSL reference environment;
- portable root/path resolution;
- JSON/CSV/PNG structural integrity;
- absence of redistributed dataset payloads;
- absence of personal machine paths, private hostnames and detected secrets.

Scientific campaigns were not rerun during GitHub reconstruction.

# ai_repo

[![format-lint](
    https://github.com/maximilienleclei/ai_repo/actions/workflows/format-lint.yaml/badge.svg?event=push)](
        https://github.com/maximilienleclei/ai_repo/actions/workflows/format-lint.yaml)
[![on-push-with-image](
    https://github.com/maximilienleclei/ai_repo/actions/workflows/on-push.yaml/badge.svg?event=push)](
        https://github.com/maximilienleclei/ai_repo/actions/workflows/on-push.yaml)
[![code style: black](
    https://img.shields.io/badge/code%20style-black-000000.svg)](
        https://github.com/psf/black)

## Examples

🔢 MNIST classification with a MLP

```python
python projects/classify_mnist/train.py task=mlp
```

🤸 Acrobot score optimization with neuroevolution

```python
python -m projects.ne_control_score task=acrobot
```

## Table of Contents

[👀 I. Introduction](#i-introduction)

[📖 II. Overview](#ii-overview)

[🌳 III. Rough repository tree](#rough-repository-tree)

## I. Introduction

This repo aims to:

**Reduce code & configuration boilerplate with:**
* [Hydra](https://github.com/facebookresearch/hydra) for task configuration.
* [Hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) for
[Hydra](https://github.com/facebookresearch/hydra) structured configuration
management.
* [Lightning](https://github.com/Lightning-AI/pytorch-lightning) for
[PyTorch](https://github.com/pytorch/pytorch) code.

**Simplify machine learning workflows:**
* Hyperparameter optimization with
[AutoRL-Sweepers](https://github.com/facebookresearch/how-to-autorl) &
[Optuna](https://hydra.cc/docs/plugins/optuna_sweeper).
* SLURM job definition, queuing and monitoring with
[Submitit](https://github.com/facebookincubator/submitit) through its
[Hydra Launcher plugin](https://hydra.cc/docs/plugins/submitit_launcher/).
* [Docker](https://www.docker.com/) / [Apptainer](https://apptainer.org/)
environment containerization for both regular & SLURM-based execution.
* Transition from regular execution to SLURM-based execution by only swapping
container technology and as little as a single
[Hydra](https://github.com/facebookresearch/hydra)
configuration field.

**Automate workspace & coding processes:**
* Package upgrades through
[Renovate](https://github.com/renovatebot/renovate).
* Docstring documentation generation with
[Sphinx](https://github.com/sphinx-doc/sphinx).
* Pre-commit formatting & linting hooks with
[pre-commit](https://pre-commit.com/).
* Documentation/Docker image validation/deployment, formatting, linting,
type-checking & unit tests upon contribution to the ``main`` branch using
[GitHub Actions](https://github.com/features/actions).

**Facilitate collaboration through:**
* An object-oriented structure for code sharing & reusability.
* A mono-repository workspace with task/experiment-specific subdirectories.
* A very informative & clear to navigate Python API reference thanks to
[Autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
and plugins like
[sphinx-autodoc-typehints](https://github.com/tox-dev/sphinx-autodoc-typehints)
and [sphinx-paramlinks](https://pypi.org/project/sphinx-paramlinks/).
* Shared logging with [Weights & Biases](https://wandb.ai/site).

**Promote high-quality and reproducible code by:**
* Linting with [Ruff](https://github.com/astral-sh/ruff),
formatting with [Black](https://github.com/psf/black),
unit-testing with [pytest](https://github.com/pytest-dev/pytest).
* Type-checking with [Mypy](https://github.com/python/mypy) (static)
& [Beartype](https://github.com/beartype/beartype) (dynamic).
* DType & Shape type hinting for [PyTorch](https://github.com/pytorch/pytorch)
tensors using [jaxtyping](https://github.com/google/jaxtyping) &
[NumPy](https://github.com/numpy/numpy) arrays using
[nptyping](https://github.com/ramonhagenaars/nptyping). Fully type checkable
at runtime with [Beartype](https://github.com/beartype/beartype).
* Providing a common [Development Container](https://containers.dev/)
recipe with the above features enabled + documentation preview
with [esbonio](https://github.com/swyddfa/esbonio).

**Smoothen up rough edges by providing:**
* Extensive documentation on how to install/execute on regular & SLURM-based
systems.
* Unassuming guides on how to contribute to the codebase.
* Tutorials on how to facilitate code transport across machines.

## II. Overview

### Service

#### 🔍 Overview

A ``service`` refers to a Python package located at
``common/SERVICE_NAME/``. Each ``service`` is meant to sketch out a form of
execution.

#### 📂 Examples

* [Optimization](
https://github.com/maximilienleclei/ai_repo/tree/main/common/optim/)

* [Inference](
https://github.com/maximilienleclei/ai_repo/tree/main/common/infer)

### Engine

#### 🔍 Overview

An ``engine`` refers to a Python package located at
``common/SERVICE_NAME/ENGINE_NAME``. Each ``engine`` is meant to drive a
specific type of execution.

#### 📂 Examples

* [Deep Learning](
https://github.com/maximilienleclei/ai_repo/tree/main/common/optim/dl)

* [Lightning Checkpoint Inference](
https://github.com/maximilienleclei/ai_repo/tree/main/common/infer/lightning)

### Project

#### 🔍 Overview

A ``project`` refers to a Python package located at ``projects/PROJECT_NAME/``.

#### 📂 Examples

* [MNIST classification](
https://github.com/maximilienleclei/ai_repo/tree/main/projects/classify_mnist/)

* [Neuroevolution Control Score Optimization](
https://github.com/maximilienleclei/ai_repo/tree/main/projects/ne_control_score/)

### Task

#### 🔍 Overview

A ``task`` is a work unit specified by a [Hydra](https://hydra.cc) ``.yaml``
config file located in ``projects/PROJECT_NAME/task/TASK_NAME.yaml``.

#### 📂 Examples

* [Neuroevolution Control Score Optimization on Acrobot](
https://github.com/maximilienleclei/ai_repo/tree/main/projects/ne_control_score/task/acrobot.yaml)

* [Learning rate random search for MNIST classification with a MLP on the
Beluga SLURM cluster](
https://github.com/maximilienleclei/ai_repo/tree/main/projects/classify_mnist/task/mlp_beluga.yaml)

### Subtask

A ``subtask`` is a sub-work unit of a ``task`` (ex: a model training run
with a specific set of hyper-parameters).

## Rough repository tree

```
ai_repo/
├─ .github/                  <-- Config files for GitHub Actions (tests, containers, etc)
├─ common/                   <-- Code common to various `projects`
│  ├─ infer/                 <-- Model inference
│  │  └─ lightning/          <-- Inference from Lightning checkpoints
│  └─ optim/                 <-- Model optimization
│     ├─ dl/                 <-- Deep Learning
│     │  ├─ datamodule/      <-- Lightning DataModules
│     │  ├─ litmodule/       <-- Lightning Modules
│     │  │  └─ nnmodule/     <-- PyTorch Modules
│     │  └─ train.py
│     └─ ne/                 <-- Neuroevolution
│        ├─ agent/
│        ├─ net/
│        ├─ space/           <-- Where agents evolve
│        └─ evolve.py
├─ docs/                     <-- Documentation
└─ projects/                 <-- Contains all existing projects
   │
   │                             ******************************************
   └─ my_new_dl_project/     <-- ******** Your new project folder *********
      ├─ task/               <-- *********** Your task folder *************
      │  └─ config.yaml      <-- ****** Your task configuration file ******
      ├─ __main__.py         <-- ************ Your entry-point ************
      ├─ datamodule.py       <-- ******* Your Lightning DataModule ********
      ├─ litmodule.py        <-- ********* Your Lightning Module **********
      └─ nnmodule.py         <-- ********** Your PyTorch Module ***********
                                 ******************************************
```


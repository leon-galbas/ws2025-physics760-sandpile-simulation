# Physics 760: Sandpile Simulation (WS 2024/25)

## Overview

This repository contains a numerical implementation of the Bak-Tang-Wiesenfeld
(BTW) model, designed to simulate and analyze the dynamics of self-organized
criticality (SOC) in sandpiles.

## Theoretical Foundation

The project primarily reproduces the work of **Christensen et al.** with a
modern implementation of the sandpile using Python. The main references are:

> Bak, P., Tang, C., & Wiesenfeld, K. (1987). _Self-organized criticality: An
> explanation of the 1/f noise_. Physical Review Letters, 59(4), 381.
>
> Christensen, K., Fogedby, H. C., & Jensen, H. J. (1991). _Dynamical and
> spatial aspects of sandpile cellular automata_. Journal of Statistical
> Physics, 63(3), 653–684.

For further details regarding this project, regard the
[paper](./report/2026_SimulatingSandpiles_Beck-Galbas).

## Installation

This project utilizes `uv` for efficient Python package management and
reproducible environments.

### 1. Install `uv`

The `uv` package can be installed as follows:

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For further details, refer to the package's
[official documentation](https://docs.astral.sh/uv/).

### 2. Initialize the Environment

After cloning the repository, navigate to the project root and initialize the
environment to ensure all dependencies are resolved and a virtual environment is
created:

```bash
uv sync
```

## Usage

The entry point for the simulation and subsequent statistical analysis is
located in `src/main.py`. In order to reproduce exactly the results discussed in
the paper, execute it using the `uv run` command:

```bash
uv run -m src.main
```

The `-m` flag is important to execute the script as a module. Otherwise, imports
within the repository will break.

[!IMPORTANT] Note that the actual simulations with $10^6$ or even $10^7$
measurements take a very long time to execute. To decrease the number of
measurements for testing purposes, one can do so by changing the respective
values in `src/config.yml`.

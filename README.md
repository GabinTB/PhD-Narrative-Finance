# Modeling Narrative Dynamics for Volatility Regime Detection in Financial Markets


## Research Question

How do financial narratives influence volatility dynamics and regime changes in financial markets?


## Research Summary

Financial markets are increasingly shaped by narratives, understood as collective interpretations that influence expectations, risk perception, and market behavior. While narratives are widely acknowledged as important drivers of market dynamics, they remain difficult to formalize and quantify within standard financial models.

This PhD research aims to bridge this gap by developing a computational framework that links the evolution of financial narratives to volatility dynamics and regime shifts in markets. The thesis integrates methods from natural language processing, high-frequency econometrics, and machine learning to:

1. Detect and quantify financial narratives across heterogeneous textual sources.
2. Measure volatility and its higher-order properties using high-frequency market data.
3. Study the causal and predictive relationships between narrative dynamics and volatility regimes.

The overarching goal is to provide a principled, data-driven approach to understanding how narrative formation, diffusion, and transformation relate to changes in market uncertainty and structural market behavior.


## Repository Structure

- `src/`: Core reusable code for narrative modeling and volatility analysis.
- `content/`: Thesis-related material (notebooks, scripts, publications, etc.) organized by chapters.
- `data/`: Datasets used and produced organised by source and thematics.
- ...


## Get started

This repository provides a lightweight development setup to ensure a consistent environment across machines.

### Development environment

**Main dependencies:**

A dedicated `.devenv/` folder is provided to bootstrap the Python environment. It contains:
- a Bash script (`setup.bashrc`) that checks for Conda, updates it if necessary, and creates a project-specific Conda environment (named `phd-nf`) if it does not already exist;
- a `requirements.txt` file listing the core dependencies required to run the main codebase.

To initialize the environment, run from the repository root:

```bash
source .devenv/setup.bashrc
```

This will:
1. Ensure Conda is available and up to date.
2. Create the project Conda environment if it does not already exist.
3. Install the core Python dependencies defined in .devenv/requirements.txt.

Once this step is completed, the environment is ready to use.

**Additional dependencies:**

Some notebooks or chapter-specific experiments may rely on additional libraries that are not part of the core codebase. These dependencies are installed locally within the corresponding notebooks or scripts when needed and are intentionally not included in the global requirements to keep the core environment minimal and stable.

After initializing the development environment, you can directly run the code, scripts, and notebooks in this repository.
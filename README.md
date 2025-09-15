# Statistical Learning / Machine Learning — Project Showcase

A lightweight portfolio of my machine learning coursework. Each project is self‑contained and reproducible, with clear implementations and compact plots. This repo is meant as a simple showcase rather than a full framework.

## Projects
- **Project 1 (formerly HW1): Decision Trees from Scratch** — minimal implementations of decision stumps/trees with entropy vs. error‑rate splits, plus a small CLI to reproduce figures.

## Quick Start
```bash
git clone https://github.com/HamedHeli/Statistical-Learning-Machine-Learning.git
cd Statistical-Learning-Machine-Learning

# optional: create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# list/run experiments
python code/main.py --help
```

> If `requirements.txt` is absent, typical deps are: `numpy pandas matplotlib scikit-learn scipy`.

## Repo Layout
```
code/      # source (algorithms & CLI)
data/      # small versioned datasets
figures/   # generated plots
README.md
```

## Notes
- Python 3.9+ recommended.
- Use the repo’s `.venv` for reproducibility.
- VS Code: select the `.venv` interpreter for smoother runs.

## License
Educational use. Please attribute if you reuse code or figures.

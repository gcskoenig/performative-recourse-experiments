# Code Accompagnying "Performative Validity of Recourse Explanations"

This repository contains the experiments for the paper "Performative Validity of Recourse Explanations" on [arXiv](https://arxiv.org/pdf/2506.15366).

## Installation

We recommend using `python 3.10.12`. The following packages were installed: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `seaborn`, `torch`, `cloudpickle`, `deap`, `networkx`, `joblib`, `pyro-ppl`, `tqdm`.
The full list of all installed packages including their versions can be found in `python_environment.txt`.

To install the local recourse code, `cd` to this folder and run

`pip install -e .``

## Experiment Scripts

To reproduce the main experiments run

`python scripts/metarun_simulation.py --N_REC 5000 --N_WORKERS 24 --seed 40`

The number of workers (parallel threads) and the seed can be adjusted to your liking. In our experiments we used the seeds 40, ..., 49.
The results will be stored in the folder `results/recourse` in the current directory.

To run the application, exectue

`python scripts/metarun_application.py --N_REC 1000 --N_WORKERS 5 --seed 40`.

Again the seeds can be adjusted to your liking.

To compile these results to the plots, run

`python scripts/run_visualization.py`

## Detailed Results

Upon request we can provide the detailed outputs (our result directory), due to size limitations they are not part of this repository.

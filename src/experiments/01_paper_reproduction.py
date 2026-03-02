import logging
from itertools import product
from multiprocessing import Pool
from os import path

import polars as pl

from src.calc.simulation import run_simulation
from src.utils import read_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"{read_config('log_dir')}/measurements.log"),
        logging.StreamHandler(),
    ],
)

hyperparameters = dict(
    dims=[(40, 2), (20, 3), (20, 4)],  # (15, 5), (10, 10)],
    boundary_conditions=["open", "closed"],
    perturbations=["conservative", "nonconservative"],
)

n_measure = 1e7  # 5e5
outpath = path.join(read_config("data_dir"), "scaling_coefficients.pq")
measurements = []


def run(dim, boundary, perturb):
    # for (N, d), boundary, perturb in parms:
    N, d = dim
    try:
        run_simulation(N, d, boundary, perturb, n_measure)
        outcome = {
            **hyperparameters,
            # **coeffs,
        }
        measurements.append(outcome)
        logging.info(f"Saving scaling exponents to {outpath}")
        df = pl.DataFrame(measurements)
        df.write_parquet(outpath)
    except Exception as e:
        logging.error(
            f"Simulation for {N=}, {d=}, {boundary=}, {perturb=}, {n_measure=} failed!\n\tException: {e}."
        )


parms = product(
    hyperparameters["dims"],
    hyperparameters["boundary_conditions"],
    hyperparameters["perturbations"],
)
with Pool() as pool:  # defaults to number of CPU cores
    pool.starmap(run, parms)

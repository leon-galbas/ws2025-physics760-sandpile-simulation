import logging

from src.calc.simulation import run_burn_in_measurement
from src.utils import read_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"{read_config('log_dir')}/burn_in_stats.log"),
        logging.StreamHandler(),
    ],
)

hyperparameters = dict(
    dims=[(40, 2), (20, 3)],
    boundary_conditions=["open", "closed"],
    perturbations=["conservative", "nonconservative"],
    inits=[None, "random", "max"],
)

run_burn_in_measurement(hyperparameters, outfile="burn_in_stats.pq")

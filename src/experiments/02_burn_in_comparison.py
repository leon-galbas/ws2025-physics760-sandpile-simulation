import logging
import time
from itertools import product
from os import path

from src.model.sandpile import SandpileModel
from src.utils import append_dict_to_parquet, read_config

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
    dims=[(40, 2), (20, 3)],
    boundary_conditions=["open", "closed"],
    perturbations=["conservative", "nonconservative"],
    inits=[None, "random", "max"],
)

outpath = path.join(read_config("data_dir"), "burn_in_stats.pq")

for (N, d), boundary, perturb, init in product(
    hyperparameters["dims"],
    hyperparameters["boundary_conditions"],
    hyperparameters["perturbations"],
    hyperparameters["inits"],
):
    try:
        start_time = time.perf_counter()
        model = SandpileModel(
            N=N, d=d, boundary_condition=boundary, perturbation=perturb, z_init=init
        )
        logging.info(
            f"Starting burn-in for {N=}, {d=}, {boundary=}, {perturb=}, {init=}."
        )
        model.burn_in()
        duration = time.perf_counter() - start_time
        measurements = {
            "N": N,
            "d": d,
            "boundary": boundary,
            "perturb": perturb,
            "init": "zero" if init is None else init,
            "z_mean_hist": model.z_mean_hist,
            "burn_in_time": len(model.z_mean_hist),
            "burn_in_seconds": duration,
        }
        logging.info(f"Saving burn in stats to {outpath}")
        append_dict_to_parquet(measurements, outfile=outpath)
        measurements = {
            measurements[k]: v for k, v in measurements.items() if k != "z_mean_hist"
        }
        logging.info(f"Measurements:\n{measurements}")
    except Exception as e:
        logging.error(
            f"Simulation for {N=}, {d=}, {boundary=}, {perturb=}, {init=} failed!\n\tException: {e}."
        )

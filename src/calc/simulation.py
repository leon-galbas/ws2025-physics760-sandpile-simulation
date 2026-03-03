import logging
import time
from itertools import product
from os import path

import torch

from src.model.io import load_model, model_exists
from src.model.sandpile import SandpileModel
from src.utils import append_dict_to_parquet, read_config


def run_simulation(
    N: int,
    d: int,
    boundary_condition: str,
    perturbation: str,
    num_measurements: int,
    z_init: str | int | torch.Tensor = "random",
    model_name: str = None,
):
    if model_name is None:
        model_name = f"N{N}d{d}_{boundary_condition}_{perturbation}.pkl"

    # convert num_measurements to int
    num_measurements = int(float(num_measurements))

    # get model with given hyperparameters
    logging.info(
        f"Running simulation with {N=}, {d=}, {boundary_condition=}, {perturbation=}, {num_measurements=}..."
    )
    if model_exists(model_name):
        logging.info(
            "A simulation with these hyperparameters has already been run. Loading..."
        )
        model = load_model(model_name)
        n = num_measurements - model.num_measurements
        if n > 0:
            logging.info(f"Performing missing {n=} measurements.")
            model.measure(num_measurements=n)
            model.save(model_name)
    else:
        model = SandpileModel(
            N,
            d,
            boundary_condition=boundary_condition,
            perturbation=perturbation,
            z_init=z_init,
        )
        model.burn_in()
        model.save(model_name)
        model.measure(num_measurements=num_measurements)
        model.save(model_name)


def run_burn_in_measurement(hyperparameters: dict, outfile="burn_in_stats.pq"):

    outpath = path.join(read_config("data_dir"), outfile)

    # Start burn in time measurement
    logging.info("Measuring burn in times for different sandpile configurations.")
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
            model_name = f"N{N}d{d}_{boundary}_{perturb}.pkl"
            logging.info(
                f"Starting burn-in for {N=}, {d=}, {boundary=}, {perturb=}, {init=}."
            )
            model.burn_in()
            if not model_exists(model_name):
                model.save(model_name)
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
            logging.info(f"Saving burn in stats to '{outpath}'.")
            append_dict_to_parquet(measurements, outfile=outpath)
        except Exception as e:
            logging.error(
                f"Simulation for {N=}, {d=}, {boundary=}, {perturb=}, {init=} failed!\n\tException: {e}."
            )
    logging.info("Burn in measurement completed!")

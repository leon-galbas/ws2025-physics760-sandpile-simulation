import logging

from src.model.io import load_model, model_exists
from src.model.sandpile import SandpileModel


def run_simulation(
    N: int,
    d: int,
    boundary_condition: str,
    perturbation: str,
    num_measurements: int,
    z_init="random",
):
    # convert num_measurements to int
    num_measurements = int(num_measurements)
    # get model with given hyperparameters
    logging.info(
        f"Running simulation with {N=}, {d=}, {boundary_condition=}, {perturbation=}, {num_measurements=}..."
    )
    model_name = f"N{N}d{d}_{boundary_condition}_{perturbation}.pkl"
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

import logging

from src.calc.scaling_exponents import compute_scaling_exponents
from src.model.io import load_model, model_exists
from src.model.sandpile import SandpileModel


def run_simulation(
    N: int, d: int, boundary_condition: str, perturbation: str, num_measurements: int
):
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
    else:
        model = SandpileModel(N, d)
        model.burn_in()
        model.measure(num_measurements=1e6)
        model.save(model_name)

    # compute scaling exponents
    df = model.data
    exponents = compute_scaling_exponents(data=df)

    return exponents

import logging
from itertools import product
from multiprocessing import Pool

from src.calc.simulation import run_simulation
from src.model.io import load_model
from src.utils import read_config

hyperparameters = dict(
    dims=[(15, 5), (10, 6)],
    boundary_conditions=["open"],
    perturbations=["conservative"],
)

n_measure = 1e5  # 5e5

model_name_5 = "N15d5_open_conservative copy.pkl"
model_name_6 = "N10d6_open_conservative copy.pkl"
z_init_5 = load_model(model_name_5).z
z_init_6 = load_model(model_name_6).z


def run(dim, boundary, perturb):
    N, d = dim

    # handle logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_file = (
        f"{read_config('log_dir')}/N{N}d{d}_{boundary}_{perturb}_measurements.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # run simulations
    try:
        if d == 5:
            run_simulation(N, d, boundary, perturb, n_measure, z_init=z_init_5)
        elif d == 6:
            run_simulation(N, d, boundary, perturb, n_measure, z_init=z_init_6)
        else:
            run_simulation(N, d, boundary, perturb, n_measure)
    except Exception as e:
        logging.error(
            f"Simulation for {N=}, {d=}, {boundary=}, {perturb=}, {n_measure=} failed!\n\tException: {e}."
        )
    finally:
        for handler in root_logger.handlers[:]:
            handler.flush()
            handler.close()
            root_logger.removeHandler(handler)


parms = product(
    hyperparameters["dims"],
    hyperparameters["boundary_conditions"],
    hyperparameters["perturbations"],
)

if __name__ == "__main__":
    with Pool() as pool:
        pool.starmap(run, parms)

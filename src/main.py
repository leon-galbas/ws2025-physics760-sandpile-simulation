import logging
from datetime import datetime
from os import path

from src.calc.simulation import run_burn_in_measurement, run_simulation
from src.utils import read_config
from src.visualization.burn_in_plots import main as run_burn_in_plots
from src.visualization.plots import main as run_scaling_plots


def main():
    data_dir = read_config("data_dir")

    # run burn in measurements
    burn_in_hyperparams = dict(
        dims=[(40, 2), (20, 3)],
        boundary_conditions=["open", "closed"],
        perturbations=["conservative", "nonconservative"],
        inits=[None, "random", "max"],
    )
    burn_in_filename = "burn_in_stats.pq"
    if not path.exists(path.join(data_dir, burn_in_filename)):
        run_burn_in_measurement(burn_in_hyperparams, outfile=burn_in_filename)

    # plot burn in times
    run_burn_in_plots()

    # run avalanche measurements
    configs = read_config("model_configurations")
    for config in configs.values():
        run_simulation(**config, z_init="max")

    # plot scaling exponents
    run_scaling_plots(plot_boundaries=True)
    run_scaling_plots(plot_boundaries=False)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                f"{read_config('log_dir')}/{timestamp}_measurements.log"
            ),
            logging.StreamHandler(),
        ],
    )
    main()

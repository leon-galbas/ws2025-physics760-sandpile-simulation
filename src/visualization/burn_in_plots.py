import logging
from os import path

import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl

from src.utils import read_config, read_plot_config

# configure matplotlib parameters
mpl.rcParams.update(read_plot_config("matplotlib_setup"))


def main():
    # load data
    try:
        data_path = path.join(read_config("data_dir"), "burn_in_stats.pq")
        df = pl.read_parquet(data_path)
    except Exception as e:
        raise FileNotFoundError(f"The burn in data could not be loaded!\nError: {e}")

    # plot
    logging.info("Generating burn-in plots...")
    plot_dir = path.join(read_config("figure_dir"), "burn_in_plots")
    for i in range(df.height):
        row = df[i]
        N = row["N"][0]
        d = row["d"][0]
        boundary_condition = row["boundary"][0]
        perturbation = row["perturb"][0]
        init = row["init"][0]
        z_mean_hist = row["z_mean_hist"][0]

        model_name = f"N{N}d{d}_{boundary_condition}_{perturbation}_{init}"
        plot_path = path.join(plot_dir, f"{model_name}_burn_in.pdf")

        plt.plot(range(len(z_mean_hist)), z_mean_hist)
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$\langle z\rangle (\tau)$")
        plt.savefig(plot_path, bbox_inches="tight")
        logging.info(f"Plot saved to '{plot_path}'.")
        plt.clf()

    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

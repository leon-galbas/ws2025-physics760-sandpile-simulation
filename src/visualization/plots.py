import logging
from os import path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calc.scaling_exponents import compute_scaling_exponents
from src.model.io import load_model, model_exists
from src.utils import decimals_from_err, read_config, read_plot_config
from src.visualization.mappings import (
    label_conv_table,
    latex_conv_table,
    model_name_to_id,
    tab1_columns,
    tab2_columns,
    x_label_conv_table,
)

# configure matplotlib parameters
mpl.rcParams.update(read_plot_config("matplotlib_setup"))


def main():
    """Generate all plots and tables used in the paper."""
    data = generate_plots()
    generate_tables(data)


def generate_plots() -> pd.DataFrame:
    exponents_latex = []

    # create plots
    logging.info("Start generating plots...")
    for name in model_name_to_id.keys():
        logging.info(f"Generating log-log plots for model '{name}'.")
        filename = f"{name}.pkl"
        model_id = model_name_to_id[name]

        # Skip non existent models
        if not model_exists(filename):
            logging.error(
                f"The model '{filename}' does not exist in the data folder! Skipping."
            )
            continue

        # Calculate scaling exponents and generate plots
        try:
            model = load_model(filename)
            df_raw = model.data
            exponents = plot_scaling_exponents(
                df_raw,
                name,
                **get_scaling_window_args(model_id),
            )
            exponents_latex.append(exponents)
            logging.info("Done!")
        except Exception as e:
            logging.error(f"Failed to create log-log plots for '{name}':\nError: {e}")

    exponents_df = pd.DataFrame(exponents_latex)

    return exponents_df


def generate_tables(exponents_df: pd.DataFrame):
    # define which tables to generate
    tables_to_generate = [
        ("open", "conservative"),
        ("open", "nonconservative"),
        ("closed", "nonconservative"),
    ]

    # generate tables
    logging.info("Start generating tables...")
    for boundary, perturbation in tables_to_generate:
        logging.info(
            f"Generating scaling exponent tables for {boundary=}, {perturbation=}."
        )

        # filter for relevant model configurations
        configs = read_config("model_configurations")
        relevant_configs = [
            k
            for k, v in configs.items()
            if v["boundary_condition"] == boundary and v["perturbation"] == perturbation
        ]
        mask = exponents_df["model"].isin(relevant_configs)
        filtered_df = exponents_df[mask]

        # generate table for probability density scaling exponents
        table_path = path.join(
            read_config("figure_dir"),
            "tables",
            f"scaling_exp_{boundary}_{perturbation}.tex",
        )
        filtered_df.to_latex(
            buf=table_path,
            index=False,
            column_format="c" * len(tab1_columns),
            columns=tab1_columns,
        )
        logging.info(f"Table 1 saved to '{table_path}'.")

        # generate table for probability density scaling exponents
        table_path = path.join(
            read_config("figure_dir"),
            "tables",
            f"scaling_exp_{boundary}_{perturbation}_gamma.tex",
        )
        filtered_df.to_latex(
            buf=table_path,
            index=False,
            column_format="c" * len(tab2_columns),
            columns=tab2_columns,
        )
        logging.info(f"Table 2 saved to '{table_path}'.")

    logging.info("Done!")


def plot_scaling_exponents(
    df,
    model_name,
    window_size,
    window_step_size,
    r_thresh,
    deviation_factor,
    do_errors=False,
    cond_bounds={},
):
    # initialize latex ready exponents dict
    if model_name not in model_name_to_id.keys():
        raise ValueError(f"Could not find a model_id for name '{model_name}'.")
    exponents_latex = {"model": model_name_to_id[model_name]}
    # calculate exponents
    exponents = compute_scaling_exponents(
        df, window_size, window_step_size, r_thresh, deviation_factor, cond_bounds
    )

    # generate plots
    plot_dir = path.join(read_config("figure_dir"), "loglog_plots")
    for key, exp in exponents.items():
        values = exp[0]
        fit = exp[1]
        x = np.asarray(values[0])
        y = np.asarray(values[1])
        yerr = np.asarray(values[2])

        # plot data
        plt.plot(x, y, label=label_conv_table[key])

        # plot error
        if do_errors:
            band_color = "tab:orange"

            plt.plot(
                x, y + yerr, color=band_color, lw=1.2, ls="--", label=r"$+\,\sigma$"
            )
            plt.plot(
                x, y - yerr, color=band_color, lw=1.2, ls="--", label=r"$-\,\sigma$"
            )

            plt.fill_between(
                x, y - yerr, y + yerr, color=band_color, alpha=0.15, linewidth=0
            )

        # plot fit
        lin_regress_x = np.linspace(np.min(x), np.max(x), 10)
        if key in ["tau", "alpha", "lambda"]:
            lin_regress_y = lin_regress_x * (1 - fit["exponent"]) + fit["intercept"]
            fitlabel = (
                "WLS fit\n"
                + r"$m=$"
                + f"{(1 - fit['exponent']):.4f}\n"
                + r"$\Delta m=$"
                + f"{fit['std_err']:.4f}"
            )
        else:
            lin_regress_y = lin_regress_x * fit["exponent"] + fit["intercept"]
            fitlabel = (
                "WLS fit\n"
                + r"$m=$"
                + f"{fit['exponent']:.4f}\n"
                + r"$\Delta m=$"
                + f"{fit['std_err']:.4f}"
            )
        plt.plot(lin_regress_x, lin_regress_y, label=fitlabel)

        # plot scaling region
        plt.axvline(
            x[fit["lower"]],
            linestyle=":",
            label="fitting window",
        )
        plt.axvline(x[fit["upper"]], linestyle=":")

        # plot layout
        plot_path = path.join(plot_dir, f"{model_name}_{key}.pdf")
        plt.legend()
        plt.xlabel(x_label_conv_table[key])
        plt.ylabel(label_conv_table[key])
        plt.savefig(plot_path, bbox_inches="tight")
        plt.clf()

        # format scaling exponents for latex tables
        sig = decimals_from_err(fit["std_err"])
        exponents_latex[latex_conv_table[key]] = (
            f"$ ({fit['exponent']:.{sig + 1}f}"
            + r"\pm"
            + f"{fit['std_err']:.{sig + 1}f}) $"
        )

    return exponents_latex


def get_scaling_window_args(model_id: str) -> dict:
    """Read the arguments for scaling window determination from config for a given model id.

    Args:
        model_id (str): The model id.

    Returns:
        dict: The scaling window determination parameters.
    """
    # load default scaling window parameters
    defaults = read_plot_config("scaling_window", "default")
    # return parameters for given model_id, fill missing values with defaults
    if model_id in read_plot_config("scaling_window").keys():
        return_dict = {}  # start from empty dict to ensure correct order of return dict
        values = read_plot_config("scaling_window", model_id)
        for key in defaults.keys():
            if key in values.keys():
                return_dict[key] = values[key]
            else:
                return_dict[key] = defaults[key]
        return return_dict
    else:
        return defaults


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

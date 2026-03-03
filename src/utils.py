import os
from pathlib import Path

import numpy as np
import polars as pl
import yaml

CONFIG_FILE = "src/config.yml"
CONFIG_FILE_PLOTS = "src/config_plots.yml"


def read_config(*keys, filepath=CONFIG_FILE):
    """Read a YAML config file and return a specific nested entry if keys are provided.

    Parameters:
        *keys: Arbitrary number of nested keys to access the value
        filepath: Path to the YAML config file (default: "config.yml")

    Returns:
        The value corresponding to the nested keys, or the full config if no keys.
    """
    with open(filepath, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error reading YAML file: {e}")

    if not keys:
        return config

    value = config
    for key in keys:
        if key not in value:
            raise KeyError(f"Key '{key}' not found in config")
        value = value[key]

    return value


def read_plot_config(*keys):
    """Read the plot YAML config file and return a specific nested entry if keys are provided.

    Parameters:
        *keys: Arbitrary number of nested keys to access the value

    Returns:
        The value corresponding to the nested keys, or the full config if no keys.
    """
    return read_config(*keys, filepath=CONFIG_FILE_PLOTS)


def numpy_to_list(obj):
    """Converts a numpy array to a list. All other types of objects are untouched.

    Args:
        obj (object): Any object.

    Returns:
        object: A list if the input is a np.ndarray, otherwise the input is returned.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def append_dict_to_parquet(dict_in: dict, outfile: str):
    """Appends a dictionary to a polars DataFrame saved in a parquet file.

    If the file does not exist yet, it is newly created. This function saves to a
    temporary file first, in order to avoid data loss if the process is stopped
    unexpectedly.

    Args:
        dict_in (dict): The dictionary.
        outfile (str): The DataFrame parquet file.
    """
    path = Path(outfile)
    tmp_path = path.with_suffix(".tmp.pq")

    # Convert numpy arrays to lists
    dict_in = {k: numpy_to_list(v) for k, v in dict_in.items()}

    # Combine existing data with new data
    new_df = pl.DataFrame([dict_in])
    if path.exists():
        existing_df = pl.read_parquet(path)
        combined_df = pl.concat([existing_df, new_df], how="vertical")
    else:
        combined_df = new_df

    # Write to temporary file and rename if successful
    combined_df.write_parquet(tmp_path)
    os.replace(tmp_path, path)


def decimals_from_err(err: float) -> int:
    """Calculates the number decimals at which the first nonzero error digit appears.

    Args:
        err (float): An error value.

    Returns:
        int: The first nonzero digit after the decimal point.
    """
    err = float(err)
    if err == 0:
        return 0
    else:
        decimals = max(0, -np.floor(np.log10(abs(err))))
        return int(decimals)

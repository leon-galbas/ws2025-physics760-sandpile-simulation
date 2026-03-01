import os
from pathlib import Path

import numpy as np
import polars as pl
import yaml


def read_config(*keys, filepath="src/config.yml"):
    """
    Read a YAML config file and return a specific nested entry if keys are provided.

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


def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def append_dict_to_parquet(dict_in, outfile):
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

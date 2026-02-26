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

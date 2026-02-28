import logging
import pickle
from os import path

from src.model.sandpile import SandpileModel
from src.utils import read_config


def load_model(filename: str) -> SandpileModel:
    """Load a save instance of SandpileModel

    Args:
        filename (str): Name of the model file.

    Returns:
        SandpileModel: The loaded model.
    """
    if not model_exists(filename):
        raise FileNotFoundError(f"Model named '{filename}' does not exist!")

    model_dir = read_config("model_dir")
    filepath = path.join(model_dir, filename)
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    logging.info(f"Model loaded from '{filepath}'.")

    return model


def model_exists(filename: str) -> bool:
    """Check if a saved model with the given filename exists.

    Args:
        filename (str): Model Filename.

    Returns:
        bool: Whether the model exists.
    """
    model_dir = read_config("model_dir")
    filepath = path.join(model_dir, filename)

    return path.exists(filepath)

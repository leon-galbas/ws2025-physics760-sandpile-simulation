import logging
import pickle
from os import path

from src.model.sandpile import SandpileModel
from src.utils import read_config


def load_model(name: str) -> SandpileModel:
    """Load a save instance of SandpileModel

    Args:
        name (str): Name of the model file '<name>.pkl'

    Returns:
        SandpileModel: The loaded model
    """
    model_dir = read_config("model_dir")
    filepath = path.join(model_dir, f"{name}.pkl")
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    logging.info(f"Model loaded from '{filepath}'.")

    return model

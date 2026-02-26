import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_scaling_exponents(data: pd.DataFrame):

    s = np.asarray(data["s"], dtype=np.int64)
    t = np.asarray(data["s"], dtype=np.int64)
    l = np.asarray(data["s"], dtype=np.int64)
    exponents = {}
    logging.info("Calculating scaling exponents.")

    # Calculate exponents from probability densities
    exponents["tau"] = get_prob_exponent(s)
    exponents["alpha"] = get_prob_exponent(t)
    exponents["lambda"] = get_prob_exponent(l)

    # Calculate exponents from conditional expectation values
    exponents["gamma_1"] = get_cond_exponent(t, s)
    exponents["1/gamma_1"] = get_cond_exponent(s, t)

    exponents["gamma_2"] = get_cond_exponent(l, s)
    exponents["1/gamma_2"] = get_cond_exponent(s, l)

    exponents["gamma_3"] = get_cond_exponent(l, t)
    exponents["1/gamma_3"] = get_cond_exponent(t, l)

    logging.info(f"Calculated exponents:\n{exponents}")
    return exponents


def get_prob_exponent(x: np.ndarray) -> tuple[float, float]:
    """Calculate probability density scaling exponent.

    This assumes that P(X=x) ~ x^(1-a). To obtain a, the function fits log(P(X=x))
    against log(x) linearly and returns a = 1-slope and its error.

    Args:
        x (np.ndarray): Array of x values.

    Returns:
        float: Estimated exponent.
        float: Exponent error.
    """
    # Compute probabilities
    unique_x, counts = np.unique(x, return_counts=True)
    probabilities = counts / x.size

    # Filter out positive x values
    valid_mask = unique_x > 0
    x = unique_x[valid_mask]
    P = probabilities[valid_mask]

    # Do a linear log-log-fit
    log_x = np.log10(x)
    log_P = np.log10(P)
    fit = linregress(log_x, log_P)
    exp = fit.slope
    std_err = fit.stderr

    return 1 - exp, std_err


def get_cond_exponent(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Calculate conditional expectation value scaling exponents.

    This assumes that E[Y|X=x] ~ x^a. To obtain a, the function fits log(E[Y|X=x])
    against log(x) linearly and returns a = slope and its error.

    Args:
        x (np.ndarray): Array of x values.
        y (np.ndarray): Array of y values.

    Returns:
        float: Estimated exponent.
        float: Exponent error.
    """
    # Do weighted bincount
    sums = np.bincount(x, weights=y)
    counts = np.bincount(x)

    # Compute conditional expectation values
    valid = counts > 0
    unique_x = np.arange(len(counts))[valid]
    cond_exp_y = sums[valid] / counts[valid]

    # Do a linear log-log-fit
    valid_log = (unique_x > 0) & (cond_exp_y > 0)
    log_x = np.log10(unique_x[valid_log])
    log_y = np.log10(cond_exp_y[valid_log])
    fit = linregress(log_x, log_y)
    exp = fit.slope
    std_err = fit.stderr

    return exp, std_err

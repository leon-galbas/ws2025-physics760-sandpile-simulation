import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_scaling_exponents(data: pd.DataFrame, window_size, window_step_size, r_thresh, k):

    s = np.asarray(data["s"], dtype=np.int64)
    t = np.asarray(data["t"], dtype=np.int64)
    l = np.asarray(data["l"], dtype=np.int64)  # noqa: E741
    exponents = {}
    logging.info("Calculating scaling exponents.")

    # Calculate exponents from probability densities
    exponents["tau"] = get_prob_exponent(s, window_size, window_step_size, r_thresh, k)
    exponents["alpha"] = get_prob_exponent(t, window_size, window_step_size, r_thresh, k)
    exponents["lambda"] = get_prob_exponent(l, window_size, window_step_size, r_thresh, k)

    # Calculate exponents from conditional expectation values
    exponents["gamma_1"] = get_cond_exponent(t, s, window_size, window_step_size, r_thresh, k)
    exponents["inv_gamma_1"] = get_cond_exponent(s, t, window_size, window_step_size, r_thresh, k)

    exponents["gamma_2"] = get_cond_exponent(l, s, window_size, window_step_size, r_thresh, k)
    exponents["inv_gamma_2"] = get_cond_exponent(s, l, window_size, window_step_size, r_thresh, k)

    exponents["gamma_3"] = get_cond_exponent(l, t, window_size, window_step_size, r_thresh, k)
    exponents["inv_gamma_3"] = get_cond_exponent(t, l, window_size, window_step_size, r_thresh, k)

    #logging.info(f"Calculated exponents:\n{exponents}")
    return exponents


def get_prob_exponent(x: np.ndarray, window_size, window_step_size, r_thresh, k) -> tuple[float, float]:
    """Calculate probability density scaling exponent.

    This assumes that P(X=x) ~ x^(1-a). To obtain a, the function fits log(P(X=x))
    against log(x) linearly and returns a = 1-slope and its error.

    Args:
        x (np.ndarray): Array of x values.

    Returns:
        dict: paramters for plotting.
        values np.array: log(x) and log(y) values for plotting
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
    lower, upper =get_scaling_window(log_x, log_P, window_size, window_step_size, r_thresh, k)
    fit = linregress(log_x[lower:upper], log_P[lower:upper])
    exp = fit.slope
    std_err = fit.stderr
    intercept = fit.intercept  
    intercept_stderr = fit.intercept_stderr
    

    parms = {
    "exponent": 1-exp,
    "std_err": std_err,
    "intercept": intercept,
    "intercept_stderr": intercept_stderr,
    "lower": lower,
    "upper": upper,
    }
    values = np.array([log_x, log_P])
    return values, parms

def get_cond_exponent(x: np.ndarray, y: np.ndarray, window_size, window_step_size, r_thresh, k) -> tuple[float, float]:
    """Calculate conditional expectation value scaling exponents.

    This assumes that E[Y|X=x] ~ x^a. To obtain a, the function fits log(E[Y|X=x])
    against log(x) linearly and returns a = slope and its error.

    Args:
        x (np.ndarray): Array of x values.
        y (np.ndarray): Array of y values.

    Returns:
        dict: paramters for plotting.
        values np.array: log(x) and log(y) values for plotting
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
    lower, upper =get_scaling_window(log_x, log_y, window_size, window_step_size, r_thresh, k)
    fit = linregress(log_x[lower:upper], log_y[lower:upper])
    exp = fit.slope
    std_err = fit.stderr
    intercept = fit.intercept  
    intercept_stderr = fit.intercept_stderr
    # parameters relevant for plotting
    parms = {
    "exponent": exp,
    "std_err": std_err,
    "intercept": intercept,
    "intercept_stderr": intercept_stderr,
    "lower": lower,
    "upper": upper,
    }
    # values for plotting
    values = np.array([log_x, log_y])
    return values, parms


def get_scaling_window(x,y, window_size, window_step_size, r_thresh, deviation_factor):
    """
    Find a heuristic scaling window (in log-log space) using a sliding-window
    regression and a slope-stability criterion.

    Parameters
    ----------
    x, y : array
    window_size : int
        Number of points per sliding regression window.
    window_step_size : int
        Step size (in points) between consecutive windows.
    r_thresh : float
        Minimum R^2 required for a window to be considered a good linear fit.
    deviation_factor : float
        Multiplier controlling how strict the slope-stability condition is.

    Returns
    -------
    lower, upper : int
        Indices that mark the start and end of the longest linear region.

    """
    # initialize important arrays and variables
    n = len(x)
    window_amount=int((n - window_size) / window_step_size)
    slopes = np.empty(window_amount) 
    std_err = np.empty(window_amount)
    r_squared = np.empty(window_amount) 

    # compute slopes r2 values and deviations for all windows
    for i in range(window_amount): 
        test_window_lower=  i * window_step_size
        test_window_upper=  i * window_step_size + window_size
        fit = linregress(x[test_window_lower:test_window_upper], y[test_window_lower:test_window_upper])   
        slopes[i]= fit.slope
        std_err[i]= fit.stderr
        r_squared[i]= fit.rvalue**2

    #check if r2 is large enogh
    good_fit= r_squared>=r_thresh
    # init upper and lower
    upper = lower = 0
    # compute difference of slopes and their adjecent window slopes
    diff = np.abs(slopes[0:-1]-slopes[1:])
    # compute std of these slopes using gaussian error propagation
    diff_std = np.sqrt(std_err[:-1]**2 + std_err[1:]**2)
    #check if the slopes are stable
    stable =  np.isfinite(diff_std) & (diff <= deviation_factor * diff_std)&good_fit[:-1]&good_fit[1:]
    
    # simple algorithm for finding the longest consecutive streak of True values in stable
    best_len = 0
    lower = None
    i = 0
    while i < len(stable):
        if not stable[i]:
            i += 1
            continue
        j = i
        while j < len(stable) and stable[j]:
            j += 1
        if (j - i) > best_len:
            best_len = j - i
            lower = i
        i = j

    # compute and return upper and lower
    upper= lower+best_len
    return lower, upper    
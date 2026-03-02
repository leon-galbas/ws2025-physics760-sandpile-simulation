import logging

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress


def linear_func(x, m, b):
    return m * x + b


def compute_scaling_exponents(
    data: pd.DataFrame, window_size, window_step_size, r_thresh, k, manuel_bounds= {}
):

    s = np.asarray(data["s"], dtype=np.int64)
    t = np.asarray(data["t"], dtype=np.int64)
    l = np.asarray(data["l"], dtype=np.int64)  # noqa: E741
    exponents = {}
    logging.info("Calculating scaling exponents.")

    # Calculate exponents from probability densities
    
    man_key= list(manuel_bounds.keys()) 
    def _bounds_for(name: str):
        return manuel_bounds.get(name) if name in man_key else None
    
    exponents["tau"] = get_prob_exponent(
    s, window_size, window_step_size, r_thresh, k, bounds=_bounds_for("tau")
    )

    exponents["alpha"] = get_prob_exponent(
        t, window_size, window_step_size, r_thresh, k, bounds=_bounds_for("alpha")
    )

    exponents["lambda"] = get_prob_exponent(
        l, window_size, window_step_size, r_thresh, k, bounds=_bounds_for("lambda")
    )

    logging.info("Calculating conditional scaling exponents.")
    # Calculate exponents from conditional expectation values
    exponents["gamma_1"] = get_cond_exponent(
        t, s, window_size, window_step_size, r_thresh, k, bounds=_bounds_for("gamma_1")
    )
    exponents["inv_gamma_1"] = get_cond_exponent(
        s, t, window_size, window_step_size, r_thresh, k, bounds=_bounds_for("inv_gamma_1")
    )

    exponents["gamma_2"] = get_cond_exponent(
        l, s, window_size, window_step_size, r_thresh, k, bounds=_bounds_for("gamma_2")
    )
    exponents["inv_gamma_2"] = get_cond_exponent(
        s, l, window_size, window_step_size, r_thresh, k, bounds=_bounds_for("inv_gamma_2")
    )

    exponents["gamma_3"] = get_cond_exponent(
        l, t, window_size, window_step_size, r_thresh, k, bounds=_bounds_for("gamma_3")
    )
    exponents["inv_gamma_3"] = get_cond_exponent(
        t, l, window_size, window_step_size, r_thresh, k, bounds=_bounds_for("inv_gamma_3")
    )

    # logging.info(f"Calculated exponents:\n{exponents}")
    return exponents


def get_prob_exponent(
    x: np.ndarray, window_size, window_step_size, r_thresh, k, bounds=None
) -> tuple[float, float]:
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
    N = len(P)
    P_err = np.sqrt(P * (1 - P) / N)

    # Do a linear log-log-fit
    log_x = np.log10(x)
    log_P = np.log10(P)
    log_P_err = P_err / (10 * P)
    if bounds == None: 
        lower, upper = get_scaling_window(
            log_x, log_P, window_size, window_step_size, r_thresh, k
        )
    else: 
        lower, upper = bounds

    popt, pcov = curve_fit(
        linear_func,
        log_x[lower:upper],
        log_P[lower:upper],
        sigma=log_P_err[lower:upper],
        absolute_sigma=True,
    )
    exp, intercept = popt
    std_err, intercept_stderr = np.sqrt(np.diag(pcov))

    parms = {
        "exponent": 1 - exp,
        "std_err": std_err,
        "intercept": intercept,
        "intercept_stderr": intercept_stderr,
        "lower": lower,
        "upper": upper,
    }

    values = np.array([log_x, log_P, log_P_err])
    return values, parms


def get_cond_exponent(
    x: np.ndarray, y: np.ndarray, window_size, window_step_size, r_thresh, k, bounds= None
) -> tuple[float, float]:
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
    # Map any discrete y values to 0, 1, ..., K-1
    unique_y, y_idx = np.unique(y, return_inverse=True)

    # N_y: count of elements for each unique y
    N_y = np.bincount(y_idx)

    # Sum of X and X^2 conditioned on y
    sum_x = np.bincount(y_idx, weights=x)
    sum_x2 = np.bincount(y_idx, weights=x**2)

    # Suppress divide-by-zero warnings for empty bins/N_y=1 during array operations
    with np.errstate(divide="ignore", invalid="ignore"):
        # Equation (24): Conditional expectation
        E = sum_x / N_y

        # Equation (26) Expanded: Conditional sample variance (ddof=1)
        # Using np.maximum to avoid small negative floats due to precision errors
        sum_sq_diff = np.maximum(sum_x2 - N_y * E**2, 0)
        var_cond = sum_sq_diff / (N_y - 1)

        # Equation (25): Standard deviation of the estimator
        sigma_E = np.sqrt(var_cond / N_y)

    # Handle mathematical undefined states directly
    E[N_y == 0] = np.nan
    sigma_E[N_y <= 1] = np.nan

    # DO fit
    valid = (
        (unique_y > 0) & (E > 0) & np.isfinite(E) & (sigma_E > 0) & np.isfinite(sigma_E)
    )
    y_val = unique_y[valid]
    E_val = E[valid]
    sigma_E_val = sigma_E[valid]

    # 2. Transform to log-log space
    log_y = np.log10(y_val)
    log_E = np.log10(E_val)

    # 3. Propagate errors to logarithmic space
    sigma_log_E = sigma_E_val / (10 * E_val)

    if bounds == None: 
        lower, upper = get_scaling_window(
            log_y, log_E, window_size, window_step_size, r_thresh, k
        )
    else: 
        lower, upper = bounds
    popt, pcov = curve_fit(
        f=linear_func,
        xdata=log_y[lower:upper],
        ydata=log_E[lower:upper],
        sigma=sigma_log_E[lower:upper],
        absolute_sigma=True,
    )
    exp, intercept = popt
    std_err, intercept_stderr = np.sqrt(np.diag(pcov))

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
    values = np.array([log_y, log_E, sigma_log_E])
    return values, parms


def get_scaling_window(x, y, window_size, window_step_size, r_thresh, deviation_factor):
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
    window_amount = int((n - window_size) / window_step_size)
    slopes = np.empty(window_amount)
    std_err = np.empty(window_amount)
    r_squared = np.empty(window_amount)

    # compute slopes r2 values and deviations for all windows
    for i in range(window_amount):
        test_window_lower = i * window_step_size
        test_window_upper = i * window_step_size + window_size
        fit = linregress(
            x[test_window_lower:test_window_upper],
            y[test_window_lower:test_window_upper],
        )
        slopes[i] = fit.slope
        std_err[i] = fit.stderr
        r_squared[i] = fit.rvalue**2

    # check if r2 is large enogh
    good_fit = r_squared >= r_thresh
    # init upper and lower
    upper = lower = 0
    # compute difference of slopes and their adjecent window slopes
    diff = np.abs(slopes[0:-1] - slopes[1:])
    # compute std of these slopes using gaussian error propagation
    diff_std = np.sqrt(std_err[:-1] ** 2 + std_err[1:] ** 2)
    # check if the slopes are stable
    stable = (
        np.isfinite(diff_std)
        & (diff <= deviation_factor * diff_std)
        & good_fit[:-1]
        & good_fit[1:]
    )

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
    upper = lower + best_len
    return lower, upper

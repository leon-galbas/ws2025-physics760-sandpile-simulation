import logging

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress


def linear_func(x, m, b):
    return m * x + b


def compute_scaling_exponents(
    data: pd.DataFrame,
    window_size: int,
    window_step_size: int,
    r_thresh: float,
    deviation_factor: float,
    manual_bounds: dict = {},
) -> dict:
    """Compute all relevant scaling exponents for a given sandpile model.

    Computes the scaling exponents alpha, tau, lambda as well as the exponents
    gamma_1, gamma_2, gamma_3 and their inverse, obtained from linear log-log fits
    of PDFs or conditional expectation values.

    The additional arguments specify the heuristic used to calculate the scaling window.

    The data is returned as a dictionary were each key is an exponent and the
    coresponding value is a dictionary of fit parameters.

    Args:
        data (pd.DataFrame): Dataframe containing the measured avalanche
            variables t, s, l of a sandpile.
        window_size (int): Number of points per sliding regression window.
        window_step_size (int): Step size (in points) between consecutive windows.
        r_thresh (float): Minimum R^2 required for a window to be considered a good
            linear fit.
        deviation_factor (float):  Multiplier controlling how strict the slope-stability condition is.
        manual_bounds (dict, optional): Allows to set manual boundaries for the scaling
            windows. Defaults to {}.

    Returns:
        dict: Calculated scaling exponents.
    """
    s = np.asarray(data["s"], dtype=np.int64)
    t = np.asarray(data["t"], dtype=np.int64)
    l = np.asarray(data["l"], dtype=np.int64)  # noqa: E741
    args_scaling_window = (window_size, window_step_size, r_thresh, deviation_factor)
    exponents = {}
    logging.info("Calculating scaling exponents.")

    # Calculate exponents from probability densities

    man_key = list(manual_bounds.keys())

    def _bounds_for(name: str):
        return manual_bounds.get(name) if name in man_key else None

    exponents["tau"] = get_prob_exponent(
        s, args_scaling_window, bounds=_bounds_for("tau")
    )
    exponents["alpha"] = get_prob_exponent(
        t, args_scaling_window, bounds=_bounds_for("alpha")
    )
    exponents["lambda"] = get_prob_exponent(
        l, args_scaling_window, bounds=_bounds_for("lambda")
    )

    logging.info("Calculating conditional scaling exponents.")
    # Calculate exponents from conditional expectation values
    exponents["gamma_1"] = get_cond_exponent(
        t, s, args_scaling_window, bounds=_bounds_for("gamma_1")
    )
    exponents["inv_gamma_1"] = get_cond_exponent(
        s,
        t,
        args_scaling_window,
        bounds=_bounds_for("inv_gamma_1"),
    )
    exponents["gamma_2"] = get_cond_exponent(
        l, s, args_scaling_window, bounds=_bounds_for("gamma_2")
    )
    exponents["inv_gamma_2"] = get_cond_exponent(
        s,
        l,
        args_scaling_window,
        bounds=_bounds_for("inv_gamma_2"),
    )
    exponents["gamma_3"] = get_cond_exponent(
        l, t, args_scaling_window, bounds=_bounds_for("gamma_3")
    )
    exponents["inv_gamma_3"] = get_cond_exponent(
        t,
        l,
        args_scaling_window,
        bounds=_bounds_for("inv_gamma_3"),
    )

    return exponents


def get_prob_exponent(
    x: np.ndarray, args_scaling_window: tuple[int, int, float, float], bounds=None
) -> tuple[np.ndarray, dict]:
    """Calculate probability density scaling exponent.

    This assumes that P(X=x) ~ x^(1-a). To obtain a, the function fits log(P(X=x))
    against log(x) linearly and returns a = 1-slope and its error.

    Args:
        x (np.ndarray): Array of x values.
        args_scaling_window (tuple[int, int, float, float]): The parameters for
            determining the scaling window

    Returns:
        np.array: The values [log(x), log(y), Delta_log(y)] for plotting
        dict: The fit parameters.
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

    # Prepare data for fitting
    log_x = np.log10(x)
    log_P = np.log10(P)
    log_P_err = P_err / (10 * P)
    if bounds == None:
        lower, upper = get_scaling_window(log_x, log_P, *args_scaling_window)
    else:
        lower, upper = bounds

    # Do linear log-log fit
    popt, pcov = curve_fit(
        linear_func,
        log_x[lower:upper],
        log_P[lower:upper],
        sigma=log_P_err[lower:upper],
        absolute_sigma=True,
    )
    exp, intercept = popt
    std_err_stat = np.sqrt(np.diag(pcov))[0]
    intercept_stderr = np.sqrt(np.diag(pcov))[1]

    # Calculate systematic error and combined error
    std_err_sys = estimate_systematic_window_error(
        log_x, log_P, log_P_err, lower, upper
    )
    std_err = np.sqrt(std_err_stat**2 + std_err_sys**2)

    # Return values and parameters
    values = np.array([log_x, log_P, log_P_err])
    fit_params = {
        "exponent": 1 - exp,
        "std_err": std_err,
        "intercept": intercept,
        "intercept_stderr": intercept_stderr,
        "lower": lower,
        "upper": upper,
    }
    return values, fit_params


def get_cond_exponent(
    x: np.ndarray,
    y: np.ndarray,
    args_scaling_window: tuple[int, int, float, float],
    bounds=None,
) -> tuple[np.ndarray, dict]:
    """Calculate conditional expectation value scaling exponents.

    This assumes that E[Y|X=x] ~ x^a. To obtain a, the function fits log(E[Y|X=x])
    against log(x) linearly and returns a = slope and its error.

    Args:
        x (np.ndarray): Array of x values.
        y (np.ndarray): Array of y values.
        args_scaling_window (tuple[int, int, float, float]): The parameters for
            determining the scaling window

    Returns:
        np.array: The values [log(x), log(y), Delta_log(y)] for plotting
        dict: The fit parameters.
    """
    # Prepare values for calculating E, Delta_E
    unique_y, y_idx = np.unique(y, return_inverse=True)
    N_y = np.bincount(y_idx)
    sum_x = np.bincount(y_idx, weights=x)
    sum_x2 = np.bincount(y_idx, weights=x**2)

    # Calculate conditional expectation values and their statistical stderr
    with np.errstate(divide="ignore", invalid="ignore"):
        E = sum_x / N_y
        sum_sq_diff = np.maximum(sum_x2 - N_y * E**2, 0)
        var_cond = sum_sq_diff / (N_y - 1)
        sigma_E = np.sqrt(var_cond / N_y)
    E[N_y == 0] = np.nan
    sigma_E[N_y <= 1] = np.nan

    # Filter valid values
    valid = (
        (unique_y > 0) & (E > 0) & np.isfinite(E) & (sigma_E > 0) & np.isfinite(sigma_E)
    )
    y_val = unique_y[valid]
    E_val = E[valid]
    sigma_E_val = sigma_E[valid]

    # Prepare data for fitting
    log_y = np.log10(y_val)
    log_E = np.log10(E_val)
    sigma_log_E = sigma_E_val / (10 * E_val)
    if bounds == None:
        lower, upper = get_scaling_window(log_y, log_E, *args_scaling_window)
    else:
        lower, upper = bounds

    # Do linear log-log fit
    popt, pcov = curve_fit(
        f=linear_func,
        xdata=log_y[lower:upper],
        ydata=log_E[lower:upper],
        sigma=sigma_log_E[lower:upper],
        absolute_sigma=True,
    )
    exp, intercept = popt
    std_err_stat = np.sqrt(np.diag(pcov))[0]
    intercept_stderr = np.sqrt(np.diag(pcov))[1]

    # Calculate systematic error and combined error
    std_err_sys = estimate_systematic_window_error(
        log_y, log_E, sigma_log_E, lower, upper
    )
    std_err = np.sqrt(std_err_stat**2 + std_err_sys**2)

    # Return values and parameters
    values = np.array([log_y, log_E, sigma_log_E])
    fit_params = {
        "exponent": exp,
        "std_err": std_err,
        "intercept": intercept,
        "intercept_stderr": intercept_stderr,
        "lower": lower,
        "upper": upper,
    }
    return values, fit_params


def get_scaling_window(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    window_step_size: int,
    r_thresh: float,
    deviation_factor: float,
) -> tuple[int, int]:
    """Find a heuristic scaling window in log-log space using a sliding-window
    regression and a slope-stability criterion.

    Args:
        x (array-like): The x-values in log-log space.
        y (array-like): The y-values in log-log space.
        window_size (int): Number of points per sliding regression window.
        window_step_size (int): Step size (in points) between consecutive windows.
        r_thresh (float): Minimum R^2 required for a window to be considered a
            good linear fit.
        deviation_factor (float): Multiplier controlling how strict the
            slope-stability condition is.

    Returns:
        tuple[int, int]: A tuple (lower, upper) representing the indices
            marking the start and end of the longest linear region.
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


def estimate_systematic_window_error(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    lower: int,
    upper: int,
    max_variation: int = 3,
) -> float:
    """Estimates systematic error by perturbing the heuristic scaling window boundaries
    and computing the standard deviation of the resulting fit slopes.

    Args:
        x (np.ndarray): The x values.
        y (np.ndarray): The y values.
        y_err (np.ndarray): The y errors.
        lower (int): Lower end of the scaling window.
        upper (int): Upper end of the scaling window.
        max_variation (int, optional): Perturbation of the window boundaries.
            Defaults to 3.

    Returns:
        float: The systematic slope error of a linear fit of x against y.
    """
    slopes = []

    # Constrain variations to array boundaries
    l_min = max(0, lower - max_variation)
    l_max = min(len(x) - 3, lower + max_variation)
    u_min = max(3, upper - max_variation)
    u_max = min(len(x), upper + max_variation)

    for l in range(l_min, l_max + 1):
        for u in range(u_min, u_max + 1):
            # Ensure sufficient points for a meaningful fit
            if u - l >= 3:
                try:
                    popt, _ = curve_fit(
                        linear_func,
                        x[l:u],
                        y[l:u],
                        sigma=y_err[l:u],
                        absolute_sigma=True,
                    )
                    slopes.append(popt[0])
                except RuntimeError:
                    continue

    if len(slopes) > 1:
        return np.std(slopes, ddof=1)
    else:
        return 0.0

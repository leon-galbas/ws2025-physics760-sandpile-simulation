import logging
import pickle
from collections import deque
from datetime import datetime
from os import path

import numpy as np
import pandas as pd
import torch
from scipy.stats import linregress
from tqdm import tqdm

from src.utils import read_config


class SandpileModel:
    """A cellular automaton model for modeling sand piles.

    The model is based on the reference paper below.

    Reference:

    Christensen, K., Fogedby, H. C., & Jeldtoft Jensen, H. (1991).
    Dynamical and spatial aspects of sandpile cellular automata.
    Journal of statistical physics, 63(3), 653-684.
    """

    # class variables
    BOUNDARY_CONDITIONS = ["open", "closed"]
    PERTURBATIONS = ["conservative", "nonconservative"]
    # class attributes (constant)
    _N: int
    _d: int
    _z_c: int
    _boundary_condition: str
    _perturbation: str
    _boundary_mask: torch.Tensor
    # class attributes (evolving)
    z: torch.Tensor
    _r_0: tuple
    _data: pd.DataFrame
    _macro_time: int

    # ---------- CONSTRUCTOR ----------

    def __init__(
        self,
        N: int,
        d: int,
        z_c: int = None,
        boundary_condition: str = "open",
        perturbation: str = "conservative",
        z_init: torch.Tensor | str = None,
    ):
        """Initializes the sandpile model.

        Args:
            N (int): The lattice size in each dimension.
            d (int): The lattice dimension.
            z_c (int, optional): The critical slope. Defaults to 2d-1.
            boundary_condition (str, optional): Boundary Condition. Defaults to "open".
            perturbation (str, optional): Perturbation type. Defaults to "conservative".
            z_init (torch.Tensor, optional): Initial slope values. Defaults to None.
        """
        # check values
        if type(N) is not int or N <= 0:
            raise ValueError("N must be >0 and of type 'int'.")
        if type(d) is not int or d <= 0:
            raise ValueError("d must be >0 and of type 'int'.")
        if boundary_condition not in type(self).BOUNDARY_CONDITIONS:
            raise ValueError(
                f"The given boundary_condition '{boundary_condition}' is not implemented. Valid values are {type(self).BOUNDARY_CONDITIONS}."
            )
        if perturbation not in type(self).PERTURBATIONS:
            raise ValueError(
                f"The given perturbation '{perturbation}' is not implemented. Valid values are {type(self).PERTURBATIONS}."
            )

        # init attributes
        self._N = N
        self._d = d
        if z_c is not None:
            self._z_c = z_c
        else:
            self._z_c = 2 * d - 1
        self._boundary_condition = boundary_condition
        self._perturbation = perturbation
        self._r_0 = None

        # init z tensor
        z_shape = (N,) * d
        if z_init is not None:
            if z_init.size() != z_shape:
                raise ValueError(
                    f"The shape {z_init.size()} of z_init does not match the shape {z_shape} given by arguments N and d!"
                )
            if z_init.dtype not in [torch.int32, torch.int64]:
                raise TypeError(
                    f"The given z_init is of dtype {z_init.dtype}. Must be of type 'torch.int32' or 'torch.int64'."
                )
            self.z = z_init.clone()
        else:
            self.z = torch.zeros(z_shape, dtype=torch.int)

        # init of boundary mask
        self._boundary_mask = torch.empty(z_shape, dtype=self.z.dtype)
        self._update_boundary_mask()

        # collect experiment data
        self._macro_time = 0
        data_schema = {
            "s": "int64",
            "t": "int64",
            "l": "int64",
        }
        self._data = pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in data_schema.items()}
        )
        self._data.loc[0] = 0

        logging.info(f"SandpileModel initialized with {N=}, {d=}, z_c={self._z_c}.")

    # ---------- PUBLIC METHODS ----------

    def burn_in(
        self, window_size: int = 50, check_interval: int = 1000, epsilon: float = None
    ):
        """Executes a "burn-in" phase during which z_mean reaches a stationary state.

        Args:
            window_size (int, optional): Number of samples in the rolling window.
                Defaults to 50.
            check_interval (int, optional): Intervals at which z_mean is computed.
                Defaults to 1000.
            epsilon (float, optional): Slope threshold for stationarity.
        """
        if self.boundary_condition == "closed" and self.perturbation == "conservative":
            raise ValueError(
                "A system with closed boundary conditions and conservative perturbation never reaches a stationary state ."
            )

        # Compute epsilon
        if epsilon is None:
            k = 5
            volume = np.pow(self.N, self.d)
            sigma_m = np.sqrt(12.0 / (volume * np.pow(window_size, 3)))
            epsilon = k * sigma_m

        z_averages = deque(maxlen=window_size)
        burn_in_steps = 0

        logging.info(f"Start traversing towards stationary state. Threshold {epsilon=}")
        # Run model until z_mean reaches stationarity
        while True:
            _, _, _ = self.relax()
            self.perturb()
            burn_in_steps += 1

            # Check stationarity at intervals to minimize overhead
            if burn_in_steps % check_interval == 0:
                z_averages.append(self.z_mean)

                # Evaluate convergence if the rolling window is full
                if len(z_averages) == window_size:
                    x = np.arange(window_size)
                    y = np.array(z_averages)
                    fit = linregress(x, y)

                    # periodically log the z_mean_slope
                    if burn_in_steps % (check_interval * window_size) == 0:
                        logging.info(
                            f"Step {burn_in_steps}: z_mean_slope = {fit.slope}"
                        )

                    # Stop of z_mean slope is below threshold
                    if abs(fit.slope) < epsilon:
                        break

        logging.info(f"Stationary state reached after {burn_in_steps} steps.")
        self._macro_time += burn_in_steps

    def measure(self, num_measurements: int):
        """Performs measurements of avalanche metrics s, t, l.

        Performs full loops of Algorithm 1 in the reference, i.e. relaxations and
        perturbations. Whenever an avalanche event happens, the metrics
        s (total dissipation), t (lifetime) and l (spatial linear size) are tracked.

        Args:
            num_measurements (int): Number of avalanche events to measure.
        """
        # track avalanche characteristics
        sizes = []
        lifetimes = []
        linear_sizes = []  # noqa: E741

        # do macroscopic time steps
        logging.info(f"Performing {num_measurements} measurements of the model...")
        with tqdm(total=num_measurements) as pbar:
            while len(sizes) < num_measurements:
                s, t, l = self.relax()  # noqa: E741
                self.perturb()
                if s > 0:
                    sizes.append(s)
                    lifetimes.append(t)
                    linear_sizes.append(l)
                    pbar.update(1)
                self._macro_time += 1
        logging.info("Done!")

        # save avalanche data
        new_df = pd.DataFrame(
            {
                "s": sizes,
                "t": lifetimes,
                "l": linear_sizes,
            }
        )
        self._data = pd.concat([self._data, new_df])

    # DEPRECATED -> USE BURN IN AND MEASURE
    def step(self, steps: int = 1):
        """Performs time steps of the models macroscopic temporary evolution.

        One macroscopic time step corresponds to one full loop of Algorithm 1 in the
        reference, i.e. one full relaxation and one perturbation.

        Args:
            steps (int, optional): Number of time steps. Defaults to 1.
        """
        logging.warning(
            "This method is deprecated. Use 'burn_in' and 'measure' to perform measurements!"
        )

        # track avalanche characteristics
        s = np.empty(steps, dtype=int)
        t = np.empty(steps, dtype=int)
        l = np.empty(steps, dtype=int)  # noqa: E741
        z_mean = np.empty(steps, dtype=int)

        # do macroscopic time steps
        logging.info(f"Performing {steps} time steps of the model...")
        for i in tqdm(range(steps)):
            s[i], t[i], l[i] = self.relax()  # noqa: E741
            self.perturb()
            z_mean[i] = self.z_mean
        logging.info("Done!")

        # save avalanche data
        start_time = self.time + 1
        macro_time = np.arange(start_time, start_time + steps)
        df = pd.DataFrame(
            {
                "macro_time": macro_time,
                "z_mean": z_mean,
                "s": s,
                "t": t,
                "l": l,
            }
        )
        self._macro_time += steps

        return df

    def relax(self) -> tuple[int, int, int]:
        """Performs the relaxation of z as described in the reference.

        Returns:
            int: Total dissipation s of the avalanche.
            int: Lifetime t of the avalanche.
            int: Spatial linear size l of the avalanche.
        """
        # check for valid boundary condition
        if self._boundary_condition not in type(self).BOUNDARY_CONDITIONS:
            raise ValueError(
                f"The perturbation {self._perturbation=} is not implemented!"
            )

        # track avalanche parameters
        total_dissipation_s = 0
        lifetime_t = 0
        cumulative_toppled_mask = torch.zeros_like(self.z, dtype=torch.bool)
        spatial_size_l = 0

        # perform relaxation steps until z(r) <= z_c everywhere
        while True:
            # Identify all unstable sites simultaneously
            unstable_mask = self.z > self._z_c
            if not unstable_mask.any():
                break

            # Cast to z's dtype to calculate sand transfer
            firings = unstable_mask.to(self.z.dtype)

            # Update avalanche parameters
            f_alpha = firings.sum().item()
            total_dissipation_s += f_alpha
            lifetime_t += 1
            cumulative_toppled_mask |= unstable_mask

            # Process dimensional shifts (nearest-neighbor transfers)
            for dim in range(self._d):
                # Slices for receiving from r +- e_i
                idx_plus = [slice(None)] * self._d
                idx_plus[dim] = slice(1, None)
                idx_minus = [slice(None)] * self._d
                idx_minus[dim] = slice(None, -1)

                # Add sand tumbling from adjacent sites and subtract from centers
                self.z[tuple(idx_plus)] += (
                    firings[tuple(idx_minus)] - firings[tuple(idx_plus)]
                )
                self.z[tuple(idx_minus)] += (
                    firings[tuple(idx_plus)] - firings[tuple(idx_minus)]
                )

            # Enforce boundary condition
            self.z.mul_(self._boundary_mask)

        # calculate spatial linear size l auf the avalanche
        if total_dissipation_s > 0 and self._r_0 is not None:
            # Get N-dimensional coordinates of all sites that toppled
            toppled_coords = torch.nonzero(cumulative_toppled_mask, as_tuple=False)
            r_0_tensor = torch.tensor(self._r_0)

            # Calculate maximum coordinate distance from r_0
            max_distances_per_dim = torch.max(
                torch.abs(toppled_coords - r_0_tensor), dim=0
            )[0]
            spatial_size_l = torch.max(max_distances_per_dim).item()

        return total_dissipation_s, lifetime_t, spatial_size_l

    def perturb(self):
        """Performs a perturbation of z as described in the reference."""
        # choose random lattice position
        r = tuple(np.random.randint(0, self._N, size=self._d))
        self._r_0 = r

        # perform perturbation
        match self._perturbation:
            case "conservative":
                self.z[r] += self._d
                for dim in range(self._d):
                    neighbor = list(r)
                    if neighbor[dim] >= 1:
                        neighbor[dim] -= 1
                        self.z[tuple(neighbor)] -= 1
            case "nonconservative":
                self.z[r] += 1
            case _:
                raise ValueError(
                    f"The perturbation {self._perturbation=} is not implemented!"
                )

    def save(self, filename=None):
        """Save the model to a file

        Args:
            name (str, optional): Name of the model file. Defaults to '<timestamp>.pkl'.
        """
        model_dir = read_config("model_dir")
        if filename is not None:
            filepath = path.join(model_dir, f"{filename}")
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            filepath = path.join(model_dir, f"{timestamp}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logging.info(f"Model saved to '{filepath}'.")

    # ---------- PRIVATE METHODS ----------

    def _update_boundary_mask(self):
        """Calculates the boundary mask used to enforce boundary conditions during relaxation."""
        self._boundary_mask = torch.ones(self.z.size(), dtype=self.z.dtype)
        for dim in range(self._d):
            # Both Open (Eq 5) and Closed (Eq 6) set z=0 at r_j = 0
            idx_0 = [slice(None)] * self._d
            idx_0[dim] = 0
            self._boundary_mask[tuple(idx_0)] = 0

            # Closed (Eq 6) also sets z=0 at r_j = N
            if self._boundary_condition == "closed":
                idx_N = [slice(None)] * self._d
                idx_N[dim] = -1
                self._boundary_mask[tuple(idx_N)] = 0

    # ---------- GETTERS/SETTERS ----------

    @property
    def N(self) -> int:
        return self._N

    @property
    def d(self) -> int:
        return self._d

    @property
    def z_c(self) -> int:
        return self._z_c

    @property
    def boundary_condition(self) -> str:
        return self._boundary_condition

    @boundary_condition.setter
    def boundary_condition(self, new_condition):
        # check value
        if new_condition not in type(self).BOUNDARY_CONDITIONS:
            raise ValueError(
                f"The given boundary_condition '{new_condition}' is not implemented. Valid values are {type(self).BOUNDARY_CONDITIONS}."
            )

        # update boundary condition and boundary mask
        self._boundary_condition = new_condition
        self._update_boundary_mask()

    @property
    def perturbation(self) -> str:
        return self._perturbation

    @property
    def r_0(self) -> tuple:
        return self._r_0

    @property
    def z_mean(self) -> float:
        return float(torch.mean(self.z, dtype=torch.float))

    @property
    def time(self) -> int:
        return self._macro_time

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def num_measurements(self) -> int:
        return len(self._data)

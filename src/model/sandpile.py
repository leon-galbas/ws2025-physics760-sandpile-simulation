import numpy as np
import torch


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
    # class attributes
    _N: int
    _d: int
    _z_c: int
    _boundary_condition: str
    _perturbation: str
    z: torch.Tensor
    t: int
    _z_mean_timeseries: list[float]
    boundary_mask: torch.Tensor
    def __init__(
        self,
        N: int,
        d: int,
        z_c: int,
        boundary_condition: str = "open",
        perturbation: str = "conservative",
        z_init: torch.Tensor = None,
    ):
        """Initializes the sandpile model.

        Args:
            N (int): The lattice size in each dimension.
            d (int): The lattice dimension.
            z_c (int): The critical slope.
            boundary_condition (str, optional): Boundary Condition. Defaults to "open".
            perturbation (str, optional): Perturbation type. Defaults to "conservative".
            z_init (torch.Tensor, optional): Initial slope values. Defaults to None.
        """
        # check values
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
        self._z_c = z_c
        self._boundary_condition = boundary_condition
        self._perturbation = perturbation
        # init z tensor
        z_shape = (N,) * d
        if z_init is not None:
            if z_init.size() != z_shape:
                raise ValueError(
                    f"The shape {z_init.size()} of z_init does not match the shape {z_shape} given by arguments N and d!"
                )
            if z_init.dtype != torch.int:
                raise TypeError(
                    f"The given z_init is of dtype {z_init.dtype}. Must be of type 'torch.int'."
                )
            self.z = z_init
        else:
            self.z = torch.zeros(z_shape, dtype=torch.int)

        # track time steps
        self.time = 0

        # historize average values of z
        self._z_mean_timeseries = [self.z_mean]
        
        # init of boundary mask 
        self.boundary_mask= torch.ones(z_shape, dtype=self.z.dtype)
        for dim in range(self._d):
            # Both Open (Eq 5) and Closed (Eq 6) set z=0 at r_j = 0
            idx_0 = [slice(None)] * self._d
            idx_0[dim] = 0
            self.boundary_mask[tuple(idx_0)] = 0

            # Closed (Eq 6) also sets z=0 at r_j = N
            if self._boundary_condition == "closed":
                idx_N = [slice(None)] * self._d
                idx_N[dim] = -1
                self.boundary_mask[tuple(idx_N)] = 0

        print(self.boundary_mask)
    def step(self, t: int = 1):
        """Performs t unit time steps of the model temporary evolution.

        Args:
            t (int, optional): Number of time steps. Defaults to 1.
        """
        for i in range(t):
            print(f"before: {self.z}")
            self.relax()#
            print(f"after: {self.z}")
            self.perturb()
            self._z_mean_timeseries.append(self.z_mean)
            self.time += 1
    def relax(self):
        """Performs the relaxation of z as described in the reference."""
        # check for valid boundary condition
        if self._boundary_condition not in type(self).BOUNDARY_CONDITIONS:
            raise ValueError(
                f"The perturbation {self._perturbation=} is not implemented!"
            )

        # perform relaxation steps until z(r) <= z_c everywhere
        while True:
            # Identify all unstable sites simultaneously
            unstable_mask = self.z > self._z_c
            if not unstable_mask.any():
                break

            # Cast to z's dtype to calculate sand transfer
            firings = unstable_mask.to(self.z.dtype)
            
            # Process dimensional shifts (nearest-neighbor transfers)
            for dim in range(self._d):
                # Slices for receiving from r +- e_i
                idx_plus = [slice(None)] * self._d
                idx_plus[dim] = slice(1, None)
                idx_minus = [slice(None)] * self._d
                idx_minus[dim] = slice(None, -1)


                # Add sand tumbling from adjacent sites and subtract from centers
                self.z[tuple(idx_plus)] += firings[tuple(idx_minus)]-firings[tuple(idx_plus)]
                self.z[tuple(idx_minus)] += firings[tuple(idx_plus)]-firings[tuple(idx_minus)]
                

            # Enforce boundary condition     
            self.z.mul_(self.boundary_mask)
            

    def perturb(self):
        """Performs a perturbation of z as described in the reference."""
        # choose random lattice position
        r = tuple(np.random.randint(0, self._N, size=self._d))

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

    def z_mean_timeseries(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns a timeseries of the models mean z values.

        The timeseries is returned as two separate numpy arrays.
        The first array are the time steps, the second are the z_mean values.

        Returns:
            tuple[np.array, np.array]: Timesteps, Mean z values.
        """
        ts = np.arange(0, self.time + 1)
        z_means = np.array(self._z_mean_timeseries)
        return ts, z_means

    # ---------- GETTERS ----------

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

    @property
    def perturbation(self) -> str:
        return self._perturbation

    @property
    def z_mean(self) -> float:
        return float(torch.mean(self.z, dtype=torch.float))

import numpy as np
import torch


class SandpileModel:
    """A cellular automaton model for modeling sand piles.

    The model is based on the following reference:

    Christensen, K., Fogedby, H. C., & Jeldtoft Jensen, H. (1991).
    Dynamical and spatial aspects of sandpile cellular automata.
    Journal of statistical physics, 63(3), 653-684.
    """

    BOUNDARY_CONDITIONS = ["open", "closed"]
    PERTURBATIONS = ["conservative", "nonconservative"]

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
        self.N = N
        self.d = d
        self.z_c = z_c
        self.boundary_condition = boundary_condition
        self.perturbation = perturbation
        # init z tensor
        z_shape = (N,) * d
        if z_init is not None:
            if z_init.size() != z_shape:
                raise ValueError(
                    f"The shape {z_init.size()} of z_init does not match the shape {z_shape} given by arguments N and d!"
                )
            self.z = z_init
        else:
            self.z = torch.zeros(z_shape, dtype=torch.int)
        # start at time 0
        self.time = 0
        # historize average values of z
        self.z_mean = torch.mean(self.z)
        self.z_mean_timeseries = [self.z_mean]

    def step(self, t: int = 1):
        """Performs t unit time steps of the model temporary evolution.

        Args:
            t (int, optional): Number of time steps. Defaults to 1.
        """
        for i in range(t):
            self.relax()
            self.perturb()
            self.z_mean = torch.mean(self.z)
            self.z_mean_timeseries.append(self.z_mean)
            self.time += 1

    def relax(self):
        # TODO
        pass

    def perturb(self):
        # choose random lattice position
        r = tuple(np.random.randint(0, self.N, size=self.d))
        # perform perturbation
        match self.perturbation:
            case "conservative":
                self.z[r] += self.d
                for i in range(self.d):
                    neighbor = list(r)
                    if neighbor[i] >= 1:
                        neighbor[i] -= 1
                        self.z[tuple(neighbor)] -= 1
            case "nonconservative":
                self.z[r] += 1
            case _:
                raise ValueError(f"The value {self.perturbation=} is not implemented!")

    def get_z_mean_timeseries(self) -> tuple[np.array, np.array]:
        """Returns a timeseries of the models mean z values.

        The timeseries is returned as two separate numpy arrays.
        The first array are the time steps, the second are the z_mean values.

        Returns:
            tuple[np.array, np.array]: Timesteps, Mean z values.
        """
        ts = np.arange(0, self.time + 1)
        z_means = np.array(self.z_mean_timeseries)
        return ts, z_means

import logging

from src.calc.simulation import run_simulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

N = 40
d = 2
boundary_condition = "open"
perturbation = "conservative"
num_measurements = 1e6

run_simulation(N, d, boundary_condition, perturbation, num_measurements)

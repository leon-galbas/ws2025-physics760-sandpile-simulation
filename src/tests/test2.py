import logging

from src.calc.scaling_exponents import compute_scaling_exponents
from src.model.sandpile import SandpileModel

# from src.model.io import load_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

N = 40
d = 2

model = SandpileModel(N, d)
model.burn_in()
model.measure(num_measurements=1e6)
model.save(f"N{N}d{d}.pkl")
# model = load_model("100k_steps.pkl")
df = model.data

# plot
# mask = df["t"] > 0
# x = df["macro_time"]
# x = x[mask]#
# y = df["t"]
# y = y[mask]

# plt.plot(x, y)
# plt.show()


compute_scaling_exponents(data=df)

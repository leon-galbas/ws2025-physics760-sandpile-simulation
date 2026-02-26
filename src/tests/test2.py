import logging

from src.calc.scaling_exponents import compute_scaling_exponents
from src.model.io import load_model

# from src.model.sandpile import SandpileModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

N = 10
d = 3
z_c = 5

# model = SandpileModel(N, d, z_c)
# model.step(100000)
# model.save("100k_steps.pkl")
model = load_model("100k_steps.pkl")
df = model.data

# plot
mask = df["t"] > 0
x = df["macro_time"]
x = x[mask]
y = df["t"]
y = y[mask]

# plt.plot(x, y)
# plt.show()


compute_scaling_exponents(data=df)

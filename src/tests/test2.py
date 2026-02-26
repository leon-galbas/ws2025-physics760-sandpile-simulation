import matplotlib.pyplot as plt

from src.model.io import load_model

N = 10
d = 3
z_c = 5

# model
# model = SandpileModel(N, d, z_c)
# model.step(10000)
# model.save()
model = load_model("2026-02-26_21:30:50")
df = model.data

# plot
mask = df["t"] > 0
x = df["macro_time"]
x = x[mask]
y = df["t"]
y = y[mask]

plt.plot(x, y)
plt.show()

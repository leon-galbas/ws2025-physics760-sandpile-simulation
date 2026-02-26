import matplotlib.pyplot as plt

from model.sandpile import SandpileModel

N = 10
d = 2
z_c = 5

model = SandpileModel(N, d, z_c)

model.step(10000)

t, zmean = model.z_mean_timeseries()

plt.plot(t, zmean)
plt.show()

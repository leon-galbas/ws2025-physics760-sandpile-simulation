import polars as pl
import matplotlib.pyplot as plt
path = "data/burn_in_stats.pq"

df = pl.read_parquet(path)
print(df)

for i in range (df.height):
    row= df[i]
    print(row)
    N = row["N"][0]
    d = row["d"][0]
    init = row["init"][0]
    boundary_condition = row["boundary"][0]
    perturbation = row["perturb"][0]
    z_mean_hist = row["z_mean_hist"][0] 
    bit= row["burn_in_time"][0]
    model_name = f"N{N}d{d}_{boundary_condition}_{perturbation}_{init}"
    plt.plot(range(len(z_mean_hist)), z_mean_hist)
    plt.xlabel(r"$\tau$",fontsize=20)
    plt.ylabel(r"$\langle z\rangle (\tau)$", fontsize=20)
    plt.savefig(f"figures/burn_in_plots/{model_name}_burn_in.png", dpi=300)
    plt.clf()
    
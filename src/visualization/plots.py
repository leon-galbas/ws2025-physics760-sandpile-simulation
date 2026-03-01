import polars as pl
import matplotlib.pyplot as plt
import os 
from src.model.sandpile import SandpileModel
from src.model.io import *


def plot_data(df, model_name): 
    plot_path = "figures/loglog_plots"
    for measure in df.columns:
        counts =df[measure].value_counts().sort_index()
        plt.plot(counts)
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(f"{plot_path}/{model_name}_{measure}.png", dpi=300)
        plt.clf()   
model_path= "models"
model_files =  os.listdir(model_path) 
for f in model_files: 
    if(f.endswith(".pkl")): 
        model = load_model(f)
        df = model.data
        plot_data(df, f[:-4])
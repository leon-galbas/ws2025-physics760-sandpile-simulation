import logging
import time
from itertools import product
from os import path

import torch

from src.model.io import load_model, model_exists
from src.model.capture_sandpile import CaptureSandpileModel
from src.utils import append_dict_to_parquet, read_config
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def capture(boundary_condition, perturbation,N, time_steps, r= None, burn_in= False, prep=None): 
    model = CaptureSandpileModel(
            N,
            time_steps,
            boundary_condition=boundary_condition,
            perturbation=perturbation,
            z_init=0
        )
    
    if burn_in: 
        model.burn_in(epsilon=0.00397)
    if prep is not None: 
        for p in prep: 
            model.step(r=p,capture=False)
    while model._micro_time< model.capture_time: 
        model.step(r=r)

    #print(model.sandpile_time_series)
    return model


def plot_heatmaps(model):
    for z in model.sandpile_time_series: 
        fig, ax = plt.subplots()
        im = ax.imshow(z)
        N = len(z[0])
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(range(N), labels=range(N),
                    rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(range(N))), labels=range(N))

        for i in range(N):
            for j in range(N):
                text = ax.text(j, i, str(z[i, j].item()),
                            ha="center", va="center", color="w")

        plt.show()


def animate_heatmaps(model, filename): 
    time_series = model.sandpile_time_series.detach().cpu().numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(
        time_series[0],
        cmap="hot",
        vmin=time_series.min(),
        vmax=time_series.max(),
        animated= True)
    T = model.capture_time
    N = model.N
    
    ax.set_xticks(range(N), labels=range(N),
                        rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(range(N))), labels=range(N))
    value_texts = []

    def draw_values(frame):

        # remove previous frame's texts
        for txt in value_texts:
            txt.remove()
        value_texts.clear()

        # draw current frame's texts
        for i in range(N):
            for j in range(N):
                txt = ax.text(
                    j, i,
                    str(time_series[frame][i, j]),
                    ha="center", va="center", color="w"
                )
                value_texts.append(txt)

    draw_values(0)
    def update(frame):
        
        im.set_data(time_series[frame])
        ax.set_title(f"t = {frame}")
        draw_values(frame)
        return [im]
    
    ani = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=170 ,   # milliseconds between frames
        blit=True
    )
    ani.save(filename, writer="pillow", fps=10)

p=(3,3)
prep= [p,p,p,(4,3),(4,3),(4,3),(3,4),(3,4)]
model = capture("open", "nonconservative", 5, 5, burn_in=False, prep=prep, r=p)
animate_heatmaps(model, "figures/heatmaps/relaxation_example.gif")
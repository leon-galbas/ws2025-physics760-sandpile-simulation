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
import numpy as np 
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
        model.burn_in()
        model._macro_time= 0
    if prep is not None: 
        for p in prep: 
            model.step(r=p,capture=False)
        model._macro_time= 0
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


def animate_heatmaps(model, filename, single_Frames=False, annotation= True, individual_number= True): 
    time_series = model.sandpile_time_series.detach().cpu().numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(
        time_series[0],
        cmap="turbo",
        vmin=time_series.min(),
        vmax=time_series.max(),
        animated= True)
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(time_series.min(), time_series.max()+1), label= r"$z$")
    lables = np.arange(time_series.min(), time_series.max()+1).astype(str)
    lables[3]= r"3$=z_c$"
    cbar.ax.set_yticklabels(lables)
   

    T = model.capture_time
    N = model.N
    if individual_number: 
        ax.set_xticks(range(N), labels=range(N),
                            rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(N), labels=range(N))
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

    if (annotation):
        draw_values(0)
    tau = 0
    def update(frame):
        nonlocal tau
        
        im.set_data(time_series[frame])
        ax.set_title(rf"$t = {model.macro_times[frame]},\ \tau = {tau}$")
        if frame == 0 or model.macro_times[frame]== model.macro_times[frame-1]: 
            tau+= 1
        else: 
            tau = 0 
        if (annotation):
            draw_values(frame)
        return [im]
    if single_Frames:
        
        for i in range(T): 
            update(i)
            fig.savefig(f"{filename}/frame-{i}.png", 
            dpi=150,
            transparent=True,
            facecolor="none",
            edgecolor="none"
            )
            
    else:
        tau = 0 
        ani = FuncAnimation(
            fig,
            update,
            frames=T,
            interval=170 ,   
            blit=True
        )
        ani.save(filename, writer="pillow", fps=10)


p=(3,3)
prep= [p,]* 35#[p,p,p,(4,3),(4,3),(4,3),(3,4),(3,4)]
model = capture("open", "nonconservative", 5, 9, burn_in=False, prep=prep, r=p)

animate_heatmaps(model, "figures/heatmaps/relaxation_example", single_Frames=True)


model = capture("open", "nonconservative", 10, 500, burn_in=False)
animate_heatmaps(model, "figures/heatmaps/burn_in", single_Frames=True)

model = capture("closed", "nonconservative", 40, 500, burn_in=True)
animate_heatmaps(model, "figures/heatmaps/stationary", single_Frames=True, annotation=False, individual_number= False)
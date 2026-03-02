import polars as pl
import matplotlib.pyplot as plt
import os 
from src.model.sandpile import SandpileModel
from src.model.io import *
from src.calc.scaling_exponents import *
import numpy as np 
import logging


def plot_scaling_exponents(df, model_name, window_size, window_step_size, r_thresh, deviation_factor): 
    plot_path = "figures/loglog_plots"
   
    exponents = compute_scaling_exponents(df, window_size, window_step_size, r_thresh, deviation_factor)#25,1,0.8,2.0
    for key, exp in exponents.items():
    
        values = exp[0]
        parms  = exp[1] 
        plt.plot(values[0], values[1])
        lin_regress_x =np.linspace(np.min(values[0]),np.max(values[0]),10)
        
        if key in ["tau", "alpha", "lambda"]: 
            lin_regress_y= lin_regress_x * (1-parms["exponent"])+parms["intercept"]
        else:
            lin_regress_y= lin_regress_x *parms["exponent"]+parms["intercept"]

        plt.plot(lin_regress_x, lin_regress_y)
        plt.axvline(values[0][parms["lower"]], linestyle=':', linewidth=1)
        plt.axvline(values[0][parms["upper"]], linestyle=':', linewidth=1)
        plt.savefig(f"{plot_path}/{model_name}_{key}.png", dpi=300)
        plt.clf()   







model_path= "models"
model_files =  os.listdir(model_path) 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
 


def run(filename, window_size, window_step_size, r_thresh, deviation_factor): 
    if(filename.endswith(".pkl")): 
        model = load_model(filename)
        df = model.data
    logging.info(f"trying to create log log plots for {filename}")

    try: 
        plot_scaling_exponents(df, filename[:-4], window_size, window_step_size, r_thresh, deviation_factor)
        logging.info(f"sucess")
    except Exception as e : 
        logging.error(f"failed to create log log plots for {filename}: \n {e}")

run("N40d2_open_nonconservative.pkl",25,1,0.8,2.0)

run("N40d2_open_conservative.pkl",25,1,0.8,2.0)

run("N40d2_closed_nonconservative.pkl",25,1,0.8,2.0)

run("N20d3_open_nonconservative.pkl",10,1,0.8,2.0)

run("N20d3_open_conservative.pkl",10,1,0.8,2.0)

run("N20d3_closed_nonconservative.pkl",10,1,0.8,2.0)

run("N20d4_open_nonconservative.pkl",10,1,0.8,2.0)

run("N20d4_open_conservative.pkl",10,1,0.8,2.0)

run("N20d4_closed_nonconservative.pkl",10,1,0.8,2.0)





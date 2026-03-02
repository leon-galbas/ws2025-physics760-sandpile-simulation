import matplotlib.pyplot as plt
import os 
from src.model.sandpile import SandpileModel
from src.model.io import *
from src.calc.scaling_exponents import *
import numpy as np 
import logging
import pandas as pd
import math
model_id= {
    "N40d2_open_nonconservative": "A", 
    "N40d2_open_conservative": "B",
    "N40d2_closed_nonconservative": "C",
    "N20d3_open_nonconservative": "D", 
    "N20d3_open_conservative": "E",
    "N20d3_closed_nonconservative": "F",
    "N20d4_open_nonconservative": "G", 
    "N20d4_open_conservative": "H",
    "N20d4_closed_nonconservative": "I"
 
}


latex_conv_table= {
    "tau": r"$\tau$",
    "alpha": r"$\alpha$",
    "lambda": r"$\lambda$",
    "gamma_1": r"$\gamma_1$",
    "inv_gamma_1": r"$\gamma_1^{-1}$",
    "gamma_2": r"$\gamma_2$",
    "inv_gamma_2": r"$\gamma_2^{-1}$",
    "gamma_3": r"$\gamma_3$",
    "inv_gamma_3": r"$\gamma_3^{-1}$"
 }

label_conv_table= {
    "tau": r"$\log P(S=s)$",
    "alpha": r"$\log P(T=t)$",
    "lambda": r"$\log P(L=l)$",
    "gamma_1": r"$\log E[S|T=t]$",
    "inv_gamma_1": r"$\log E[T|S=s]$",
    "gamma_2": r"$\log E[S|L=l]$",
    "inv_gamma_2": r"$\log E[L|S=s]$",
    "gamma_3": r"$\log E[T|L=l]$",
    "inv_gamma_3": r"$\log E[L|T=t]$"
 }

x_label_conv_table= {
    "tau": r"$\log s$",
    "alpha": r"$\log t$",
    "lambda": r"$\log l$",
    "gamma_1": r"$\log t$",
    "inv_gamma_1": r"$\log s$",
    "gamma_2": r"$\log l$",
    "inv_gamma_2": r"$\log s$",
    "gamma_3": r"$\log l$",
    "inv_gamma_3": r"$\log t$"
 }
def decimals_from_err(err):
    return 0 if err == 0 else max(0, -math.floor(math.log10(abs(err))))
def plot_scaling_exponents(df, model_name, window_size, window_step_size, r_thresh, deviation_factor, exponent_data, do_errors= False, cond_bounds={}): 
    plot_path = "figures/loglog_plots"
    data = {}
    exponents = compute_scaling_exponents(df, window_size, window_step_size, r_thresh, deviation_factor,cond_bounds)#25,1,0.8,2.0
    #exponent_data.append(exponents)
    for key, exp in exponents.items():
        values = exp[0]
        parms  = exp[1] 
        x = np.asarray(values[0])
        y = np.asarray(values[1])
        yerr = np.asarray(values[2])

       
        plt.plot(x, y, label=label_conv_table[key])

        if do_errors:
            band_color = "tab:orange"

        
            plt.plot(x, y + yerr, color=band_color, lw=1.2, ls="--", label=r"$+\,\sigma$")
            plt.plot(x, y - yerr, color=band_color, lw=1.2, ls="--", label=r"$-\,\sigma$")

        
            plt.fill_between(x, y - yerr, y + yerr, color=band_color, alpha=0.15, linewidth=0)
        #plt.plot(values[0], values[1], label= f"P({latex_conv_table[key]})")
        lin_regress_x =np.linspace(np.min(values[0]),np.max(values[0]),10)
        
        if key in ["tau", "alpha", "lambda"]: 
            lin_regress_y= lin_regress_x * (1-parms["exponent"])+parms["intercept"]
        else:
            lin_regress_y= lin_regress_x *parms["exponent"]+parms["intercept"]

        plt.plot(lin_regress_x, lin_regress_y, label= "WLS fit")
        plt.axvline(values[0][parms["lower"]], linestyle=':', linewidth=2, label="fitting window")
        plt.axvline(values[0][parms["upper"]], linestyle=':', linewidth=2)
        plt.legend()
        plt.xlabel(x_label_conv_table[key], fontsize= "18")
        plt.ylabel(label_conv_table[key], fontsize= "18")
        plt.savefig(f"{plot_path}/{model_name}_{key}.png", dpi=300)
        plt.clf()   
        data[f"model"]=model_id[model_name]
        sig = decimals_from_err(parms["std_err"])
        #print(f"{sig=}")
        data[latex_conv_table[key]]=f"$ ({parms["exponent"]:.{sig}f}"+r"\pm"+f"{parms["std_err"]:.{sig}f}) $"

    d= pd.DataFrame([data])
    exponent_data=pd.concat([exponent_data,d])    
    return exponent_data







model_path= "models"
model_files =  os.listdir(model_path) 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
 


def run(filename, window_size, window_step_size, r_thresh, deviation_factor, exponent_data, cond_bounds={}): 
    if(filename.endswith(".pkl")): 
        model = load_model(filename)
        df = model.data
        #print(exponent_data)
    logging.info(f"trying to create log log plots for {filename}")

    try: 
        exponent_data= plot_scaling_exponents(df, filename[:-4], window_size, window_step_size, r_thresh, deviation_factor, exponent_data, cond_bounds=cond_bounds)
        logging.info(f"sucess")
    except Exception as e : 
        logging.error(f"failed to create log log plots for {filename}: \n {e}")
    return exponent_data

tab1 = [
    "model", 
     r"$\tau$",
     r"$\alpha$",
     r"$\lambda$"
]
tab2= ["model",
     r"$\gamma_1$",
     r"$\gamma_1^{-1}$",
     r"$\gamma_2$",
     r"$\gamma_2^{-1}$",
     r"$\gamma_3$",
     r"$\gamma_3^{-1}$"
    ]
exponent_data= pd.DataFrame() 
exponent_data=run("N40d2_open_nonconservative.pkl",25,1,0.8,2.0, exponent_data)
exponent_data=run("N20d3_open_nonconservative.pkl",10,1,0.8,2.0, exponent_data)
exponent_data=run("N20d4_open_nonconservative.pkl",10,1,0.8,2.0, exponent_data)
exponent_data.to_latex(buf= "figures/tables/scaling_exp_open_nonconservative.tex",index=False, column_format="c"*len(tab1), columns= tab1)
exponent_data.to_latex(buf= "figures/tables/scaling_exp_open_nonconservative_gamma.tex",index=False, column_format="c"*len(tab2), columns= tab2)

exponent_data= pd.DataFrame() 
exponent_data=run("N40d2_open_conservative.pkl",25,1,0.8,2.0, exponent_data)
exponent_data=run("N20d3_open_conservative.pkl",10,1,0.8,2.0, exponent_data)
exponent_data=run("N20d4_open_conservative.pkl",10,1,0.8,2.0,exponent_data)
exponent_data.to_latex(buf= "figures/tables/scaling_exp_open_conservative.tex",index=False, column_format="c"*len(tab1), columns= tab1)
exponent_data.to_latex(buf= "figures/tables/scaling_exp_open_conservative_gamma.tex",index=False, column_format="c"*len(tab2), columns= tab2)

exponent_data= pd.DataFrame() 
exponent_data=run("N40d2_closed_nonconservative.pkl",25,1,0.8,2.0, exponent_data)
exponent_data=run("N20d3_closed_nonconservative.pkl",10,1,0.8,2.0, exponent_data)
exponent_data=run("N20d4_closed_nonconservative.pkl",10,1,0.8,2.0, exponent_data)
exponent_data.to_latex(buf= "figures/tables/scaling_exp_closed_nonconservative.tex",index=False, column_format="c"*len(tab1), columns= tab1)
exponent_data.to_latex(buf= "figures/tables/scaling_exp_closed_nonconservative_gamma.tex",index=False, column_format="c"*len(tab2), columns= tab2)





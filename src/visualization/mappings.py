model_name_to_id = {
    "N40d2_open_nonconservative": "A",
    "N40d2_open_conservative": "B",
    "N40d2_closed_nonconservative": "C",
    "N20d3_open_nonconservative": "D",
    "N20d3_open_conservative": "E",
    "N20d3_closed_nonconservative": "F",
    "N20d4_open_nonconservative": "G",
    "N20d4_open_conservative": "H",
    "N20d4_closed_nonconservative": "I",
}

latex_conv_table = {
    "tau": r"$\tau$",
    "alpha": r"$\alpha$",
    "lambda": r"$\lambda$",
    "gamma_1": r"$\gamma_1$",
    "inv_gamma_1": r"$\gamma_1^{-1}$",
    "gamma_2": r"$\gamma_2$",
    "inv_gamma_2": r"$\gamma_2^{-1}$",
    "gamma_3": r"$\gamma_3$",
    "inv_gamma_3": r"$\gamma_3^{-1}$",
}

label_conv_table = {
    "tau": r"$\log P(S=s)$",
    "alpha": r"$\log P(T=t)$",
    "lambda": r"$\log P(L=l)$",
    "gamma_1": r"$\log E[S|T=t]$",
    "inv_gamma_1": r"$\log E[T|S=s]$",
    "gamma_2": r"$\log E[S|L=l]$",
    "inv_gamma_2": r"$\log E[L|S=s]$",
    "gamma_3": r"$\log E[T|L=l]$",
    "inv_gamma_3": r"$\log E[L|T=t]$",
}

x_label_conv_table = {
    "tau": r"$\log s$",
    "alpha": r"$\log t$",
    "lambda": r"$\log l$",
    "gamma_1": r"$\log t$",
    "inv_gamma_1": r"$\log s$",
    "gamma_2": r"$\log l$",
    "inv_gamma_2": r"$\log s$",
    "gamma_3": r"$\log l$",
    "inv_gamma_3": r"$\log t$",
}

tab1_columns = ["model", r"$\tau$", r"$\alpha$", r"$\lambda$"]

tab2_columns = [
    "model",
    r"$\gamma_1$",
    r"$\gamma_1^{-1}$",
    r"$\gamma_2$",
    r"$\gamma_2^{-1}$",
    r"$\gamma_3$",
    r"$\gamma_3^{-1}$",
]

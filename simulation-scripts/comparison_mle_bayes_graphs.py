import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


mle_data = pd.read_csv("./simulation-results/mle_vs_kappa.csv", header=None)
bayes_data = pd.read_csv("./simulation-results/bayes_vs_kappa.csv", header=None)

kappa_val_name = ["Kappa=1/10","Kappa=1/9","Kappa=1/8","Kappa=1/7","Kappa=1/6","Kappa=1/5","Kappa=1/4","Kappa=1/3"]

plt.style.use('ggplot')

column_names = ["Kappa","Alpha","Sigma","MSE"]

graph_names = ["mle_vs_bayes_alpha.png",
               "mle_vs_bayes_sigma.png",
               "mle_vs_bayes_mse.png"]

for i in range(1, 4):
    x = mle_data.iloc[:, i].values
    y = bayes_data.iloc[:, i].values

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='royalblue', marker='x')

    for j, txt in enumerate(kappa_val_name):
        plt.annotate(f"{txt}", (x[j], y[j]), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8)

    plt.xlabel(column_names[i] + " for Maximum Likelihood Estimator")
    plt.ylabel(column_names[i] + " for Posterior Mean")
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.minorticks_on()

    plt.savefig(graph_names[i-1], dpi=1200)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

os.chdir('..')

current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

variable_map = {"kappa": 0, "gamma": 1}
data_map = {"alpha": 2, "sigma": 3, "mse": 4, "debiased": 5}

variable = input("What varies? [kappa/gamma] ")
graph_type = input("What type of graph? [alpha/sigma/mse/debiased] ")

assert variable in ["kappa","gamma"], "Not valid type of variable!"
assert graph_type in ["alpha","sigma","mse","debiased"], "Nos valid type of graph!"

file_path = "./simulation-results/comparison_mle_bayes_"+variable+".csv"
data = np.loadtxt(file_path, delimiter=',',skiprows=1)

mpl.rcParams.update({
    "text.usetex": False,                 # keep it lightweight
    "mathtext.fontset": "cm",             # Computer Modern for math
    "font.family": "serif",               # serif text
    "font.size": 16,                      # base font size
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.linewidth": 1,                  # thicker axes
    "figure.dpi": 150,
    "savefig.dpi": 600,                   # high-res exports
    "figure.constrained_layout.use": True # tidy spacing
})

if data_map[graph_type] < 5:
    variable_value = data[:, variable_map[variable]]
    data_mle = data[:,data_map[graph_type]]
    data_bayes = data[:,data_map[graph_type]+3]
else:
    variable_value = data[:-1, variable_map[variable]]
    data_mle = (data[:-1,3]/data[:-1,2])**2
    data_bayes = (data[:-1, 6] / data[:-1, 5]) ** 2

if variable == "kappa":
    symbol2 = r"$\kappa$"
else:
    symbol2 = r"$\gamma$"

if graph_type == "alpha":
    symbol1 = r"$\alpha$"
elif graph_type == "sigma":
    symbol1 = r"$\sigma$"
elif graph_type == "debiased":
    symbol1 = r"$\left(\frac{\sigma}{\alpha}\right)^2$"
else:
    symbol1 = "MSE"

fig, ax = plt.subplots(figsize=(5, 5))

ax.set_xlabel(fr"Bayes {symbol1}", labelpad=6)
ax.set_ylabel(fr"MLE {symbol1}",  labelpad=6)

ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.4)
ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.3)
ax.minorticks_on()

ax.tick_params(which="both", direction="in")
ax.tick_params(which="major", length=6, width=1)
ax.tick_params(which="minor", length=3, width=0.8)

plt.xlim(np.min(data_bayes)-0.05, np.max(data_bayes)+0.28)

ax.scatter(data_bayes,data_mle,s=10, alpha=0.7)

for i in range(len(variable_value)):
    plt.text(data_bayes[i] + 0.003, data_mle[i], symbol2+" = "+str(variable_value[i]))

fname = f"comparison_mle_bayes_{graph_type}_{variable}.png"
plt.savefig(fname, bbox_inches="tight", facecolor="white")
plt.show()
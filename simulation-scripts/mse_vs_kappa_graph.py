import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

file_path = "./simulation-results/gauss_rademacher_mse_data.csv"
data = np.loadtxt(file_path, delimiter=',',skiprows=1)

kappa_values = np.linspace(0.1,1.0,10)

mse_mcmc = data[:, 0]
std_mcmc = data[:, 1]
mse_iter = data[:, 2]
std_iter = data[:, 3]

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

fig, ax = plt.subplots(figsize=(6,4))

ax.errorbar(
    kappa_values, mse_iter, yerr=std_iter, fmt='o', color='tab:blue',
    ecolor='lightblue', elinewidth=1.5, capsize=4, label='Fixed point equations'
)

ax.errorbar(
    kappa_values, mse_mcmc, yerr=std_mcmc, fmt='s', color='tab:orange',
    ecolor='peachpuff', elinewidth=1.5, capsize=4, label='MCMC'
)

ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.4)
ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.3)
ax.minorticks_on()

ax.tick_params(which="both", direction="in")
ax.tick_params(which="major", length=6, width=1)
ax.tick_params(which="minor", length=3, width=0.8)

ax.legend()

plt.ylim(0.6,1.0)

plt.xlabel(r"$\kappa$")
plt.ylabel("Mean Square Error")

plt.savefig("mse_vs_kappa_gauss_rademacher.png", dpi=1200)



fig, ax = plt.subplots(figsize=(6,4))

ax.errorbar(
    kappa_values, mse_iter, yerr=std_iter, fmt='o', color='tab:blue',
    ecolor='lightblue', elinewidth=1.5, capsize=4, label='sin(x)'
)

plt.ylim(0.8,0.86)

ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.4)
ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.3)
ax.minorticks_on()

ax.tick_params(which="both", direction="in")
ax.tick_params(which="major", length=6, width=1)
ax.tick_params(which="minor", length=3, width=0.8)

plt.xlabel(r"$\kappa$")
plt.ylabel("Mean Square Error")

plt.savefig("mse_vs_kappa_gauss_rademacher2.png", dpi=1200)

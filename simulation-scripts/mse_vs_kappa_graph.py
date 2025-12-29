import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import make_smoothing_spline
import os

os.chdir("..")

file_path = "./simulation-results/gauss_rademacher_mse_data.csv"
data = np.loadtxt(file_path, delimiter=',', skiprows=1)

kappa_values = np.linspace(0.1, 1.0, 10)

mse_mcmc = data[:, 0]
std_mcmc = data[:, 1]   # kept but not plotted
mse_iter = data[:, 2]
std_iter = data[:, 3]   # kept but not plotted

mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.linewidth": 1,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "figure.constrained_layout.use": True
})

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(
    kappa_values, mse_iter, 'o',
    color='tab:blue', label='Fixed point equations'
)

ax.plot(
    kappa_values, mse_mcmc, 's',
    color='tab:orange', label='MCMC'
)

poly_coeffs = np.polyfit(kappa_values, mse_iter, deg=2)
poly = np.poly1d(poly_coeffs)

kappa_fine = np.linspace(kappa_values.min(), kappa_values.max(), 300)
mse_poly = poly(kappa_fine)

ax.plot(
    kappa_fine, mse_poly,
    '--', color='red', linewidth=2
)

ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.4)
ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.3)
ax.minorticks_on()

ax.tick_params(which="both", direction="in")
ax.tick_params(which="major", length=6, width=1)
ax.tick_params(which="minor", length=3, width=0.8)

ax.legend()

plt.ylim(0.6, 1.0)
plt.xlabel(r"$\kappa$")
plt.ylabel("Mean Square Error")

plt.savefig("mse_vs_kappa_gauss_rademacher.png", dpi=1200)


fig, ax = plt.subplots(figsize=(6, 4))


ax.plot(
    kappa_values, mse_iter, 'o',
    color='tab:blue'
)

ax.plot(
    kappa_fine, mse_poly,
    '--', color='red', linewidth=2
)

plt.ylim(0.8, 0.86)

ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.4)
ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.3)
ax.minorticks_on()

ax.tick_params(which="both", direction="in")
ax.tick_params(which="major", length=6, width=1)
ax.tick_params(which="minor", length=3, width=0.8)

plt.xlabel(r"$\kappa$")
plt.ylabel("Mean Square Error")

plt.savefig("mse_vs_kappa_gauss_rademacher2.png", dpi=1200)

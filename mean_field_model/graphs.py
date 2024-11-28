import numpy as np
from matplotlib import pyplot as plt
from mean_field_model.block_computation import BlockComputation

class MseGraphCreator(BlockComputation):
    """
    ComputeMseGraph computes graphs derived from the MeanFieldGLM class.

    This class allows for computing and analyzing various statistics and visualizations based on different
    configurations of MeanFieldGLM models.

    Parameters
    ----------
    var_list : list, optional
        List of values for the variable of interest (e.g., kappa, snr). Default is [1.0].
    variable : str, optional
        Name of the variable being varied (e.g., "kappa", "snr"). Default is "kappa".
    num_per_var : int, optional
        Number of samples per variable value. Default is 5.
    delta : float, optional
        Smooth approximation parameter for logistic regression. Default is 1.0.
    fixed_var : float, optional
        Fixed value for the variable that is not varied. Default is 1.0.
    prior : str, optional
        Prior distribution for the MeanFieldGLM model. Must be "Normal" or "Beta". Default is "Normal".
    signal : str, optional
        Signal distribution for the MeanFieldGLM model. Must be "Normal", "Rademacher", or "Beta". Default is "Normal".
    save : bool, optional
        If True, saves the computed statistics or graphs. Default is True.
    bayes_optimal : bool, optional
        If True, assumes the MeanFieldGLM model is Bayes optimal. Default is False.

    Attributes
    ----------
    stats : object
        Stores computed statistics or graph data.
    data : NoneType
        Placeholder for any required data storage.

    Notes
    -----
    - The `ComputeMseGraph` class is designed to work in conjunction with the `MeanFieldGLM`.
    - The `compute_stats()` and `plot_graph_MSE()` methods are implemented separately to perform specific
      computations and generate visual outputs based on the specified parameters.

    """
    def __init__(self, var_list= (0.1,1.0), init_params = (1.0,0.0,0.0,1e-6,0.0,0.0), variable="kappa", num_per_var=5,
                 delta=1.0, fixed_var=1.0, prior="Normal", signal="Normal", tolerance = 0.01, max_it = 7,
                 log_likelihood = "Logistic", save=True, bayes_optimal=False):
        super().__init__(var_list=var_list, init_params=init_params, variable=variable,
                         num_per_var=num_per_var,delta=delta, fixed_var=fixed_var, prior=prior,
                         signal=signal, tolerance=tolerance, max_it=max_it, log_likelihood=log_likelihood,
                         save=save, bayes_optimal=bayes_optimal)

        self.stats = None

    def compute_stats(self):
        """
        Computes statistics from the computed data.

        If self.stats is already computed, it does nothing (to avoid recomputation).

        Computes:
        - Mean and standard deviation of critical quantities (columns 2 and 3) for each kappa value.
        """
        if self.stats is None:
            n_kappa = len(self.var_list)
            self.stats = np.zeros((n_kappa, 6))

            for i in range(n_kappa):
                block_data = self.data[i * self.num_per_var:(i + 1) * self.num_per_var, :]
                self.stats[i, 0] = block_data[0, 0]  # kappa value
                self.stats[i, 1] = np.mean(block_data[:, 2])  # mean of column 2 (cB)
                self.stats[i, 2] = np.std(block_data[:, 2])  # std deviation of column 2
                self.stats[i, 3] = np.mean(block_data[:, 3])  # mean of column 3 (cBBs)
                self.stats[i, 4] = np.std(block_data[:, 3])  # std deviation of column 3
        else:
            pass  # stats already computed, do nothing

    def plot_graph_mse(self, save=False,limits=None):
        """
        Plots the Mean Squared Error (MSE) graph as a function of kappa/snr.

        Parameters:
        - save (bool): If True, saves the plot and data to files (default: False).
        """
        self.compute_data()  # Ensure data is computed
        self.compute_stats()  # Compute statistics from computed data

        x = self.stats[:, 0]  # kappa/snr values from computed stats

        # Determine y values and errors based on prior type
        if self.prior == "Beta":
            y = 0.3 * np.ones_like(self.stats[:, 1])  # MSE initialization for Beta prior
        elif self.prior == "Normal":
            y = 1.0 * np.ones_like(self.stats[:, 1]) # MSE initialization for Normal prior

        if self.bayes_optimal:
            y -= self.stats[:, 3]  # MSE calculation for Bayes optimal model
            y_error = self.stats[:, 4] / self.num_per_var ** (1 / 4)  # Error bars for Bayes optimal model
        else:
            y += self.stats[:, 1] - 2 * self.stats[:, 3]  # MSE calculation for non Bayes optimal model
            y_error = (self.stats[:,2] + 2 * self.stats[:, 4]) / self.num_per_var ** (1 / 4)  # Error bars for non Bayes optimal model

        x_error = np.zeros_like(x)  # No x-error for this plot

        # Plotting setup
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(8, 6))
        plt.errorbar(x, y, xerr=x_error, yerr=y_error, fmt='o-', color='royalblue', capsize=7)
        if self.prior == "Beta" and limits is None:
            plt.ylim(0.0, 0.055)  # Limit y-axis to 0.0 to 0.055
        elif self.prior == "Normal" and limits is None:
            plt.ylim(0.55, 0.9)  # Limit y-axis to 0.0 to 1.0
        else:
            plt.ylim(limits[0],limits[1])
        plt.xlabel(self.variable)
        plt.ylabel('MSE')
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.minorticks_on()

        # Save plot and data if save=True
        if save:
            plt.savefig("MSE_plot.png", dpi=600)  # Save plot as PNG file

            # Save graph data as CSV file
            graph_data = np.array([x, y, y_error]).transpose()
            np.savetxt("MSE_graph_data.csv", graph_data, delimiter=",")

        plt.legend()  # Show legend
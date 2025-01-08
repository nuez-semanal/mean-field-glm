import numpy as np
from matplotlib import pyplot as plt
from bayesian_glm.model import ModelGLM
from mean_field_glm.marginal import MeanFieldMarginalGLM
from mean_field_glm.block_computation import BlockComputation

class MseGraphCreator(BlockComputation):
    """
    MseGraphCreator computes graphs derived from the MeanFieldGLM class.

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
    def __init__(self, var_tuple= (0.1,1.0), init_params = (1.0,0.0,0.0,1e-6,0.0,0.0), variable="kappa", num_per_var=5,
                 delta=1.0, fixed_var=1.0, prior="Normal", signal="Normal", tolerance = 0.01, max_it = 7,
                 log_likelihood = "Logistic", save=True, bayes_optimal=False,file_name="Computed_data"):
        super().__init__(var_tuple=var_tuple, init_params=init_params, variable=variable,
                         num_per_var=num_per_var,delta=delta, fixed_var=fixed_var, prior=prior,
                         signal=signal, tolerance=tolerance, max_it=max_it, log_likelihood=log_likelihood,
                         save=save, bayes_optimal=bayes_optimal, file_name=file_name)

        self.stats = None

    def compute_stats(self):
        """
        Computes statistics from the computed data.

        If self.stats is already computed, it does nothing (to avoid recomputation).

        Computes:
        - Mean and standard deviation of critical quantities (columns 2 and 3) for each kappa value.
        """
        if self.stats is None:
            n_kappa = len(self.var_tuple)
            self.stats = np.zeros((n_kappa, 6))

            for i in range(n_kappa):
                block_data = self.data[i * self.num_per_var:(i + 1) * self.num_per_var, :]
                self.stats[i, 0] = block_data[0, 1]  # kappa/snr value
                self.stats[i, 1] = np.mean(block_data[:, 4])  # mean of column 5 (c_b)
                self.stats[i, 2] = np.std(block_data[:, 4])  # std deviation of column 5
                self.stats[i, 3] = np.mean(block_data[:, 5])  # mean of column 6 (c_bbs)
                self.stats[i, 4] = np.std(block_data[:, 5])  # std deviation of column 6
        else:
            pass  # stats already computed, do nothing

    def plot_boxplot(self, save=False):
        pass

    def plot_graph_mse(self, save=False):
        """
        Plots the Mean Squared Error (MSE) graph as a function of kappa/snr.

        Parameters:
        - save (bool): If True, saves the plot and data to files (default: False).
        """
        if self.data is None:
            self.compute_data()  # Ensure data is computed

        self.compute_stats()  # Compute statistics from computed data

        if self.variable == "kappa":
            x = 1/self.stats[:, 0]  # If variable is kappa we plot as function of 1/kappa
        else:
            x = self.stats[:, 0]

        # Determine y values and errors based on prior type
        if self.prior == "Beta":
            y = 0.3 * np.ones_like(self.stats[:, 1])  # MSE initialization for Beta prior
        elif self.prior == "Normal":
            y = 1.0 * np.ones_like(self.stats[:, 1]) # MSE initialization for Normal prior

        if self.bayes_optimal:
            y -= self.stats[:, 1]  # MSE calculation for Bayes optimal model
            y_error = self.stats[:, 2] / self.num_per_var ** (1 / 4)  # Error bars for Bayes optimal model
        else:
            y += self.stats[:, 1] - 2 * self.stats[:, 3]  # MSE calculation for non Bayes optimal model
            y_error = (self.stats[:,2] + 2 * self.stats[:, 4]) / self.num_per_var ** (1 / 4)  # Error bars for non Bayes optimal model

        x_error = np.zeros_like(x)  # No x-error for this plot

        # Plotting setup
        plt.style.use('ggplot')
        plt.figure(figsize=(8, 6))
        plt.errorbar(x, y, xerr=x_error, yerr=y_error, fmt='o', color='royalblue', capsize=7)

        plt.ylim(np.min(y-np.abs(y_error)) - 0.025 ,np.max(y+np.abs(y_error)) + 0.025)

        if self.variable == "kappa":
            plt.xlabel("1/"+self.variable)
        else:
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
        plt.show()

class MarginalGraphCreator:
    def __init__(self, n = 500, p = 500, k = 10, snr = 1.0, prior = "Normal", signal = "Normal",
                 log_likelihood = "Logistic", parameters = (0.173219994666055,0.173219994666055,0.416196903358369)):
        self.snr = snr
        self.prior = prior
        self.parameters = parameters
        self.theoretical_model = self.theoretical_marginal = None
        self.empirical_marginal = self.true_beta = self.noise = None
        self.model = ModelGLM(n = n, p = p, snr = snr,
                              prior = prior, signal = signal,
                              log_likelihood = log_likelihood)
        self.model.draw_sample()
        self.change_coordinate(k)

    @staticmethod
    def compute_quantiles(data, q):
        normalization = (data < 1e23).sum()
        quantiles = np.zeros(q)
        values = np.linspace(np.min(data), np.max(data), q * 10)
        j = 0
        for i in range(q):
            proportion = 0
            while q * proportion < (i + 1) and j < q * 10 - 1:
                j += 1
                proportion = (data < values[j]).sum() / normalization
            quantiles[i] = values[j]
        return quantiles

    def sample_theoretical_model(self):
        self.theoretical_model = MeanFieldMarginalGLM(parameters=self.parameters,
                                        prior=self.prior,
                                        snr=self.snr,
                                        betas=self.true_beta,
                                        noise=self.noise)
        self.theoretical_model.sample()
        self.theoretical_marginal = self.theoretical_model.marginal_sample

    def change_coordinate(self,k):
        self.empirical_marginal = self.model.posterior[:,k]
        self.true_beta = self.model.true_beta[k]
        self.noise = self.model.compute_noise(k)
        self.sample_theoretical_model()

    def fix_noise_value(self,noise):
        self.noise = noise
        self.sample_theoretical_model()

    def histogram_marginal(self):
        if self.prior == "Beta":
            left_lim = 0.0
            right_lim = 1.0
        else:
            left_lim = -3.0
            right_lim = 3.0
        n_bins_theo = 40
        n_bins_emp = 20
        delta = (right_lim - left_lim) / n_bins_theo

        bins_theo = np.linspace(left_lim, right_lim, n_bins_theo)
        bins_emp = np.linspace(left_lim, right_lim, n_bins_emp)

        histogram_theo = np.histogram(self.theoretical_marginal, bins=bins_theo)
        frequency_theo = np.array(histogram_theo[0] / np.sum(histogram_theo[0])) / delta

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.empirical_marginal, bins=bins_emp, color='gray', density=True)
        ax.plot(histogram_theo[1][1:n_bins_theo] - delta / 2, frequency_theo)

        plt.grid()
        plt.xlabel("Beta")
        plt.ylabel("Density")

        plt.axvline(x=self.true_beta, color='red', linestyle='dashed', linewidth=2)
        plt.savefig("Marginal_comparison.png", dpi=600)
        plt.show()

    def qq_plot_marginal(self):
        quantiles_emp = self.compute_quantiles(self.empirical_marginal, 100)
        quantiles_theo = self.compute_quantiles(self.theoretical_marginal, 100)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(quantiles_emp[:99], quantiles_theo[:99])
        ax.plot([np.min(quantiles_emp), np.max(quantiles_emp)], [np.min(quantiles_emp), np.max(quantiles_emp)],
                color='red', linestyle='--')

        plt.grid()
        plt.xlabel("Empirical quantiles")
        plt.ylabel("Theoretical quantiles")
        plt.savefig("QQ plot.png", dpi=600)
        plt.show()
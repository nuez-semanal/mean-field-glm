import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

class MeanFieldMarginalGLM:
    """
    Class for generating samples from the asymptotic marginals of Generalized Linear Models (GLMs)
    in the high-dimensional limit and visualizing their densities.

    Attributes:
    - parameters (list): List of parameters [tr, ts, tt].
    - prior (str): Prior distribution for beta parameter ('Beta' or 'Normal').
    - betas (float): Scaling factor for the signal.
    - snr (float): Signal-to-noise ratio.
    - p (int): Dimensionality of beta parameter (fixed at 1000).
    - marginal_sample (np.ndarray): Samples from the posterior distribution of beta.
    - seed (int): Random seed for reproducibility.

    Methods:
    - sample(draws=1000, tune=2000): Draws samples from marginal distribution.
    - plot_graph_marginals(): Plots the density of the sampled marginal data (beta parameter).
    """
    def __init__(self, parameters: list, prior: str = "Normal", betas: float = 0.75, snr: float = 1.0, noise : float = None):
        """
        Initializes a MarginalGLM object.

        Parameters:
        - parameters (list): List of parameters [tr, ts, tt].
        - prior (str): Prior distribution for beta parameter ('Beta' or 'Normal'). Default is 'Normal'.
        - betas (float): Signal to be used to compare with samples. Default is 0.75.
        - snr (float): Signal-to-noise ratio. Default is 1.0.
        """
        self.parameters = parameters  # Parameters [r1, r2, r3]
        self.prior = prior  # Prior distribution for beta parameter
        self.betas = betas  # Signal to be used to compare with samples
        self.snr = snr  # Signal-to-noise ratio
        self.p = 1000  # Dimensionality of beta parameter (fixed at 1000)

        self.marginal_sample = None  # Placeholder for sampled marginal data
        self.seed = np.random.randint(1000)  # Random seed initialization
        np.random.seed(self.seed)  # Seed the random number generator

        # PyMC3 model initialization
        with pm.Model() as self.marginal_measure:
            tr, ts, tt = parameters[0], parameters[1], parameters[2]  # Extract parameters

            # Define prior distribution for beta
            if self.prior == "Beta":
                beta = pm.Beta("beta", alpha=2.0, beta=2.0, shape=self.p)
            elif self.prior == "Normal":
                beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=self.p)
            else:
                print("Prior argument should take either value 'Beta' or 'Normal'")
                raise ValueError()
            if noise is None:
                z = np.random.normal(0.0, 1.0)  # Sample from standard normal distribution
            else:
                z = noise

            # Generate observations based on the chosen prior and parameters
            observation = np.sqrt(snr) * ts / np.sqrt(tr) * betas + tt / np.sqrt(tr) * z
            observation = np.array([observation for i in range(self.p)])  # Repeat observation for each dimension

            # Likelihood of observations given beta
            likelihood = pm.Normal("likelihood", mu = np.sqrt(snr * tr) * beta, sigma = 1.0, observed = observation)

    def sample(self):
        """
        Draws samples from the marginal distribution of beta parameter.

        This method runs a PyMC sampler to draw samples from the posterior
        distribution of beta parameter based on the initialized model.

        After sampling, stores the sampled values in self.marginal_sample.

        Returns:
        - None
        """
        with self.marginal_measure:
            sample = pm.sample(1000, progressbar=False, tune=1000)

        self.marginal_sample = np.array(sample["posterior"]["beta"][0])

    def check_if_sample(self):
        """
        Checks if samples have been computed.

        Raises a ValueError if self.marginal_sample is None, indicating that
        samples have not yet been computed or stored.

        Returns:
        - None
        """
        if self.marginal_sample is None:
            print("Please, compute sample first!")
            raise ValueError()

    def plot_graph_marginals(self):
        """
        Plots the density histogram of the sampled marginal distribution of beta.

        This method generates a histogram of the samples stored in self.marginal_sample,
        normalizes it, and then plots the density histogram. It also includes a vertical
        dashed line at the value of self.betas and labels it with LaTeX symbol representation.

        Raises:
        - ValueError: If self.marginal_sample is None, indicating samples have not been computed.

        Returns:
        - None
        """
        self.check_if_sample()  # Ensure that samples have been computed

        # Define bins based on the prior distribution
        if self.prior == "Beta":
            bins = np.linspace(0, 1, 50)
        elif self.prior == "Normal":
            bins = np.linspace(-3, 3, 80)

        # Compute histogram and normalize
        self.histogram = np.histogram(self.marginal_sample, bins=bins)
        self.histogram[0] = self.histogram[0] / np.sum(self.histogram[0])

        # Plot histogram
        plt.plot(self.histogram[1], self.histogram[0])

        # Add vertical dashed line at self.betas
        plt.axvline(x=self.betas, color='red', linestyle='dashed', linewidth=2)

        # Label the dashed line with LaTeX symbol for beta_star_j0
        latex_symbol = r"$\beta_{\star,j_0}$"
        plt.text(self.betas + 0.1, 0.5, latex_symbol, ha='center', va='bottom', fontsize=12)

        # Show the plot
        plt.show()
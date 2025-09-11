import numpy as np
import pymc as pm
from mean_field_glm.auxiliary import AuxiliaryFunctions

class MeanFieldBetaGLM(AuxiliaryFunctions):
    """
    MeanFieldGLM class for Mean-Field inference in Bayesian Generalized Linear Models (GLMs).

    This class implements the Mean-Field inference for approximating Bayesian GLMs in the large-dimensional limit.

    Parameters
    ----------
    p : int
        Number of parameters in the GLM model. (Required)
    n : int
        Number of observations available. (Required)
    kappa : float, optional
        Ratio of p/n. If not provided, it is set to p/n. (Default: None)
    draws : int, optional
        Number of draws used in the MCMC simulations. (Default: 1000)
    tune : int, optional
        Tuning parameter for the MCMC simulations. (Default: 2000)
    tolerance : float, optional
        Tolerance parameter for the FPE stopping criteria. (Default: 0.02)
    max_it : int, optional
        Maximum number of FPE iterations allowed. (Default: 7)
    v_b : float, optional
        Initial value for the critical quantity 'v_b'. (Default: 1.0)
    c_b : float, optional
        Initial value for the critical quantity 'c_b'. (Default: 0.0)
    c_bbs : float, optional
        Initial value for the critical quantity 'c_bbs'. (Default: 0.0)
    r_1 : float, optional
        Initial value for the critical quantity 'r_1'. (Default: 1e-6)
    r_2 : float, optional
        Initial value for the critical quantity 'r_2'. (Default: 0.0)
    r_3 : float, optional
        Initial value for the critical quantity 'r_3'. (Default: 0.0)
    log_likelihood : str, optional
        Type of GLM to consider. Must be "Logistic" or "Linear". (Default: "Logistic")
    signal : str, optional
        Distribution of the parameters to be inferred. Must be "Rademacher", "Normal", or "Beta". (Default: "Normal")
    prior : str, optional
        Prior distribution for the posterior. Must be "Normal" or "Beta". (Default: "Normal")
    delta : float, optional
        Smooth approximation parameter, only used for logistic regression. (Default: 0.01)
    seed : int, optional
        Seed for the random number generator used by NumPy. (Default: None)
    bayes_optimal : bool, optional
        If True, the model is assumed to be Bayes optimal, reducing the complexity of the FPEs. (Default: False)

    Examples
    --------
    Returns the values of the order parameters corresponding to the Bayes optimal Linear Regression model of Normal signal and prior with a signal-to-noise ratio of 5.0 and kappa of 1.0.
    """
    def __init__(self, p=1000, n=1000, kappa=None, cores = 4, chains = 4, draws=1000, tune=2000, tolerance=0.02,
                max_it=7, v_b=1.0, c_b=0.0, c_bbs=0.0, r_1=1e-6, r_2=0.0, r_3=0.0,
                delta=0.01, seed=None, bayes_optimal=False):
        """
        Initialize the MeanFieldGLM class with the specified parameters.
        """

        # Set kappa to p/n if not provided
        if kappa is None:
            kappa = p / n

        # Generate a random seed if not provided
        if seed is None:
            seed = np.random.randint(1000)

        # Store the provided parameters as class attributes
        self.p, self.n, self.kappa = p, n, kappa
        self.draws, self.tune, self.chains,  = draws, tune, chains
        self.tolerance, self.max_it, self.cores = tolerance, max_it, cores
        self.delta, self.bayes_optimal = delta, bayes_optimal
        self.v_b, self.c_b, self.c_bbs  = v_b, c_b, c_bbs
        self.r_1, self.r_2, self.r_3 = r_1, r_2, r_3
        self.seed = seed

        # Seed the NumPy random number generator
        np.random.seed(seed)

        # Initialize true_beta based on the signal type
        if bayes_optimal:
            self.true_beta = np.random.beta(2, 2, size=p) - 0.5
        else:
            self.true_beta = np.random.beta(2, 5, size=p)

        # Calculate the gamma parameter as the square root of the mean squared true_beta
        self.gamma = np.sqrt(np.mean(self.true_beta ** 2))

        # Initialize placeholders for MCMC samples and posteriors
        self.sample1 = self.posterior1 = self.sample2 = self.posterior2 = self.sample3 = self.posterior3 = None

        # Compute some initial parameter
        self.t_gamma = self.compute_t_gamma()

        # Initialize a counter for rejected samples
        self.rejected_samples = 1

        # Creates the mean-field models for the analyzed GLM
        self.update_mean_field_models()

    def is_critical(self, xi, z, e):
        """
        Check if the values (xi, z, e) are critical for the logistic regression.

        Parameters
        ----------
        xi : float
            A random normal variable.
        z : float
            A random normal variable.
        e : float
            A logit-transformed random variable.

        Returns
        -------
        bool
            True if the values are critical, False otherwise.
        """
        k = self.kappa
        if self.c_b > 0:
            # Compute theta using c_b, c_bbs, gamma, and kappa
            theta = np.sqrt(k * (self.gamma**2 - self.c_bbs**2 / self.c_b)) * xi + self.c_bbs * np.sqrt(k / self.c_b) * z
        else:
            # Compute theta using gamma and kappa when c_b is non-positive
            theta = np.sqrt(k) * self.gamma * xi
        # Check if theta is close enough to e considering the delta tolerance
        return abs(theta - e) < 5 * self.delta

    def sample_one_critical(self):
        """
        Sample one set of critical values (xi, z, e) that are close to 0.

        Returns
        -------
        tuple
            A tuple containing sampled values xi, z, e, and the number of samples generated to find the critical value.
        """
        n_samples = 0
        while True:
            n_samples += 1
            # Generate a logit-transformed random variable
            e = self.logit(np.random.random())
            # Generate two normal random variables
            xi, z = np.random.normal(0.0, 1.0, size=2)
            # Check if the sampled values are critical
            if self.is_critical(xi, z, e):
                return xi, z, e, n_samples

    def sample_criticals(self, n):
        """
        Sample n sets of critical values.

        Parameters
        ----------
        n : int
            Number of critical samples to generate.

        Returns
        -------
        tuple
            A tuple containing arrays of xi, z, e, and the total number of samples generated.
        """
        total_samples = 0
        # Initialize arrays to store sampled values
        xi = np.zeros(n)
        z = np.zeros(n)
        e = np.zeros(n)
        for i in range(n):
            # Sample one critical set of values
            xi[i], z[i], e[i], n_samples = self.sample_one_critical()
            # Accumulate the total number of samples generated
            total_samples += n_samples
        return xi, z, e, total_samples

    def compute_t_gamma(self):
        """
        Computes the parameter t_gamma of the model based on the logistic or linear regression type.

        Returns
        -------
        float
            The computed t_gamma parameter.
        """
        # Compute constants for logistic regression
        c = np.sqrt(self.kappa) * self.gamma
        d = self.delta

        # Define range for theta
        range_xth = np.linspace(-5, 5, 1000)

        # Create a grid of theta and e values
        X = np.array([[[x_th, x_e] for x_e in np.linspace(c * x_th - 5 * d, c * x_th + 5 * d, 1000)] for x_th in range_xth])

        slice_integrals = []

        for i in range(1000):
            th = X[i, :][0, 0]
            e = X[i, :][:, 1]

            # Compute the integrand for the given theta and e
            y = self.gauss_density(th) * np.multiply(self.dif_tilde_t_delta(c * th - e,self.delta), self.dif_sigmoid(e))

            # Integrate over e
            slice_integrals.append(np.trapz(y, e))

        # Integrate over theta to get t_gamma
        return np.trapz(slice_integrals, range_xth)

    def generate_thetas(self, xi_b, xi_bs, z_bbs):
        """
        Generate theta and theta_s values for the mean-field models 2 and 3.

        Parameters
        ----------
        xi_b : array-like
            Random normal variables for theta computation.
        xi_bs : array-like
            Random normal variables for theta_s computation.
        z_bbs : array-like
            Random normal variables for theta and theta_s computation.

        Returns
        -------
        tuple
            A tuple containing theta and theta_s values.
        """
        if self.c_b > 0.0001:
            # Compute theta_s and theta when c_b is positive
            theta_s = np.sqrt(self.kappa * (self.gamma**2 - self.c_bbs**2 / self.c_b)) * xi_bs + self.c_bbs * np.sqrt(self.kappa / self.c_b) * z_bbs
            theta = pm.Deterministic("theta", np.sqrt(self.kappa * (self.v_b - self.c_b)) * xi_b + np.sqrt(self.kappa * self.c_b) * z_bbs)
        else:
            # Compute theta_s and theta when c_b is non-positive
            theta_s = np.sqrt(self.kappa) * self.gamma * xi_bs
            theta = pm.Deterministic("theta", np.sqrt(self.kappa * self.v_b) * xi_b)
        return theta, theta_s

    def update_mean_field_models(self):
        """
        Update the mean-field models based on the current configuration.

        This method updates the two mean-field models. If the logistic regression model is used,
        it updates model 3 as well which models the second mean-field model with critical observations.
        """
        self.update_mean_field_model1()
        self.update_mean_field_model2()
        self.update_mean_field_model3()

    def update_mean_field_model1(self):
        """
        Update the first mean-field model.

        This model involves creating a PyMC model where the prior distribution of beta can be either
        Beta or Normal, depending on the prior configuration.
        """
        with pm.Model() as self.mean_field_model1:
            # Define prior for beta based on the specified prior type
            beta = pm.Beta("beta", alpha=2.0, beta=2.0, shape=self.p)

            # Generate random noise
            z = np.random.normal(0.0, 1.0, size=self.p)

            # Generate observations based on the true beta values and noise
            self.observations1 = self.r_2 / np.sqrt(self.r_1) * self.true_beta + self.r_3 / np.sqrt(self.r_1) * z

            # Define the likelihood
            likelihood = pm.Normal("likelihood", mu = np.sqrt(self.r_1) * beta, sigma=1.0, observed=self.observations1)

    def update_mean_field_model2(self):
        """
        Update the second mean-field model.

        This model involves generating theta and theta_s values and updating the PyMC model
        based on the type of regression (Logistic or Linear).
        """
        with pm.Model() as self.mean_field_model2:
            # Generate random normal variables for theta computation
            z_bbs = xi_bs = np.random.normal(0, 1, size=self.n)
            xi_b = pm.Normal("xi_b", shape=self.n)

            # Generate theta and theta_s values
            theta, theta_s = self.generate_thetas(xi_b, xi_bs, z_bbs)

            # Generate observations based on the type of regression
            e = self.logit(np.random.random(self.n))

            self.observations2 = theta_s - e

            # Define the likelihood based on the type of regression
            likelihood = pm.Bernoulli("likelihood", p=self.sigmoid(theta), observed=self.indicator(self.observations2))

    def update_mean_field_model3(self):
        """
        Update the third mean-field model (only for Logistic regression).

        This model involves sampling critical values and updating the PyMC model based on these values.
        """
        with pm.Model() as self.mean_field_model3:
            # Sample critical values
            xi_bs, z_bbs, e, self.rejected_samples = self.sample_criticals(self.n)
            xi_b = pm.Normal("xi_b", shape=self.n)

            # Generate theta and theta_s values
            theta, theta_s = self.generate_thetas(xi_b, xi_bs, z_bbs)

            self.observations3 = theta_s - e

            # Define the likelihood for logistic regression
            likelihood = pm.Bernoulli("likelihood", p=self.sigmoid(theta), observed=self.indicator(self.observations3))

    def draw_samples(self):
        """
        Draw samples from the posterior distributions of the mean-field models.

        This method samples from the posterior distributions of the mean-field models 1, 2, and optionally 3
        (if the logistic regression model is used). The samples are stored in the corresponding attributes.

        Updates
        -------
        self.sample1 : pm.MultiTrace
            Samples from the first mean-field model.
        self.sample2 : pm.MultiTrace
            Samples from the second mean-field model.
        self.sample3 : pm.MultiTrace
            Samples from the third mean-field model (only for logistic regression).
        self.posterior1 : np.ndarray
            Posterior samples of beta from the first mean-field model.
        self.posterior2 : np.ndarray
            Posterior samples of theta from the second mean-field model.
        self.posterior3 : np.ndarray
            Posterior samples of theta from the third mean-field model (only for logistic regression).
        """

        # Draw samples from the first mean-field model
        with self.mean_field_model1:
            self.sample1 = pm.sample(cores = self.cores, draws = self.draws, chains=self.chains, progressbar=False, tune=self.tune)
        # Extract posterior samples for beta from the first mean-field model
        self.posterior1 = np.array(self.sample1["posterior"]["beta"][0])

        # Draw samples from the second mean-field model
        with self.mean_field_model2:
            self.sample2 = pm.sample(cores = self.cores, draws = self.draws, chains=self.chains, progressbar=False, tune=self.tune)
        # Extract posterior samples for theta from the second mean-field model
        self.posterior2 = np.array(self.sample2["posterior"]["theta"][0])

        # If logistic regression, draw samples from the third mean-field model
        with self.mean_field_model3:
            self.sample3 = pm.sample(cores = self.cores, draws = self.draws, chains=self.chains, progressbar=False, tune=self.tune)
        # Extract posterior samples for theta from the third mean-field model
        self.posterior3 = np.array(self.sample3["posterior"]["theta"][0])

    def update_order_parameters1(self):
        """
        Update order parameters v_b, c_b, and c_bbs based on posterior samples from the first mean-field model.

        This method calculates and updates v_b, c_b, and c_bbs by computing specific dot products and taking their means.
        Optionally adjusts v_b and c_b based on the bayes_optimal flag and prior distribution type.

        Updates
        -------
        self.v_b : float
            Updated value of v_b.
        self.c_b : float
            Updated value of c_b.
        self.c_bbs : float
            Updated value of c_bbs.
        """

        order_parameters = np.zeros((self.draws, 3))

        # Compute dot products that define the order parameters
        for i in range(self.draws):
            t, s = np.random.randint(0, self.draws, 2)

            order_parameters[i, 0] = np.dot(self.posterior1[i], self.posterior1[i]) / self.p
            order_parameters[i, 1] = np.dot(self.posterior1[t], self.posterior1[s]) / self.p
            order_parameters[i, 2] = np.dot(self.posterior1[i], self.true_beta) / self.p

        # Calculate means of order parameters to update v_b, c_b, and c_bbs
        v_b = np.mean(order_parameters[:, 0])
        c_b = np.mean(order_parameters[:, 1])
        c_bbs = np.mean(order_parameters[:, 2])

        if self.bayes_optimal:
            v_b, c_b = 0.3, c_bbs

        # Update class attributes with computed values
        self.v_b = v_b
        self.c_b = c_b
        self.c_bbs = c_bbs

    def compute_s_theta(self,observations: np.ndarray, posterior: np.ndarray, l: int):
        return self.tilde_t_delta(observations, self.delta) - self.sigmoid(posterior[l])

    def compute_s_theta_star(self,observations: np.ndarray, posterior: np.ndarray, l: int):
        return np.multiply(self.dif_tilde_t_delta(observations, self.delta), posterior[l])

    def second_dif_A(self,x: float):
        return self.dif_sigmoid(x)

    def update_order_parameters2(self):
        """
        Update order parameters r_1, r_2, and r_3 based on posterior samples from mean-field models.

        This method calculates and updates r_1, r_2, and r_3 by computing dot products and taking their means
        from posterior samples of the mean-field models 2 and optionally 3 (if logistic regression).

        Updates
        -------
        self.r_1 : float
            Updated value of r_1.
        self.r_2 : float
            Updated value of r_2.
        self.r_3 : float
            Updated value of r_3.
        """

        order_parameters = np.zeros((self.draws, 5))

        # Compute the dot products that define the order parameters
        for i in range(self.draws):
            t, s = np.random.randint(0, self.draws, 2)
            s_theta_1 =  self.compute_s_theta(self.observations2, self.posterior2, t)
            s_theta_2 = self.compute_s_theta(self.observations2, self.posterior2, s)
            s_theta_critical_1 = self.compute_s_theta(self.observations3, self.posterior3, t)
            s_theta_critical_2 = self.compute_s_theta(self.observations3, self.posterior3, s)
            s_theta_star_critical_1 = self.compute_s_theta_star(self.observations3, self.posterior3, t)

            order_parameters[i, 0] = np.dot(s_theta_1, s_theta_1) / self.n
            order_parameters[i, 1] = np.dot(s_theta_1, s_theta_2) / self.n
            order_parameters[i, 2] = np.dot(s_theta_critical_1, s_theta_star_critical_1) / self.rejected_samples
            order_parameters[i, 3] = np.dot(s_theta_critical_2, s_theta_star_critical_1) / self.rejected_samples
            order_parameters[i, 4] = np.sum(self.second_dif_A(self.posterior2[i])) / self.n

        # Calculate r_1, r_2, and r_3 as the mean of parameters computed above
        r_1 = np.mean(order_parameters[:, 4] + order_parameters[:, 1] - order_parameters[:, 0])
        r_2 = np.mean(order_parameters[:, 2] - order_parameters[:, 3]) + self.t_gamma
        r_3 = np.sqrt(np.mean(order_parameters[:, 1]))

        # If the problem is Bayes optimal, we use Nishimori identities to simplify the output
        if self.bayes_optimal:
            r_2 = r_1
            r_3 = np.sqrt(r_1)

        # Update class attributes with computed values
        self.r_1 = r_1
        self.r_2 = r_2
        self.r_3 = r_3

    def run_one_iteration(self):
        """
        Run one iteration of the Mean-Field GLM algorithm.

        This method executes one iteration of the Mean-Field GLM algorithm, which involves the following steps:
        1. Draw posterior samples from mean-field models.
        2. Update order parameters v_b, c_b, c_bbs, r_1, r_2, r_3 based on the drawn samples.
        3. Update parameters and likelihoods of mean-field models based on the updated order parameters.
        """
        # Step 1: Draw samples from posterior distributions of mean-field models
        self.draw_samples()

        # Step 2: Update order parameters v_b, c_b, c_bbs based on posterior samples
        self.update_order_parameters1()
        self.update_order_parameters2()

        # Step 3: Update parameters and likelihoods of mean-field models
        self.update_mean_field_model1()
        self.update_mean_field_model2()

        # If logistic regression, update the third mean-field model also
        self.update_mean_field_model3()

    def check_stability(self):
        """
        Check stability of order parameters across iterations.

        This method checks the stability of order parameters (v_b, c_b, c_bbs, r_1, r_2, r_3) across iterations
        by computing the relative change in their values before and after running one iteration of the Mean-Field GLM algorithm.

        Returns
        -------
        float
            Relative change in order parameters between consecutive iterations.
        """
        # Store current order parameters
        old_order_parameters = np.array([self.v_b, self.c_b, self.c_bbs, self.r_1, self.r_2, self.r_3])

        # Perform one iteration of the Mean-Field GLM algorithm to update order parameters
        self.run_one_iteration()

        # Store updated order parameters after the iteration
        new_order_parameters = np.array([self.v_b, self.c_b, self.c_bbs, self.r_1, self.r_2, self.r_3])

        # Compute relative change in order parameters
        stability_measure = np.linalg.norm(old_order_parameters - new_order_parameters) / np.linalg.norm(old_order_parameters)

        return stability_measure

    def run_iterations(self):
        """
        Run multiple iterations of the Mean-Field GLM algorithm until convergence criteria are met.

        This method iterates through the Mean-Field GLM algorithm until either the maximum number of iterations
        (`max_it`) is reached or the stability distance (`tolerance`) between consecutive iterations falls below
        a specified threshold.

        Prints progress information after each iteration.

        Returns
        -------
        None
        """
        i, distance = 0, 1.0

        # Continue iterating until convergence criteria are met or maximum number of iterations reached
        while distance > self.tolerance and i < self.max_it:
            i += 1
            distance = self.check_stability()  # Check stability of order parameters

            # Print progress information
            print(f"\n[{i} iteration/s out of {self.max_it} max ready. Distance achieved = {distance}]\n")
            print("Actual values: ",self.v_b,self.c_b,self.c_bbs,self.r_1,self.r_2,self.r_3,"\n")

    def show_order_parameters(self, output=False):
        """
        Display or return the current values of order parameters.

        Parameters
        ----------
        show : bool, optional
            If True, print the current values of order parameters to the console. Default is True.
        output : bool, optional
            If True, return the current values of order parameters as a list. Default is False.

        Returns
        -------
        list or None
            If `output` is r_1ue, returns a list containing the values of order parameters [v_b, c_b, c_bbs, r_1, r_2, r_3].
            If `output` is False and `show` is True, prints the values of order parameters to the console.
            If both `output` and `show` are False, returns None.
        """
        print(self.v_b, self.c_b, self.c_bbs, self.r_1, self.r_2, self.r_3)

        if output:
            return [self.v_b, self.c_b, self.c_bbs, self.r_1, self.r_2, self.r_3]
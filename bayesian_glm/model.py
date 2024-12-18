import numpy as np
import pymc as pm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mean_field_glm.auxiliary import AuxiliaryFunctions

from bayesian_glm.noise import NoiseComputer

class ModelGLM(AuxiliaryFunctions):
    def __init__(self, p=1000, n=1000, cores = 4, chains = 4, draws=1000, tune=2000, log_likelihood="Logistic", 
                 signal="Normal", prior="Normal", snr=1.0, seed=None):

        self.p, self.n, self.cores, self.chains, self.draws = p, n, cores, chains, draws
        self.tune, self.log_likelihood, self.signal = tune, log_likelihood, signal
        self.prior, self.snr = prior, snr
                    
        # Here I generate the Gaussian data available
        if seed is None:
            self.seed = np.random.randint(1000)
        else:
            self.seed = seed

        np.random.seed(seed)

        self.data = np.random.normal(loc=0.0,scale=1/np.sqrt(n),size=[n, p])
        self.log_likelihood = log_likelihood

        if signal == "Rademacher":
            self.true_beta = np.sign(2*np.random.random(p)-1)
        elif signal == "Normal":
            self.true_beta = np.random.normal(0.0,1.0,size=p)
        elif signal == "Beta":
            self.true_beta = np.random.normal(0.0,1.0,size=p)
        else:
            print("Signal argument should take either value 'Rademacher' or 'Normal'")
            raise ValueError()

        true_projections = np.dot(self.data, self.true_beta)

        # Observations available
        if log_likelihood == "Logistic":
            self.noise = self.logit(np.random.random(size=n))
            self.observations = self.indicator(np.sqrt(snr)*true_projections-self.noise)
        elif log_likelihood == "Linear":
            self.noise = np.random.normal(size=n)
            self.observations = np.sqrt(snr)*true_projections-self.noise
        self.sample = None
        self.posterior = None

        # Here I define the logistic regression model in PyMC
        with pm.Model() as self.model:

            X = pm.MutableData("X", self.data)
            y = pm.MutableData("y", self.observations)

            if prior == "Beta":
                beta = pm.Beta("beta", alpha = 2.0, beta = 2.0, shape=p)
            elif prior == "Normal":
                beta = pm.Normal("beta",mu=0.0,sigma=1.0,shape=p)
            else:
                print("Prior argument should take either value 'Beta' or 'Normal'")
                raise ValueError()

            # Here I compute the fitted values
            fitted_values = pm.math.dot(X, beta)

            proba = pm.Deterministic('proba', pm.math.invlogit(np.sqrt(snr)*fitted_values))

            # Here I define the likelihood for a logistic regression model
            if self.log_likelihood == "Logistic":
                likelihood = pm.Bernoulli("likelihood",p=proba,observed=y)
            elif self.log_likelihood == "Linear":
                likelihood = pm.Normal("likelihood",mu = np.sqrt(snr)*fitted_values,observed=y)

    def draw_sample(self):
        with self.model:
            # I use MCMC to sample from the posterior distribution.
            # The default method in PyMC is a variant of Hamilotinian Monte Carlo called NUTS
            self.sample = pm.sample(cores=self.cores, tune=self.tune, draws=self.draws) # Tune parameter fixes the steps before starting sampling

        # I extract from the InferenceData object an array containing the samples from the posterior
        self.posterior = np.array(self.sample["posterior"]["beta"][0])

    def check_if_sample(self):
        if self.sample is None:
            print("This computation requires drawing a sample first. Use draw_sample() method.")
            raise TypeError()

    def compute_order_parameters(self):
        # Initialization of the first group of order parameters. This are the "spin" order parameters that have a simple geometrical interpretation:
        # vB is the mean norm of the samples, c_b is the overlap, and c_bbs is the magnetisations with signal.
        self.check_if_sample()
        
        v_b, c_b, c_bbs = 0,0,0

        for i in range(self.draws):
            v_b += np.sum(self.posterior[i]**2)/(self.draws*self.p)
            t, s = np.random.randint(0,self.draws,2) # I compute the overlap between random pairs of replicas to avoid correlations
            c_b += np.dot(self.posterior[t],self.posterior[s])/(self.draws*self.p)
            c_bbs += np.dot(self.posterior[i],self.true_beta)/(self.draws*self.p)

        return v_b, c_b, c_bbs

    def compute_hq(self):
        self.check_if_sample()

        fitted_values = self.data * self.posterior
        hq_values = np.zeros(self.draws)

        if self.log_likelihood == "Logistic":
            def compute_s_theta(observations: np.ndarray, fitted_val: np.ndarray, l: int):
                return self.indicator(observations) - self.sigmoid(fitted_val[l])
        elif self.log_likelihood == "Linear":
            def compute_s_theta(observations: np.ndarray, fitted_val: np.ndarray, l: int):
                return observations - fitted_val[l]

        for i in range(self.draws):
            t, s = np.random.randint(0, self.draws, 2)
            s_theta_1, s_theta_2 = compute_s_theta(self.observations, fitted_values, t), compute_s_theta(self.observations, fitted_values, s)
            hq_values[i] = np.dot(s_theta_1, s_theta_2) / self.n

        return np.mean(hq_values)


    def compute_noise(self, k):
        self.check_if_sample()
        hq = self.compute_hq()
        noise_computer = NoiseComputer(self,hq)
        return noise_computer.compute_noise(k)
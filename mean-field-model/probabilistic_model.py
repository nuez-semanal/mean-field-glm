import numpy as np
import pymc as pm
from auxiliary import AuxiliaryFunctions

class ProbabilisticModel(AuxiliaryFunctions):
    def __init__(self, fixed_point_equations, n_iterations, tune, chains):

        self.fixed_point_equations = fixed_point_equations
        self.n_iterations, self.tune, self.chains = n_iterations, tune, chains

        p = self.fixed_point_equations.mean_field_model.p
        signal = self.fixed_point_equations.mean_field_model.signal

        if signal == "Rademacher":
            self.true_beta = np.sign(2 * np.random.random(p) - 1)
        elif signal == "Normal":
            self.true_beta = np.random.normal(0.0, 1.0, size=p)
        else:
            print("Signal argument should take either value 'Rademacher' or 'Normal'")
            raise ValueError()

        self.sample1 = self.sample2 = self.posterior1 = self.posterior2 = None
        self.observations1 = self.observations2 = None

        self.update_mean_field_model1()
        self.update_mean_field_model2()

    def get_mean_field_parameters(self):
        n = self.fixed_point_equations.mean_field_model.n
        p = self.fixed_point_equations.mean_field_model.p
        snr = self.fixed_point_equations.mean_field_model.snr
        prior = self.fixed_point_equations.mean_field_model.prior
        param1 = self.fixed_point_equations.mean_field_model.param1
        param2 = self.fixed_point_equations.mean_field_model.param2
        return n, p, snr, prior, param1, param2

    def get_order_parameters(self):
        v_b = self.fixed_point_equations.v_b
        c_b = self.fixed_point_equations.c_b
        c_bbs = self.fixed_point_equations.c_bbs
        r_1 = self.fixed_point_equations.r_1
        r_2 = self.fixed_point_equations.r_2
        r_3 = self.fixed_point_equations.r_3
        return v_b, c_b, c_bbs, r_1, r_2, r_3

    def update_mean_field_model1(self):

        with pm.Model() as self.mean_field_model1:
            mf_parameters = self.get_mean_field_parameters()

            p, snr, prior, param1, param2 = mf_parameters[1:]
            r_1, r_2, r_3 = self.get_order_parameters()[3:]

            if prior == "Beta":
                beta = pm.Beta("beta", alpha=param1, beta=param2, shape=p)
            elif prior == "Normal":
                beta = pm.Normal("beta", mu=param1, sigma=param2, shape=p)
            else:
                print("Prior argument should take either value 'Beta' or 'Normal'")
                raise ValueError()

            z = np.random.normal(0.0, 1.0, size=p)

            self.observations1 = snr * r_2 / np.sqrt(r_1) * self.true_beta + r_3 / np.sqrt(r_1) * z

            likelihood = pm.Normal("likelihood", mu=snr * np.sqrt(r_1) * beta, sigma=1.0,
                                   observed=self.observations1)

    def update_mean_field_model2(self):

        with pm.Model() as self.mean_field_model2:

            n, p, snr = self.get_mean_field_parameters()[:3]
            v_b, c_b, c_bbs = self.get_order_parameters()[:3]

            kappa = p / n

            e = self.logit(np.random.random(n))

            z_bbs = np.random.normal(0, 1, size=n)
            xi_bs = np.random.normal(0, 1, size=n)
            xi_b = pm.Normal("xi_b", shape=n)

            if c_b > 0:
                theta_s = np.sqrt(kappa * (1 - c_bbs ** 2 / c_b)) * xi_bs + c_bbs * np.sqrt(kappa / c_b) * z_bbs
                theta = pm.Deterministic("theta", np.sqrt(kappa * (v_b - c_b)) * xi_b + np.sqrt(kappa * c_b) * z_bbs)
            else:
                theta_s = np.sqrt(kappa) * xi_bs
                theta = pm.Deterministic("theta", np.sqrt(kappa * v_b) * xi_b)

            self.observations2 = snr * theta_s - e

            likelihood = pm.Bernoulli("likelihood", p=self.sigmoid(snr * theta),
                                      observed=self.indicator(self.observations2))

    def draw_samples(self):
        with self.mean_field_model1:
            self.sample1 = pm.sample(self.chains, progressbar=False, tune=self.tune, cores=4)

        with self.mean_field_model2:
            self.sample2 = pm.sample(self.chains, progressbar=False, tune=self.tune, cores=4)

        self.posterior1 = np.array(self.sample1["posterior"]["beta"][0])
        self.posterior2 = np.array(self.sample2["posterior"]["theta"][0])
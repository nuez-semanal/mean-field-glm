import numpy as np
import pymc as pm
from auxiliary import AuxiliaryFunctions

class ProbabilisticModel(AuxiliaryFunctions):
    def __init__(self, fixed_point_equations, n_iterations, draws, tune, chains):

        self.fixed_point_equations = fixed_point_equations
        self.n_iterations, self.draws, self.tune, self.chains = n_iterations, draws, tune, chains

        p = self.fixed_point_equations.mean_field_model.p
        signal = self.fixed_point_equations.mean_field_model.signal

        if signal == "Rademacher":
            self.true_beta = np.sign(2 * np.random.random(p) - 1)
        elif signal == "Normal":
            self.true_beta = np.random.normal(0.0, 1.0, size=p)
        else:
            print("Signal argument should take either value 'Rademacher' or 'Normal'")
            raise ValueError()

        self.gamma = np.sqrt(np.mean(self.true_beta**2))

        self.sample1 = self.sample2 = self.posterior1 = self.posterior2 = None
        self.sample3 = self.posterior3 = self.observations3 = None
        self.observations1 = self.observations2 = None

        self.update_mean_field_models()

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

    def generate_thetas(self, xi_b, xi_bs, z_bbs):
        n, p = self.get_mean_field_parameters()[:2]
        v_b, c_b, c_bbs = self.get_order_parameters()[:3]

        kappa = p / n

        if c_b > 0.0001:
            # Compute theta_s and theta when cB is positive
            theta_star = np.sqrt(kappa * (self.gamma**2 - c_bbs / c_b)) * xi_bs + c_bbs * np.sqrt(kappa / c_b) * z_bbs
            theta = pm.Deterministic("theta", np.sqrt(kappa * (v_b - c_b)) * xi_b + np.sqrt(kappa * c_b) * z_bbs)
        else:
            # Compute theta_s and theta when cB is non-positive
            theta_star = np.sqrt(kappa) * self.gamma * xi_bs
            theta = pm.Deterministic("theta", np.sqrt(kappa * v_b) * xi_b)
        return theta, theta_star

    def update_mean_field_models(self):
        self.update_mean_field_model1()
        self.update_mean_field_model2()
        self.update_mean_field_model3()

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

            e = self.logit(np.random.random(n))

            z_bbs = np.random.normal(0, 1, size=n)
            xi_bs = np.random.normal(0, 1, size=n)
            xi_b = pm.Normal("xi_b", shape=n)

            theta, theta_star = self.generate_thetas(xi_b,xi_bs,z_bbs)

            self.observations2 = snr * theta_star - e

            likelihood = pm.Bernoulli("likelihood", p=self.sigmoid(snr * theta),
                                      observed=self.indicator(self.observations2))

    def update_mean_field_model3(self):

        with pm.Model() as self.mean_field_model3:
            n, p, snr = self.get_mean_field_parameters()[:3]

            critical_point_generator = CriticalPointsGenerator(self)

            xi_bs, z_bbs, e, self.rejected_samples = critical_point_generator.sample_critical_points(n)
            xi_b = pm.Normal("xi_B", shape=n)

            theta, theta_star = self.generate_thetas(xi_b,xi_bs,z_bbs)

            self.observations3 = snr * theta_star - e

            likelihood = pm.Bernoulli("likelihood", p=self.sigmoid(snr * theta),
                                      observed=self.indicator(self.observations3))

    def draw_samples(self):
        with self.mean_field_model1:
            self.sample1 = pm.sample(draws = self.draws, chains = self.chains, progressbar=False, tune=self.tune, cores=4)

        with self.mean_field_model2:
            self.sample2 = pm.sample(draws = self.draws, chains = self.chains, progressbar=False, tune=self.tune, cores=4)

        with self.mean_field_model3:
            self.sample3 = pm.sample(draws = self.draws, chains = self.chains, progressbar=False, tune=self.tune, cores=4)

        self.posterior1 = np.array(self.sample1["posterior"]["beta"][0])
        self.posterior2 = np.array(self.sample2["posterior"]["theta"][0])
        self.posterior3 = np.array(self.sample3["posterior"]["theta"][0])

class CriticalPointsGenerator:
    def __init__(self,probabilistic_model):
        self.probabilistic_model = probabilistic_model
        n, p = self.probabilistic_model.get_mean_field_parameters()[:2]
        self.delta = self.probabilistic_model.fixed_point_equations.mean_field_model.delta
        self.kappa = p / n

    def __is_critical(self, xi, z, e):
        k = self.kappa
        v_b, c_b, c_bbs = self.probabilistic_model.get_mean_field_parameters[:3]

        if c_b > 0:
            theta = np.sqrt(k * (self.probabilistic_model.gamma**2 - c_bbs**2 / c_b)) * xi + c_bbs * np.sqrt(k / c_b) * z
        else:
            theta = np.sqrt(k) * self.probabilistic_model.gamma * xi

        return abs(theta - e) < 5 * self.delta

    def __sample_one_critical(self):
        n_samples = 0

        while True:
            n_samples += 1
            e = self.probabilistic_model.logit(np.random.random())
            xi, z = np.random.normal(0.0, 1.0, size=2)

            if self.__is_critical(xi, z, e):
                return xi, z, e, n_samples

    def sample_critical_points(self, n):
        total_samples = 0

        xi = np.zeros(n)
        z = np.zeros(n)
        e = np.zeros(n)

        for i in range(n):
            xi[i], z[i], e[i], n_samples = self.__sample_one_critical()
            total_samples += n_samples

        return xi, z, e, total_samples
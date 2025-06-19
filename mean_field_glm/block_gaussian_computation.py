import numpy as np
from IPython.display import clear_output
from mean_field_glm.model_gaussian import MeanFieldGaussianGLM

class BlockGaussianComputation:
    """
    BlockComputation computes iteratively many fixed points derived from the MeanFieldGLM class.

    Parameters
    ----------
    var_tuple : tuple, optional
        Tuple of values for the variable of interest (e.g., kappa, snr). Default is (0.1,1.0).
    variable : str, optional
        Name of the variable being varied. Should be either "kappa" or "snr". Default is "kappa".
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
        If True, saves the computed data. Default is True.
    bayes_optimal : bool, optional
        If True, assumes the MeanFieldGLM model is Bayes optimal and uses Nishimori identities. Default is False.

    Attributes
    ----------
    data : NoneType
        Placeholder for the fixed points computed.

    Notes
    -----
    - The `BlockComputation` class is designed to work in conjunction with the `MeanFieldGLM`.

    """
    def __init__(self, var_tuple= (0.1,1.0), init_params = (1.0,0.0,0.0,1e-6,0.0,0.0), variable="kappa", num_per_var=5,
                 delta=0.001, fixed_var=1.0, signal="Normal", tolerance = 0.02, max_it = 7,
                 log_likelihood = "Logistic", save=True, bayes_optimal=False, file_name = "Computed_data", seed = None):
        self.var_tuple = var_tuple
        self.num_per_var = num_per_var
        self.max_it = max_it
        self.init_params = init_params
        self.delta = delta
        self.log_likelihood = log_likelihood
        self.signal = signal
        self.fixed_var = fixed_var
        self.save = save
        self.bayes_optimal = bayes_optimal
        self.variable = variable
        self.file_name = file_name
        self.seed = seed
        self.stats = None
        self.data = None
        self.tolerance = tolerance

    def compute_data(self):
        """
        Computes many fixed points based on the specified variable and its values.

        For each value in var_tuple:
        - If variable is 'kappa', computes MeanFieldGLM with varying kappa and fixed snr.
        - If variable is 'snr', computes MeanFieldGLM with varying snr and fixed kappa.
        Collects order parameters and stores them in self.data.

        If save is True, saves the computed data with order parameters to file.
        """
        self.data = np.zeros(shape=(len(self.var_tuple)*self.num_per_var, 9))

        for i in range(len(self.var_tuple)):
            variable_value = self.var_tuple[i]

            for n in range(self.num_per_var):
                print(f"%%%%%%%%%% COMPUTING DATA POINT {n+1} OUT OF {self.num_per_var} FOR {self.variable.upper()} = {variable_value} %%%%%%%%%%\n\n")

                # Compute data point for the current variable value
                self.compute_data_for_variable(variable_value, n, i, self.data)

                clear_output(wait=True)

            if self.save:
                self.save_data()

    def compute_data_for_variable(self, variable_value, n, i, data):
        """
        Computes a single fixed point for the specified variable and its values.

        Stores the values of the order parameters in the fixed point and parameters/seed used in data.
        """
        if self.variable == "kappa":
            model = MeanFieldGaussianGLM(kappa=variable_value,v_b=self.init_params[0],c_b=self.init_params[1],
                                 c_bbs=self.init_params[2],r_1=self.init_params[3],r_2=self.init_params[4],
                                 r_3=self.init_params[5], max_it=self.max_it, log_likelihood = self.log_likelihood,
                                 p=1000, n=1000, delta=self.delta, tolerance=self.tolerance,
                                 signal=self.signal, snr=self.fixed_var,
                                 seed=self.seed)
        elif self.variable == "snr":
            model = MeanFieldGaussianGLM(snr=variable_value,v_b=self.init_params[0],c_b=self.init_params[1],
                                 c_bbs=self.init_params[2],r_1=self.init_params[3],r_2=self.init_params[4],
                                 r_3=self.init_params[5], max_it=self.max_it, log_likelihood = self.log_likelihood,
                                 p=1000, n=1000, delta=self.delta, tolerance=self.tolerance,
                                 signal=self.signal, kappa=self.fixed_var,prior_sigma=variable_value+3.0,
                                 seed=self.seed)
        else:
            raise ValueError(f"Unsupported variable type: {self.variable}")

        model.run_iterations()
        order_parameters = model.show_order_parameters(show=False, output=True)

        data_index = n + self.num_per_var * i
        data[data_index, 0] = model.seed
        data[data_index, 1] = variable_value
        data[data_index, 2] = self.fixed_var

        for j in range(6):
            data[data_index, j + 3] = order_parameters[j]

    def check_if_data(self):
        """
        Checks if self.data has been computed.

        Raises:
        - ValueError: If self.data is None, indicating data has not been computed.
        """
        if self.data is None or not np.any(self.data):
            print("Please, compute the data first!")
            raise ValueError()

    def save_data(self):
        """
        Saves the computed data to a CSV file.

        The file generated contains the following columns:
        SEED | ENUMERATED VARIABLE | FIXED VARIABLE | V_B | C_B | C_BBS | R_1 | R_2 | R_3

        Parameters:
        - name (str): Name of the CSV file (default: "Computed data").
        """
        self.check_if_data()  # Ensure data is computed
        np.savetxt(self.file_name + ".csv", self.data, delimiter=",")
import numpy as np
from IPython.display import clear_output
from mean_field import MeanFieldGLM

class BlockComputation():
    """
    DataGraphGLM computes graphs derived from the MeanFieldGLM class.

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
    data : NoneType
        Placeholder for any required data storage.

    Notes
    -----
    - The `BlockComputation` class is designed to work in conjunction with the `MeanFieldGLM`.

    """
    def __init__(self, var_list= (0.1,1.0), init_params = (1.0,0.0,0.0,1e-6,0.0,0.0), variable="kappa", num_per_var=5,
                 delta=1.0, fixed_var=1.0, prior="Normal", signal="Normal", tolerance = 0.01, max_it = 7,
                 log_likelihood = "Logistic", save=True, bayes_optimal=False):
        self.var_list = var_list
        self.num_per_var = num_per_var
        self.max_it = max_it
        self.init_params = init_params
        self.delta = delta
        self.log_likelihood = log_likelihood
        self.prior = prior
        self.signal = signal
        self.fixed_var = fixed_var
        self.save = save
        self.bayes_optimal = bayes_optimal
        self.variable = variable
        self.stats = None
        self.data = None
        self.tolerance = tolerance

    def compute_graph_data(self):
        """
        Computes the data points for the graph based on the specified variable and its values.

        For each value in var_list:
        - If variable is 'kappa', computes MeanFieldGLM with varying kappa and fixed snr.
        - If variable is 'snr', computes MeanFieldGLM with varying snr and fixed kappa.
        Collects critical quantities and stores them in self.data.

        If save is True, saves the computed data with critical quantities to file.
        """
        data = np.zeros(shape=(len(self.var_list)*self.num_per_var, 8))

        for i in range(len(self.var_list)):
            variable_value = self.var_list[i]

            for n in range(self.num_per_var):
                print(f"%%%%%%%%%% COMPUTING DATA POINT {n+1} OUT OF {self.num_per_var} FOR {self.variable.upper()} = {variable_value} %%%%%%%%%%\n\n")

                # Compute data point for the current variable value
                self.compute_data_for_variable(variable_value, n, i, data)

                clear_output(wait=True)

            self.data = data

        if self.save:
            self.save_data()

    def compute_data_for_variable(self, variable_value, n, i, data):
        """
        Computes a single data point for the graph based on the specified variable and its values.

        For each value in var_list:
        - If variable is 'kappa', computes MeanFieldGLM with varying kappa and fixed snr.
        - If variable is 'snr', computes MeanFieldGLM with varying snr and fixed kappa.
        Collects critical quantities and stores them in self.data.

        If save is True, saves the computed data with critical quantities to file.
        """
        if self.variable == "kappa":
            model = MeanFieldGLM(kappa=variable_value,v_b=self.init_params[0],c_b=self.init_params[1],
                                 c_bbs=self.init_params[2],r_1=self.init_params[3],r_2=self.init_params[4],
                                 r_3=self.init_params[5], max_it=self.max_it, log_likelihood = self.log_likelihood,
                                 p=3000, n=3000, delta=self.delta, prior=self.prior,tolerance=self.tolerance,
                                 signal=self.signal, snr=self.fixed_var, bayes_optimal=self.bayes_optimal)
        elif self.variable == "snr":
            model = MeanFieldGLM(snr=variable_value,v_b=self.init_params[0],c_b=self.init_params[1],
                                 c_bbs=self.init_params[2],r_1=self.init_params[3],r_2=self.init_params[4],
                                 r_3=self.init_params[5], max_it=self.max_it, log_likelihood = self.log_likelihood,
                                 p=3000, n=3000, delta=self.delta, prior=self.prior,tolerance=self.tolerance,
                                 signal=self.signal, kappa=self.fixed_var, bayes_optimal=self.bayes_optimal)
        else:
            raise ValueError(f"Unsupported variable type: {self.variable}")

        model.run_iterations()
        quantities = model.show_order_parameters(show=False, output=True)

        data_index = n + self.num_per_var * i
        data[data_index, 0] = variable_value
        data[data_index, 1] = self.fixed_var

        for j in range(6):
            data[data_index, j + 2] = quantities[j]

    def check_if_data(self):
        """
        Checks if self.data is computed.

        Raises:
        - ValueError: If self.data is None, indicating data has not been computed.
        """
        if self.data is None or not np.any(self.data):
            print("Please, compute the data first!")
            raise ValueError()

    def save_data(self, name="Computed data"):
        """
        Saves the computed data to a CSV file.

        Parameters:
        - name (str): Name of the CSV file (default: "Computed data").
        """
        self.check_if_data()  # Ensure data is computed
        np.savetxt(name + ".csv", self.data, delimiter=",")
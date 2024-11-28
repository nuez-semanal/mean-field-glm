import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mean_field_model.block_computation import BlockComputation

block_arguments = {"variable" : "kappa",
                   "fixed_var" : 1.0,
                   "num_per_var" : 10,
                   "var_list" : [0.001, 0.01, 0.1, 1.0, 10.0],
                   "prior" : "Normal",
                   "signal" : "Normal",
                   "log_likelihood" : "Logistic",
                   "file_name" : "order_parameters_vs_snr",
                   "bayes_optimal" : True}

block = BlockComputation(**block_arguments)

block.compute_data()
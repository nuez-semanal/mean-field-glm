import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mean_field_glm as mf

def main():
    block_arguments = {"variable" : "kappa",
                       "fixed_var" : 1.0,
                       "num_per_var" : 10,
                       "var_tuple" : (5.0, 4.0, 3.0, 2.0),
                       "prior" : "Normal",
                       "signal" : "Normal",
                       "log_likelihood" : "Logistic",
                       "file_name" : "order_parameters_vs_kappa",
                       "bayes_optimal" : True}

    block = mf.block_computation.BlockComputation(**block_arguments)

    block.compute_data()

if __name__ == "__main__":
    main()
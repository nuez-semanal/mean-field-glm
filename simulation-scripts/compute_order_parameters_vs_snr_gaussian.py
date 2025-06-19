import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mean_field_glm as mf

def main():
    block_arguments = {"variable": "snr",
                       "fixed_var": 1.0,
                       "num_per_var": 10,
                       "var_tuple": (0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                                     3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
                                     7.0, 8.0, 9.0, 10.0, 15.0),
                       "signal": "Normal",
                       "log_likelihood": "Logistic",
                       "file_name": "order_parameters_vs_snr_gaussian",
                       "bayes_optimal": True}

    block = mf.block_gaussian_computation.BlockGaussianComputation(**block_arguments)
    block.compute_data()

if __name__ == "__main__":
    main()
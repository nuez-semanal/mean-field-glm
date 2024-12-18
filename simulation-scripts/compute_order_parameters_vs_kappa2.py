import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mean_field_glm import block_computation

def main():
    block_arguments = {"variable": "kappa",
                       "fixed_var": 100.0,
                       "num_per_var": 20,
                       "var_tuple": (1/10,3/10,1/2,7/10,9/10,11/10,13/10,15/10),
                       "prior": "Beta",
                       "signal": "Beta",
                       "log_likelihood": "Logistic",
                       "file_name": "order_parameters_vs_kappa2",
                       "bayes_optimal": False}

    block = block_computation.BlockComputation(**block_arguments)
    block.compute_data()

if __name__ == "__main__":
    main()
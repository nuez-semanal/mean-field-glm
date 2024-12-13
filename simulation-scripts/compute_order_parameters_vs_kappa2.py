import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mean_field_glm import block_computation

def main():
    block_arguments = {"variable": "kappa",
                       "fixed_var": 10000.0,
                       "num_per_var": 10,
                       "var_tuple": (1/8,1/6,1/4,1/2),
                       "prior": "Beta",
                       "signal": "Beta",
                       "log_likelihood": "Logistic",
                       "file_name": "order_parameters_vs_kappa",
                       "bayes_optimal": True}

    block = block_computation.BlockComputation(**block_arguments)
    block.compute_data()

if __name__ == "__main__":
    main()
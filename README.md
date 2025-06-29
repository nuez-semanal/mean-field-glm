# Bayesian GLM Mean Field Equations simulator

This repository contains numerical tools to simulate the Fixed Point Equations associated to Bayesian GLMs as described by the theory in the paper "Finite Marginals of Bayesian GLMs in the High-Dimensional Regime" by Manuel Sáenz and Pragya Sur.

## Features

- **MeanFieldGLM**: A class for simulating the GLM Fixed Point Equations. For this, MCMC simulations for the low dimensional associated measures are implemented. Support for Bayesian GLMs with different priors, signal distributions, and likelihoods.
- **BlockComputation**: Tools for automatically computing many fixed points interatively.
- **MseGraphCreator**: Methods for generating graphs and visualizing Mean Squared Error (MSE) across varying model parameters.

## Installation

Clone the repository:
   ```bash
   git clone https://github.com/nuez-semanal/mean-field-glm.git
   ```
## Dependencies

- **Python 3.12+**: Ensure you have Python installed.  
- **Dependencies Management**: This project uses `Poetry` for dependency management. All required packages are specified in `pyproject.toml`. 
- **Packages used**: `numpy`, `pymc`, and `matplotlib`.

### Installation Instructions

1. Install Poetry if not already available:
   ```bash
   pip install poetry
   ```
2. Install the project's dependencies:
   ```bash
   poetry install
   ```

To handle dependencies, a poetry toml file is included in the repository.

## Usage

### 1. Define a Mean-Field GLM
Use the `MeanFieldGLM` class to define your Bayesian GLM with desired parameters:

```python
from mean_field_glm.model_beta import MeanFieldGLM

model = MeanFieldGLM(
    p=1000,
    n=500,
    log_likelihood="Logistic",
    prior="Beta",
    signal="Beta",
    snr=5.0,
    tolerance=0.01
)

model.run_iterations()

model.show_order_parameters()
```

### 2. Compute many fixed points

To compute many fixed points at once, use the `BlockComputation` class:

```python
from mean_field_glm.block_beta_computation import BlockComputation

block = BlockComputation(
   var_list=[0.5, 1.0, 2.0],
   variable="kappa",
   log_likelihood="Linear",
   prior="Normal",
   signal="Rademacher",
   num_per_var=10,
)

block.compute_data()

block.save_data(name="Results of simulations")
```

### 3. Generate MSE Graphs
Visualize the Mean Squared Error (MSE) for different model parameters using `MseGraphCreator`:

```python
from mean_field_glm.graphs import MseGraphCreator

graph_creator = MseGraphCreator(
    var_list=[0.5, 1.0, 2.0],
    variable="kappa",
    log_likelihood="Linear",
    prior="Normal",
    signal="Rademacher",
    num_per_var = 10,
)

graph_creator.plot_graph_MSE(save = True)
```

## Project Structure

```
.
├── mean-field-glm/           # Folder containing all class implementations
│   ├── auxiliary.py            # Auxiliary functions for the project
│   ├── model.py                # MeanFieldGLM class implementation
│   ├── block_computation.py    # BlockComputation class implementation
│   ├── graphs.py               # MseGraphCreator class implementation
├── simulation-results/       # Folder containing graphs, config, and csv files of results
├── simulation-scripts/       # Folder containing scripts to compute data and create graphs
│   ├── compute_..._kappa.py    # Compute simulations for Logistic Normal-Normal for varying kappa
│   ├── compute_..._snr.py      # Compute simulations for Logistic Normal-Normal for varying SNR
│   ├── mse_graph_from_data.py  # Create an MSE graph from previously computed data
├── pyproject.toml              # Poetry configuration file
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation (this file)
```

## License

This project is licensed under the MIT License. 

## Acknowledgments

- **Pymc**: For providing robust tools for Bayesian modeling.
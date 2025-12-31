# Bayesian GLM Mean Field Equations simulator

This repository contains the numerical tools used to simulate the Fixed Point Equations associated to Bayesian GLMs as described by the theory in the paper "Characterizing Finite-Dimensional Posterior Marginals in  High-Dimensional GLMs via Leave-One-Out" by Manuel Sáenz and Pragya Sur. It also contains code to rune MCMC simulations of the relevant high-dimensional inference problems and to simulate the limits of associated MLEs.

## Dependencies and virtual environment

Clone the repository:
   ```bash
   git clone https://github.com/nuez-semanal/mean-field-glm.git
   ```
## Dependencies

- **Python 3.12+**: Ensure you have Python installed.  
- **Dependencies Management**: This project uses `Poetry` for dependency management. All required packages are specified in `pyproject.toml`. 

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

3. Once the environment is set, run in it with the command
   ```bash
   poetry shell
   ```

## Usage

To reproduce simulations, run the corresponding scripts provided in ```simulation-scripts``` folder. The results of simulations and graphs are stored in ```simulation-results```. 

## Acknowledgments

We thank the PyMC team for providing fantastic tools for Bayesian modeling.
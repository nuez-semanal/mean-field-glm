import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mean_field_glm.model_gaussian import MeanFieldGaussianGLM

model = MeanFieldGaussianGLM(kappa=0.5,snr = 3.0,prior_sigma=3.0,log_likelihood="Logistic",signal="Normal")
model.run_iterations()
model.show_order_parameters(show=True,output=True)
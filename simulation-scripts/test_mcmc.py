import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bayesian_glm.model import ModelGLM


for i in range(5):
    model = ModelGLM(p=500, n= 1000,log_likelihood="Logistic", signal="Normal", prior="Normal", gamma=3.0, sigma=3.0)
    model.draw_sample()
    print(model.compute_order_parameters())
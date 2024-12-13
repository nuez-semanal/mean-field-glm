import numpy as np
from mean_field_glm.marginal import MeanFieldMarginalGLM

k = np.random.randint(900)

hq = 0.1800
parameters = [ hq, hq, np.sqrt(hq) ]

marginal = MeanFieldMarginalGLM(parameters = parameters,
                                prior = "Normal",
                                snr = 1.0,
                                betas = true_beta[k],
                                noise = noise[k])
marginal.sample()
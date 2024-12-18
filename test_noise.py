from bayesian_glm import ModelGLM

model = ModelGLM(n=300,p=200)

model.draw_sample()
model.compute_order_parameters()
model.compute_hq()
model.compute_noise(25)
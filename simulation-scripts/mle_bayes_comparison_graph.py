import numpy as np
import matplotlib.pyplot as plt

variable_map = {"kappa": 0, "gamma": 1}
data_map = {"alpha": 2, "sigma": 4, "mse": 6}

variable = input("What varies? [kappa/gamma] ")
graph_type = input("What type of graph? [alpha/sigma/mse] ")

assert variable in ["kappa","gamma"], "Not valid type of variable!"
assert graph_type in ["alpha","sigma","mse"], "Nos valid type of graph!"

file_path = "./simulation-results/comparison_mle_bayes_"+variable+".csv"
data = np.loadtxt(file_path, delimiter=',',skiprows=1)

variable_value = data[:,variable_map[variable]]

plt.style.use('ggplot')
plt.figure(figsize=(8, 6))

data_bayes = data[:,data_map[graph_type]]
data_mle = data[:,data_map[graph_type]+5]

plt.scatter(data_bayes,data_mle,marker='x')

for i in range(len(variable_value)):
    plt.text(data_bayes[i] + 0.007, data_mle[i], variable+" = "+str(variable_value[i]))

plt.xlabel("Bayes " + graph_type, fontsize=14)
plt.ylabel("MLE " + graph_type, fontsize=14)
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.minorticks_on()

plt.savefig("comparison_mle_bayes_"+graph_type+"_"+variable+".png", dpi=1200)
plt.show()
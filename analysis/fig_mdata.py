import sys
import numpy as np
import matplotlib.pyplot as plt

#=========================#
# data load
#=========================#
#set the fitting range and target files

args = sys.argv

f0 = open('./matrixelement_data/mdata', 'rt')

x0 = []
y0 = []
y0_err = []

for string in f0:
    data = string[:-1].split(' ')
    x0.append(float(data[0]))
    y0.append(float(data[1]))
    y0_err.append(float(data[2]))

colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))

# error bar
plt.errorbar(x0, y0, yerr = y0_err, capsize=5, fmt='o', markersize=5, ecolor=colors[0], markeredgecolor=colors[0], color=colors[0])
plt.xlabel('chebyshev order', fontsize=15)
plt.ylabel('matrix elements', fontsize=15)
plt.ylim(-2.5,2.5)
plt.axhspan(1.0, 5.0, color='black', alpha=0.1)
plt.axhspan(-5.0, -1.0, color='black', alpha=0.1)
plt.show()


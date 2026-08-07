import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot

#=============================================================#
#=============================================================#
# Confidence interval 
confidence_factor = 0.68
q_upper = 1. - (1. - confidence_factor) * 0.5
q_lower = 1. - q_upper

#=============================================================#
#=============================================================#
# data generation

args = sys.argv
jkfile = args[1]

bootstrap_means = []
f = open(jkfile, 'rt')
for string in f:
    data = string[:-1].split(' ')
    bootstrap_means.append(float(data[1]))
    # bootstrap_means.append(-np.log(float(data[1])))
Bs = len(bootstrap_means)
boot_means = np.nan_to_num(bootstrap_means)

#=============================================================#
#=============================================================#
# QQ 

probplot(boot_means, dist="norm", plot=plt)
plt.title('Quantile-Quantile plot')
plt.grid(True)
plt.tight_layout()
plt.savefig('QQplot.pdf')
plt.show()

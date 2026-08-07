import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis, probplot
from sklearn.utils import resample

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
# 68% confidence interval for the bootstrap mean 
ci_boot_lower = np.percentile(boot_means, q_lower*100)
ci_boot_upper = np.percentile(boot_means, q_upper*100)

# expectation value and standard deviation for the bootstrap mean
mu = np.mean(boot_means)
sigma = np.std(boot_means)

st_lower = mu - sigma
st_upper = mu + sigma

# skewness and kurtosis 
g1 = skew(boot_means)
g2 = kurtosis(boot_means, fisher=True)  # fisher=Truen for normalized

er_boot_lower = mu - ci_boot_lower
er_boot_upper = ci_boot_upper - mu

# using median
median = np.median(boot_means)
me_boot_lower = median - ci_boot_lower
me_boot_upper = ci_boot_upper - median

#=============================================================#
#=============================================================#
# Cornish-Fisher expansion correction (3rd order)

def Cumulants4th(data, order):
    moments = [np.mean(data**k) for k in range(1, order+1)]
    
    cumulants = []
    # C1: mean
    cumulants.append(moments[0])
    
    if order >= 2:
        # C2: variance
        cumulants.append(moments[1] - moments[0]**2)
    if order >= 3:
        # C3: third cumulant (related to skewness)
        mu1, mu2, mu3 = moments[0], moments[1], moments[2]
        cumulants.append(mu3 - 3*mu1*mu2 + 2*mu1**3)
    if order >= 4:
        # C4: fourth cumulant (related to kurtosis)
        mu1, mu2, mu3, mu4 = moments[0], moments[1], moments[2], moments[3]
        c4 = mu4 - 4*mu1*mu3 - 3*mu2**2 + 12*mu1**2*mu2 - 6*mu1**4
        cumulants.append(c4)
    
    return cumulants

def cornish_fisher(z, g1, g2):
    term1 = (g1 / 6) * (z**2 - 1)
    term2 = (g2 / 24) * (z**3 - 3*z)
    term3 = - (g1**2 / 36) * (2*z**3 - 5*z)
    return z + term1 + term2 + term3

cumulants = Cumulants4th(np.array(boot_means), 4)

# z score for any-% confidence interval 
z_lower = norm.ppf(q_lower)
z_upper = norm.ppf(q_upper)

# Cornish–Fisher correction 
q_lower = cornish_fisher(z_lower, g1, g2)
q_upper = cornish_fisher(z_upper, g1, g2)
ci_cf_lower = mu + q_lower * sigma
ci_cf_upper = mu + q_upper * sigma

#=============================================================#
#=============================================================#
# Print results
print("cumulants:", cumulants[0], cumulants[1], cumulants[2], cumulants[3])

print("Mean, 68%CF:", mu, q_lower*sigma, q_upper*sigma)
print("Mean, 68%BS:", mu, er_boot_lower, er_boot_upper)
print("Median, 68%BS:", median, me_boot_lower, me_boot_upper)

#=============================================================#
#=============================================================#
# histogram 
plt.figure(figsize=(10, 6))
plt.hist(boot_means, bins=int(Bs*0.5), density=True, alpha=0.5, label='Bootstrap Mean Distribution')
plt.axvline(mu, color='black', linestyle='-', label='Sample Mean')
# plt.axvline(median, color='brown', linestyle='-', label='Sample Median')

# Standard
plt.axvline(st_lower, color='red', linestyle='dotted', label='Bootstrap 1$\\sigma$ standard deviation')
plt.axvline(st_upper, color='red', linestyle='dotted')

# Bootstrap 
plt.axvline(ci_boot_lower, color='blue', linestyle='--', label='Bootstrap 68% CI')
plt.axvline(ci_boot_upper, color='blue', linestyle='--')


# Cornish-Fisher interval
plt.axvline(ci_cf_lower, color='green', linestyle='-.', label='CF 68% CI')
plt.axvline(ci_cf_upper, color='green', linestyle='-.')


plt.title('Comparison of Cornish–Fisher and Bootstrap 68% Confidence Intervals')
plt.xlabel('Mean Estimate')
plt.ylabel('Density')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('dist.pdf')
plt.show()


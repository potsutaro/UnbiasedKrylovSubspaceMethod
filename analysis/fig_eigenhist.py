import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

#=========================#
# data load
#=========================#
#set the fitting range and target files

args = sys.argv
filename = args[1]
gilename = args[2]
Ba = int(args[3])
Bg = int(Ba*0.5)

f = open(filename, 'rt')
g = open(gilename, 'rt')

dlist0 = []
ylist1 = []
dlist1 = []

lin = 0
for string in f:
    data = string[:-1].split(' ')
    dlist0.append(float(data[0]))
    lin = lin + 1

lin = 0
for string in g:
    data = string[:-1].split(' ')
    dlist1.append(float(data[0]))
    lin = lin + 1


#=========================#
# Show Histogram
#=========================#

hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
plt.title('Bootstrap Effective mass histogram')
plt.xlabel('$E^{(m)}_n = -\\ln{\\mathrm{Re}[\\lambda^{(m)}_0]}$', fontsize=15)
plt.hist(dlist0, color=hcolors[1], bins=Ba, label="Unfiltered")
# plt.hist(dlist0, color=hcolors[1], bins=Ba, label="Unfiltered")
# plt.hist(dlist1, color=hcolors[0], bins=Bg, label="Filtered")
plt.legend(fontsize=13)
plt.xlim(0.0, 1.8)
plt.ylim(0, 100)
plt.savefig("EigenHist.pdf")
plt.show()


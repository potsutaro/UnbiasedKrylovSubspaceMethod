import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

#=========================#
# data load
#=========================#
#set the fitting range and target files

args = sys.argv
fileN = []
tsep = int(args[1])
for ts in range(tsep+1):
    fileN.append(str(ts))

filename = './traditional_data/ratio'

for i in range(len(fileN)):
    ifilename = filename + fileN[i]

    y_list = []
    e_list = []
    x_list = []


    lin = 0
    f = open(ifilename, 'rt')
    for string in f:
        if (lin <= int(fileN[i])):
            data = string[:-1].split(' ')
            x_list.append(float(data[0])-float(fileN[i])*0.5)
            y_list.append(float(data[1]))
            e_list.append(float(data[2]))
        lin = lin + 1

    plt.errorbar(x_list, y_list, e_list, linestyle='solid', capsize=5, fmt='o', markersize=5, label=fileN[i])

plt.xlabel("$t$",fontsize=15)
plt.hlines(1.0, -tsep*0.5, tsep*0.5, color='gray', linestyles='dotted')
plt.ylabel("$R(t;t_\\mathrm{sep})$",fontsize=15)
plt.legend()
plt.show()

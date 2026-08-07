import sys
import numpy as np
import scipy as sp
import copy
import math
import cmath
import statistics
import matplotlib.pyplot as plt

import TransferGEVP

#=============================================================#
# input parameters for a target state with a certain rank 

args = sys.argv

svdrank = int(args[11])
Istate = 0
Fstate = 0

#=============================================================#
#set the fitting range and target files

tole = 10**(-12)
zcwtole = 10**(-6)

exx_list = []
exy_list = []

rex_list = [] # mean value
rey_list = []

cor_list = []

inx_list = []
iny_list = []

lin = 0
lower = float(args[1])
upper = float(args[2])
fitrange = [lower, upper]
nt = int(args[3])
dof = int(upper - lower + 1 )
datapath = args[5] # for results
datafile = args[6]

size = int(args[4])+1

flag = int(args[7]) # 0 for Jackknife, 1 for Bootstrap

NT = int(args[8])

resultspath = args[9]

t0 = int(args[10])


jkfile = datapath + '/' + datafile + '_jk'
avfile = datapath + '/' + datafile 

f = open(jkfile,'rt');
f2 = open(avfile, 'rt')

for string in f:
    if fitrange[0] <= (lin % size) <= fitrange[1]:
        data = string[:-1].split(' ')
        inx_list.append(float(data[0]))
        iny_list.append(float(data[1]))
        if (lin % size) == fitrange[1]:
            exx_list.append(inx_list)
            exy_list.append(iny_list)
            inx_list = []
            iny_list = []
    lin += 1

lin = 0

sig_list = []

for string in f2:
    if fitrange[0] <= (lin % size) <= fitrange[1]:
        data = string[:-1].split(' ')
        rex_list.append(float(data[0]))
        rey_list.append(float(data[1]))
        sig_list.append(float(data[2]))
    lin += 1

lin = 0

#=============================================================#
#=============================================================#
# mass and normalization

mass_mean = 0.4

normalization_mean = 0
normalization_err = 0

normfile = './correlator_data/normalization'

nfile = open(normfile, 'rt')

for string in nfile:
    data = string[:-1].split(' ')
    normalization_mean = (float(data[0]))
    normalization_err = (float(data[1]))

#=============================================================#
#=============================================================#
# standard effective mass plot for a comparsion

mx_via = []
my_via = []
me_via = []

mx_list = []
my_list = []
me_list = []

mavfile = datapath + '/mass'

g2 = open(mavfile, 'rt')

for string in g2:
    if fitrange[0] <= (lin % NT) <= fitrange[1]:
        data = string[:-1].split(' ')
        mx_list.append(float(data[0]))
        my_list.append(float(data[1]))
        me_list.append(float(data[2]))
    lin += 1

lin = 0

#=============================================================#
#=============================================================#
# Bootstrap

conf = len(exx_list)
Bs = conf 


#=============================================================#
#=============================================================#
# list for the results

compevs_jk = []
compevs = []

realevs_jk = []
realevs = []

compvec_jk = []
compvec = []

lcompvec_jk = []
lcompvec = []

realvec_jk = []
realvec = []

singularvalues_jk = []

#=============================================================#
#=============================================================#
# matrix size

m = int(math.floor(nt*0.5 + 0.5))

#=============================================================#
#=============================================================#
# loop with configuration

mmin = 0
for i in range(conf):

    # Eigenvalues
    compevs_via = []
    realevs_via = []

    compvec_via = []
    realvec_via = []

    lcompvec_via = []

    # Singular values
    singularvalues_via = []

    for j in range(mmin, m):
        T, V = TransferGEVP.construct_TV(exy_list[i], j, 0) # 0 for 0-shift = naive
        eigenvalues, leigenvectors, eigenvectors, singularvalues = TransferGEVP.SingularValueDecompositionGEVP(T,V,svdrank)
        realev, compev, realve, compve, lcompve = TransferGEVP.RealOrComp_eigenvalue(eigenvalues, eigenvectors, leigenvectors)


        realevs_via.append(realev)
        compevs_via.append(compev)

        realvec_via.append(realve)
        compvec_via.append(compve)

        lcompvec_via.append(lcompve)

        singularvalues_via.append(singularvalues)


    realevs_jk.append(realevs_via)
    compevs_jk.append(compevs_via)

    realvec_jk.append(realvec_via)
    compvec_jk.append(compvec_via)

    lcompvec_jk.append(lcompvec_via)

    singularvalues_jk.append(singularvalues_via)

#=============================================================#
#=============================================================#

print("Complex eigenvalues")
for i in range(conf):
    for j in range(len(compevs_jk[i])):
        print(j, compevs_jk[i][j])

# print("Complex eigenvectors")
# for i in range(conf):
#     for j in range(len(compevs_jk[i])):
#         print(j, compvec_jk[i][j])
# 
#=============================================================#
#=============================================================#
# Get Physical and Ground

zcwevs_jk, zcwvec_jk, lzcwvec_jk = TransferGEVP.GetZCW(compevs_jk, compvec_jk, lcompvec_jk, exy_list, zcwtole)
print("ZCW eigenvalues")
for i in range(conf):
    for j in range(len(zcwevs_jk[i])):
        print(j, zcwevs_jk[i][j])

#=============================================================#
#=============================================================#
# Get Physical and Ground

physevs_jk, physvec_jk, lphysvec_jk = TransferGEVP.GetPhysical(zcwevs_jk, zcwvec_jk, lzcwvec_jk, zcwtole)
gsevs_jk, gsvec_jk, gslvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 0)

#=============================================================#
#=============================================================#
# Nested bootstrap

for j in range(mmin, m):
    ms = j + 1
    # Num.0 state
    jthevs_jk, jthvec_jk, jthlvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 0)
    jdata_jk = TransferGEVP.Evs2List(physevs_jk, j)
    histogram, jmed, jerr = TransferGEVP.NestedBootstrapEstimator(jdata_jk, Bs, Bs)
    print(jmed, jerr)


#=============================================================#
# #=============================================================#
# 
# print("Physical eigenvalues")
# for i in range(conf):
#     for j in range(len(physevs_jk[i])):
#         print(j, physevs_jk[i][j])
# 
# print("Physical eigenvectors")
# for i in range(conf):
#     for j in range(len(physevs_jk[i])):
#         print(j, physvec_jk[i][j][0].size, physvec_jk[i][j])
# 
# 
# #=============================================================#
# #=============================================================#
# # Ground state histogram
# 
# 
# hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
# for j in range(len(physevs_jk[0])):
#     gstate = []
#     astate = []
#     for i in range(conf):
#         gstate.append(-np.log(physevs_jk[i][j][0]))
#         for k in range(len(physevs_jk[i][j])):
#             astate.append(-np.log(physevs_jk[i][j][k]))
#     afile = TransferGEVP.PrintHist(astate, 'All', j)
#     gfile = TransferGEVP.PrintHist(gstate, 'Ground', j)
#     plt.title('Bootstrap Effective mass histogram')
#     plt.xlabel('$-\\ln{\\mathrm{Re}[\\lambda^{(m)}_0]}$', fontsize=15)
#     jlabel = "j=" + str(j)
#     plt.hist(astate, bins=Bs, label=jlabel, color=hcolors[1])
#     plt.hist(gstate, bins=Bs, label=jlabel, color=hcolors[0])
#     plt.legend(fontsize=18)
#     plt.xlim(0.0, 2.0)
#     plt.show()
# 
#=============================================================#
#=============================================================#
# Print effective mass and correlators



# #=============================================================#
# #=============================================================#
# # Figure
# colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
# 
# plt.errorbar(mx_list, my_list, yerr=me_list, capsize=5, fmt='^', markersize=5, ecolor=colors[0], markeredgecolor=colors[0], color=colors[0], label='Standard')
# plt.errorbar(lpx_list, lpy_list, yerr=lpe_list, capsize=5, fmt='*', markersize=5, ecolor=colors[2], markeredgecolor=colors[2], color=colors[2], label='TGEVP -1st excited State')
# plt.errorbar(lgx_list, lgy_list, yerr=lge_list, capsize=5, fmt='1', markersize=5, ecolor=colors[1], markeredgecolor=colors[1], color=colors[1], label='TGEVP -Ground State')
# 
# plt.hlines(0.1, 0, 64, color='gray', linestyles='dotted')
# plt.hlines(0.2, 0, 64, color='gray', linestyles='dotted')
# plt.hlines(0.3, 0, 64, color='gray', linestyles='dotted')
# plt.hlines(0.4, 0, 64, color='gray', linestyles='dotted')
# plt.hlines(0.5, 0, 64, color='gray', linestyles='dotted')
# plt.hlines(0.6, 0, 64, color='gray', linestyles='dotted')
# plt.hlines(0.7, 0, 64, color='gray', linestyles='dotted')
# plt.hlines(0.8, 0, 64, color='gray', linestyles='dotted')
# plt.hlines(0.9, 0, 64, color='gray', linestyles='dotted')
# plt.hlines(1.0, 0, 64, color='gray', linestyles='dotted')
# plt.hlines(1.1, 0, 64, color='gray', linestyles='dotted')
# 
# plt.xlabel('$t/a=2m-1$', fontsize=15)
# plt.ylabel('$-\\ln{\\mathrm{Re}[\\lambda^{(m)}_n]}$', fontsize=15)
# 
# plt.xlim(lower-1, upper+1)
# plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=18)
# 
# plt.show()

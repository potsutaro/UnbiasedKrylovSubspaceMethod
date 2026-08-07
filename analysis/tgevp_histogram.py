import sys
import numpy as np
import scipy as sp
import copy
import math
import cmath
import statistics
import matplotlib.pyplot as plt

import TransferGEVP
import TransferLanczos

#=============================================================#
#set the fitting range and target files

tole = 10**(-12)

exx_list = []
exy_list = []

rex_list = [] # mean value
rey_list = []

cor_list = []

inx_list = []
iny_list = []

args = sys.argv

lin = 0
lower = float(args[1])
upper = float(args[2])+1
fitrange = [lower, upper]
nt = int(args[3])
dof = int(upper - lower + 1 )
datapath = args[5] # for results
datafile = args[6]

size = int(args[4])+1

flag = int(args[7]) # 0 for Jackknife, 1 for Bootstrap

NT = int(args[8])

t0 = int(args[9])


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

print("correlator")
print(exy_list)


#=============================================================#
#=============================================================#
# mass and normalization

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

compvec_jk = []
compvec = []

lcompvec_jk = []
lcompvec = []

singularvalues_jk = []

#=============================================================#
#=============================================================#
# matrix size

m = int(math.floor(nt*0.5 + 0.5))

#=============================================================#
#=============================================================#
# loop with configuration

for i in range(conf):

    # Eigenvalues
    compevs_via = []

    compvec_via = []

    lcompvec_via = []

    # Singular values
    singularvalues_via = []

    for svdrank in range(m-1, m):
        T, V = TransferGEVP.construct_TV(exy_list[i], m-1, 0)
        print("T=",T)
        print("V=",V)
        eigenvalues, leigenvectors, eigenvectors, singularvalues = TransferGEVP.SingularValueDecompositionGEVP(T,V,svdrank)
        realev, compev, realve, compve, lcompve = TransferLanczos.RealOrComp_eigenvalue(eigenvalues, eigenvectors, leigenvectors)
        print(eigenvectors)


        compevs_via.append(compev)

        compvec_via.append(compve)

        lcompvec_via.append(lcompve)

        singularvalues_via.append(singularvalues)


    compevs_jk.append(compevs_via)

    compvec_jk.append(compvec_via)

    lcompvec_jk.append(lcompvec_via)

    singularvalues_jk.append(singularvalues_via)

#=============================================================#
#=============================================================#
# Get Physical and Ground

physevs_jk, physvec_jk, lphysvec_jk = TransferGEVP.GetPhysical(compevs_jk, compvec_jk, lcompvec_jk, tole)
gsevs_jk, gsvec_jk, gslvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 0)

#=============================================================#
#=============================================================#
# Ground state histogram

print("Physical")
hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
for j in range(len(physevs_jk[0])):
    gstate = []
    astate = []
    for i in range(conf):
        print(physevs_jk[i][j])
        for k in range(len(physevs_jk[i][j])):
            astate.append(-np.log(physevs_jk[i][j][k]))
    afile = TransferGEVP.PrintHist(astate, 'All', j)
    plt.title('Bootstrap Effective mass histogram')
    plt.xlabel('$-\\ln{\\mathrm{Re}[\\lambda^{(m)}_0]}$', fontsize=15)
    jlabel = "j=" + str(j)
    plt.hist(astate, bins=Bs, label=jlabel, color=hcolors[1])
    plt.hist(gstate, bins=Bs, label=jlabel, color=hcolors[0])
    plt.legend(fontsize=18)
    plt.xlim(0.0, 2.0)
    plt.show()


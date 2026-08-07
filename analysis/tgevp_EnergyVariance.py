import sys
import numpy as np
import scipy as sp
import copy
import math
import cmath
import statistics
import matplotlib.pyplot as plt

from . import TransferGEVP

from config import Config

#=============================================================#
# call as function

def main(cfg: Config):
    
    tole = 10**(-12)

    lower = cfg.analysis.init
    upper = cfg.analysis.size
    size = cfg.analysis.size
    dof = cfg.analysis.dof
    datapath = cfg.paths.corr
    datafile = cfg.files.ndata
    flag = cfg.statistics.stati
    T = cfg.lattice.T
    t0 = cfg.analysis.normalization
    svdrankmax = cfg.analysis.svdrankmax

    
    #=============================================================#
    #=============================================================#

    exx_list = []
    exy_list = []
    
    rex_list = [] # mean value
    rey_list = []
    
    cor_list = []
    
    inx_list = []
    iny_list = []
    
    lin = 0
    upper = upper + 1 # for variance
    dof = dof + 1 # for variance
    fitrange = [lower, upper]
    
    
    jkfile = datapath + '/' + datafile + '_jk'
    avfile = datapath + '/' + datafile 
    
    f = open(jkfile,'rt');
    f2 = open(avfile, 'rt')
    
    for string in f:
        if fitrange[0] <= (lin % dof) <= fitrange[1]:
            data = string[:-1].split(' ')
            inx_list.append(float(data[0]))
            iny_list.append(float(data[1]))
            if (lin % dof) == fitrange[1]:
                exx_list.append(inx_list)
                exy_list.append(iny_list)
                inx_list = []
                iny_list = []
        lin += 1
    
    lin = 0
    
    sig_list = []
    
    for string in f2:
        if fitrange[0] <= (lin % dof) <= fitrange[1]:
            data = string[:-1].split(' ')
            rex_list.append(float(data[0]))
            rey_list.append(float(data[1]))
            sig_list.append(float(data[2]))
        lin += 1
    
    lin = 0
    
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
        if fitrange[0] <= (lin % T) <= fitrange[1]:
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
    
    m = int(math.floor(size*0.5 + 0.5))
    
    #=============================================================#
    #=============================================================#
    # loop with configuration
    
    for i in range(conf):
    
        # Eigenvalues
        compevs_via = []
        realevs_via = []
    
        compvec_via = []
        realvec_via = []
    
        lcompvec_via = []
    
        # Singular values
        singularvalues_via = []
    
        for svdrank in range(1, svdrankmax):
            T, V = TransferGEVP.construct_TV(exy_list[i], m-1, 0)
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
    # Get Physical and Ground
    
    physevs_jk, physvec_jk, lphysvec_jk = TransferGEVP.GetPhysical(compevs_jk, compvec_jk, lcompvec_jk, tole)
    gsevs_jk, gsvec_jk, gslvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 0)
    
    #=============================================================#
    #=============================================================#
    # Singular values
    
    srun = TransferGEVP.FigSingularValues(singularvalues_jk, m)
    
    #=============================================================#
    #=============================================================#
    # ground state variance
     
    lpx_list = []
    lpy_list = []
    lpe_list = []
    
    ithevs_jk, ithvec_jk, ithlvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 0)
    ithvar_jk = TransferGEVP.EigenvalueVariance_same(exy_list, ithevs_jk, ithvec_jk, ithlvec_jk)
    ithvar_means, ithvar_errs = TransferGEVP.MultiStatisticalAnal(ithvar_jk, conf, 0)
    
    for i in range(conf):
        for j in range(len(ithvar_means)):
            for k in range(len(ithvar_means[j])):
                lpx_list.append(j+1)
                lpy_list.append(ithvar_means[j][k])
                lpe_list.append(ithvar_errs[j][k])
    
    rrun = TransferGEVP.PrintEigenvalueVariance(ithvar_means, ithvar_errs, 0)
    
    #=============================================================#
    #=============================================================#
    # print variance
    
    ithevs_jk, ithvec_jk, ithlvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 0)
    ithvar_jk = TransferGEVP.EigenvalueVariance_same(exy_list, ithevs_jk, ithvec_jk, ithlvec_jk)
    TransferGEVP.PrintSingleEigenvalueVariance(ithvar_jk, m, 0)

    #=============================================================#
    #=============================================================#
    # ground state residual bound
     
    lpx_list = []
    lpy_list = []
    lpe_list = []
    
    ithevs_jk, ithvec_jk, ithlvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 0)
    ithrb_jk = TransferGEVP.ResidualBound_same(exy_list, ithevs_jk, ithvec_jk, ithlvec_jk)
    ithrb_means, ithrb_errs = TransferGEVP.MultiStatisticalAnal(ithrb_jk, conf, 0)
    
    for i in range(conf):
        for j in range(len(ithrb_means)):
            for k in range(len(ithrb_means[j])):
                lpx_list.append(j+1)
                lpy_list.append(ithrb_means[j][k])
                lpe_list.append(ithrb_errs[j][k])
    
    rrun = TransferGEVP.PrintResidualBound(ithrb_means, ithrb_errs, 0)
    
    #=============================================================#
    #=============================================================#
    # print residual bound
    
    ithevs_jk, ithvec_jk, ithlvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 0)
    ithrb_jk = TransferGEVP.ResidualBound_same(exy_list, ithevs_jk, ithvec_jk, ithlvec_jk)
    TransferGEVP.PrintSingleResidualBound(ithrb_jk, m, 0)
    
    # #=============================================================#
    # #=============================================================#
    # # Figure
    # colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
    # 
    # plt.errorbar(lpx_list, lpy_list, yerr=lpe_list, capsize=5, fmt='1', markersize=5, ecolor=colors[1], markeredgecolor=colors[1], color=colors[1], label='TGEVP -Ground State')
    # 
    # plt.ylabel('$[\\lambda^{(m)}_n]^2$', fontsize=15)
    # 
    # plt.xlim(lower-1, upper+1)
    # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=18)
    # 
    # plt.show()


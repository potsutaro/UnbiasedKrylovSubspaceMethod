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
    
    Istate = 0
    Fstate = 0
    
    tole = 10**(-12)

    lower = cfg.analysis.init
    upper = cfg.analysis.size
    size = cfg.analysis.size
    dof = cfg.analysis.dof
    datapath = cfg.paths.corr
    datafile = cfg.files.ndata
    flag = cfg.statistics.stati
    T =  cfg.lattice.T
    resultspath = cfg.paths.diag
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
    fitrange = [lower, upper]
    dof = dof + 1
    
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
    # Bootstrap
    
    conf = len(exx_list)
    Bs = conf 

    for svdrank in range(1, svdrankmax):
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
        
        mmin = m-1
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
        
        print("Complex eigenvectors")
        for i in range(conf):
            for j in range(len(compevs_jk[i])):
                print(j, compvec_jk[i][j])
        
        #=============================================================#
        #=============================================================#
        # Get Physical and Ground
        
        physevs_jk, physvec_jk, lphysvec_jk = TransferGEVP.GetPhysical(compevs_jk, compvec_jk, lcompvec_jk, tole)
        gsevs_jk, gsvec_jk, gslvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 0)
        
        #=============================================================#
        #=============================================================#
        # Singular values
        
        # srun = TransferGEVP.FigSingularValues(singularvalues_jk, m)
        
        #=============================================================#
        #=============================================================#
        # make full correlators
        
        print("Make Full Correlator")
        
        correlator_time = 20
        fcorrelator_all = TransferGEVP.MakeFullCorrelator(compevs_jk, compvec_jk, lcompvec_jk, exy_list, normalization_mean, correlator_time, t0)
        fcorrelator_gs = TransferGEVP.MakeFullCorrelator(gsevs_jk, gsvec_jk, gslvec_jk, exy_list, normalization_mean, correlator_time, t0)
        
        frun = TransferGEVP.FigCorrelators(fcorrelator_all, fcorrelator_gs)
        
        #=============================================================#
        #=============================================================#
        
        print("Physical eigenvalues")
        for i in range(conf):
            for j in range(len(physevs_jk[i])):
                print(j, physevs_jk[i][j])
        
        print("Physical eigenvectors")
        for i in range(conf):
            for j in range(len(physevs_jk[i])):
                print(j, physvec_jk[i][j][0].size, physvec_jk[i][j])
        
        #=============================================================#
        #=============================================================#
        # organize complex and real
         
        lpx_list = []
        lpy_list = []
        lpe_list = []
        
        ithevs_jk, ithvec_jk, ithlvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 1)
        pmass_jk = TransferGEVP.EffectiveMassPlot(ithevs_jk)
        pmass_means, pmass_errs = TransferGEVP.MultiStatisticalAnal(pmass_jk, conf, 0)
        
        for i in range(conf):
            for j in range(len(pmass_means)):
                for k in range(len(pmass_means[j])):
                    lpx_list.append(2*(j+mmin)+1)
                    lpy_list.append(pmass_means[j][k])
                    lpe_list.append(pmass_errs[j][k])
        
        #=============================================================#
        #=============================================================#
        # Ground state
        
        gmass_jk = TransferGEVP.EffectiveMassPlot(gsevs_jk)
        gmass_means, gmass_errs = TransferGEVP.MultiStatisticalAnal(gmass_jk, conf, 0)
        
        lgx_list = []
        lgy_list = []
        lge_list = []
        
        for j in range(len(gmass_means)):
            for k in range(len(gmass_means[j])):
                lgx_list.append(2*(j+mmin)+1)
                lgy_list.append(gmass_means[j][k])
                lge_list.append(gmass_errs[j][k])
        
        print("Ground state right eigenvectors")
        for i in range(conf):
            for j in range(len(gsvec_jk[i])):
                print(j, gsvec_jk[i][j][0].size, gsvec_jk[i][j])
        
        #============================================================#
        #=============================================================#
        # File output
        
        gjkfile = resultspath + '/mass_jk'
        gavfile = resultspath + '/mass'
        
        gjk = open(gjkfile, 'w')
        for i in range(len(gmass_jk)):
            for j in range(len(gmass_jk[i])):
                strdata = str(j) + ' ' + str(gmass_jk[i][j][0]) + '\n'
                gjk.write(strdata)
        gjk.close()
        
        gav = open(gavfile, 'w')
        for j in range(len(gmass_means)):
            strdata = str(j) + ' ' + str(gmass_means[j][0]) + ' ' + str(gmass_errs[j][0]) + '\n'
            gav.write(strdata)
        gav.close()
        
        #=============================================================#
        #=============================================================#
        # Print eigenvectors for the target state
        
        tsIevs_jk, tsIvec_jk, tsIlvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, Istate)
        tsFevs_jk, tsFvec_jk, tsFlvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, Fstate)
        
        norm_jk = TransferGEVP.NormEnergyEigenvector(compevs_jk, compvec_jk, lcompvec_jk, exy_list, "both")
        InormR_jk = TransferGEVP.NormEnergyEigenvector(tsIevs_jk, tsIvec_jk, tsIlvec_jk, exy_list, "right")
        InormL_jk = TransferGEVP.NormEnergyEigenvector(tsIevs_jk, tsIvec_jk, tsIlvec_jk, exy_list, "left")
        FnormR_jk = TransferGEVP.NormEnergyEigenvector(tsFevs_jk, tsFvec_jk, tsFlvec_jk, exy_list, "right")
        FnormL_jk = TransferGEVP.NormEnergyEigenvector(tsFevs_jk, tsFvec_jk, tsFlvec_jk, exy_list, "left")
        
        vIRjkfile = resultspath + '/gstateI=' + str(Istate) + 'rank=' + str(svdrank) + 'm=' + str(m) + 'R_jk'
        vILjkfile = resultspath + '/gstateI=' + str(Istate) + 'rank=' + str(svdrank) + 'm=' + str(m) + 'L_jk'
        uIRjkfile = resultspath + '/geigenI=' + str(Istate) + 'rank=' + str(svdrank) + 'm=' + str(m) + 'R_jk'
        uILjkfile = resultspath + '/geigenI=' + str(Istate) + 'rank=' + str(svdrank) + 'm=' + str(m) + 'L_jk'
        
        vIRjk = open(vIRjkfile, 'w')
        vILjk = open(vILjkfile, 'w')
        uIRjk = open(uIRjkfile, 'w')
        uILjk = open(uILjkfile, 'w')
        for i in range(conf):
            for j in range(len(tsIvec_jk[i])):
                for l in range(len(tsIvec_jk[i][j][0])):
                    strRdata = str(j) + ' ' + str(tsIvec_jk[i][j][0][l].real) + ' ' + str(tsIvec_jk[i][j][0][l].imag) + '\n'
                    strLdata = str(j) + ' ' + str(tsIlvec_jk[i][j][0][l].real) + ' ' + str(tsIlvec_jk[i][j][0][l].imag) + '\n'
                    utrRdata = str(j) + ' ' + str(tsIvec_jk[i][j][0][l].real / InormR_jk[j][i][0]) + ' ' + str(tsIvec_jk[i][j][0][l].imag / InormR_jk[j][i][0]) + '\n'
                    utrLdata = str(j) + ' ' + str(tsIlvec_jk[i][j][0][l].real / InormL_jk[j][i][0]) + ' ' + str(tsIlvec_jk[i][j][0][l].imag / InormL_jk[j][i][0]) + '\n'
                    vIRjk.write(strRdata)
                    vIRjk.write(strLdata)
                    uIRjk.write(utrRdata)
                    uILjk.write(utrLdata)
        vIRjk.close()
        vILjk.close()
        uIRjk.close()
        uILjk.close()
        
        
        vFRjkfile = resultspath + '/gstateF=' + str(Fstate) + 'rank=' + str(svdrank) + 'm=' + str(m) +  'R_jk'
        vFLjkfile = resultspath + '/gstateF=' + str(Fstate) + 'rank=' + str(svdrank) + 'm=' + str(m) +  'L_jk'
        uFRjkfile = resultspath + '/geigenF=' + str(Fstate) + 'rank=' + str(svdrank) + 'm=' + str(m) +  'R_jk'
        uFLjkfile = resultspath + '/geigenF=' + str(Fstate) + 'rank=' + str(svdrank) + 'm=' + str(m) +  'L_jk'
        
        vFRjk = open(vFRjkfile, 'w')
        vFLjk = open(vFLjkfile, 'w')
        uFRjk = open(uFRjkfile, 'w')
        uFLjk = open(uFLjkfile, 'w')
        for i in range(conf):
            for j in range(len(tsFvec_jk[i])):
                for l in range(len(tsFvec_jk[i][j][0])):
                    strRdata = str(j) + ' ' + str(tsFvec_jk[i][j][0][l].real) + ' ' + str(tsFvec_jk[i][j][0][l].imag) + '\n'
                    strLdata = str(j) + ' ' + str(tsFlvec_jk[i][j][0][l].real) + ' ' + str(tsFlvec_jk[i][j][0][l].imag) + '\n'
                    utrRdata = str(j) + ' ' + str(tsFvec_jk[i][j][0][l].real / FnormR_jk[j][i][0]) + ' ' + str(tsFvec_jk[i][j][0][l].imag / FnormR_jk[j][i][0]) + '\n'
                    utrLdata = str(j) + ' ' + str(tsFlvec_jk[i][j][0][l].real / FnormL_jk[j][i][0]) + ' ' + str(tsFlvec_jk[i][j][0][l].imag / FnormL_jk[j][i][0]) + '\n'
                    vFRjk.write(strRdata)
                    vFRjk.write(strLdata)
                    uFRjk.write(utrRdata)
                    uFLjk.write(utrLdata)
        vFRjk.close()
        vFLjk.close()
        uFRjk.close()
        uFLjk.close()
        
        
        
        #=============================================================#
        #=============================================================#
        # Print effective mass and correlators
        
        for j in range(mmin, m):
            ms = j + 1
            # Num.0 state
            jthevs_jk, jthvec_jk, jthlvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 0)
            TransferGEVP.PrintEffectiveMass(jthevs_jk, 0, svdrank)
            TransferGEVP.PrintSingleEigenvalue(jthevs_jk, ms, 0, svdrank)
            fcorrelator_jth = TransferGEVP.MakeFullCorrelator(jthevs_jk, jthvec_jk, jthlvec_jk, exy_list, normalization_mean, correlator_time, t0)
            fzcw_jth = TransferGEVP.ZCW(jthevs_jk, jthvec_jk, jthlvec_jk, exy_list, normalization_mean, 1, t0)
            TransferGEVP.PrintSingleCorrelator(fcorrelator_jth, ms, 0, svdrank)
            TransferGEVP.PrintSingleAmplitude(fcorrelator_jth, ms, 0, svdrank)
            TransferGEVP.PrintSingleZCW(fzcw_jth, ms, 0, svdrank)
            
            # Num.1 state
            jthevs_jk, jthvec_jk, jthlvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 1)
            TransferGEVP.PrintEffectiveMass(jthevs_jk, 1, svdrank)
            fcorrelator_jth = TransferGEVP.MakeFullCorrelator(jthevs_jk, jthvec_jk, jthlvec_jk, exy_list, normalization_mean, correlator_time, t0)
            TransferGEVP.PrintSingleCorrelator(fcorrelator_jth, ms, 1, svdrank)
            
            # Num.2 state
            jthevs_jk, jthvec_jk, jthlvec_jk = TransferGEVP.PhysicalState(physevs_jk, physvec_jk, lphysvec_jk, 2)
            TransferGEVP.PrintEffectiveMass(jthevs_jk, 2, svdrank)
            fcorrelator_jth = TransferGEVP.MakeFullCorrelator(jthevs_jk, jthvec_jk, jthlvec_jk, exy_list, normalization_mean, correlator_time, t0)
            TransferGEVP.PrintSingleCorrelator(fcorrelator_jth, ms, 2, svdrank)


#=============================================================#
# from command line

if __name__ == "__main__":

    import sys

    main(
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        sys.argv[5],
        sys.argv[6],
        int(sys.argv[7]),
        int(sys.argv[8]),
        sys.argv[9],
        int(sys.argv[10]),
        int(sys.argv[11]),
    )

import numpy as np
import scipy as sp
import statistics
import matplotlib.pyplot as plt
import matplotlib.colors as cls

import copy
import math
import cmath


#=============================================================#
#=============================================================#
# Statistics

def JackknifeAnal(result, n):
    ave = np.sum(np.array(result)) / n
    var = 0
    for r in result:
        var += (1. - 1. / n) * ((ave -r)**2)
    err = np.sqrt(var)
    return ave, err

def BootstrapAnal(result, n):
    ave = np.sum(np.array(result)) / n
    var = 0
    for r in result:
        var += 1. / n * ((ave -r)**2)
    err = np.sqrt(var)
    return ave, err

def StatisticalAnal(result, n, state):

    means = []
    for j in range(len(result[0])):
        via = np.array(result[0][j][state])
        for i in range(1, n):
            via = via + np.array(result[i][j][state])
        via = via / n
        print(via)
        print(result[i][j][state])
        via_list = []
        via_list.append(via)
        means.append((via_list))

    errs = []
    for j in range(len(result[0])):
        via = []
        var = 0
        for i in range(n):
            var = var + 1. / n * ((result[i][j][state] - means[j][0])**2)
        via.append(np.sqrt(var))
        errs.append((via))

    return means, errs

def MultiMedianConfidenceIntAnal(result, n, state):

    # 1-sigma confidence interval
    confidence_factor = 0.68
    q = 1. - (1. - confidence_factor) / 2.
    t_critical = sp.stats.t.ppf(q=q, df = n-1)

    medians = []
    cis = []
    for j in range(len(result[0])):
        dist = []
        for i in range(n):
            dist.append(result[i][j][state])
        median = statistics.median(dist)
        stdev = np.array(dist).std(ddof=1)
        err = t_critical*(stdev/np.sqrt(n))
        bottom = np.percentile(dist, 16)
        up = np.percentile(dist, 100-16) - median
        via_median = []
        via_median.append(median)
        medians.append(via_median)
        via_cis = []
        via_cis.append(err)
        cis.append(via_cis)

    return medians, cis


def MultiStatisticalAnal(result, n, state):

    means = []
    for j in range(len(result[0])):
        via = np.array(result[0][j][state])
        for i in range(1, n):
            via = via + np.array(result[i][j][state])
        via = via / n
        print(via)
        print(result[i][j][state])
        via_list = []
        via_list.append(via)
        means.append((via_list))

    errs = []
    for j in range(len(result[0])):
        via = []
        var = 0
        for i in range(n):
            var = var + 1. / n * ((result[i][j][state] - means[j][0])**2)
        via.append(np.sqrt(var))
        errs.append((via))

    return means, errs

#=============================================================#
#=============================================================#
# Pick up

def RealOrComp_eigenvalue(eigenvalue, eigenvector, leigenvector):
    tole = 10**(-15)
    realev = []
    compev = []

    realve = []
    compve = []

    lcompve = []

    for k in range(len(eigenvalue)):
        compev.append(eigenvalue[k])
        compve.append(eigenvector[:,k])
        lcompve.append(leigenvector[:,k])
        if (np.abs(cmath.phase(eigenvalue[k])) < tole):
            realev.append(eigenvalue[k].real)
            realve.append(eigenvector[:,k])

    return realev, compev, realve, compve, lcompve


def SortedList(target_list):

    t_list = copy.deepcopy(target_list)

    indices = []
    for k in range(len(t_list)):
        indices.append(k)

    # find nan
    for k in range(len(t_list)):
        if math.isnan(t_list[k]):
            nan = t_list.pop(k)
            t_list.append(nan)
            num = indices.pop(k)
            indices.append(num)

    # t_list.sort(reverse=True)
    for k in range(len(t_list)):
        for l in range(k+1, len(t_list)):
            if t_list[k] <= t_list[l]:
                t_list[k], t_list[l] = t_list[l], t_list[k]
                indices[k], indices[l] = indices[l], indices[k]

    if (t_list[0] >= 1):
        num = 0
        for k in range(1, len(t_list)):
            if (t_list[k] >= 1):
                num = num + 1
        sliced_latter = t_list[:num+1]
        sliced_former = t_list[num+1:]

        indices_latter = indices[:num+1]
        indices_former = indices[num+1:]

        via = sliced_former + sliced_latter
        via_i = indices_former + indices_latter
    else:
        via = t_list
        via_i = indices

    return via, via_i


def Mean_eigenvalueByState(eigenvalue, state):
    means = []

    conf = len(eigenvalue)
    for j in range(len(eigenvalue[0])):
        via = np.array(eigenvalue[0][j])
        for i in range(1, conf):
            via = via + np.array(eigenvalue[i][j][state])
        via = via / conf
        means.append(list(via))

    ev = copy.deepcopy(eigenvalue)

    test = []
    for j in range(len(ev[0])):
        via = []
        mean = 0
        for i in range(len(ev)):
            sorted_list, indices = SortedList(ev[i][j])
            mean = mean + sorted_list[state]
        mean = mean / conf
        via.append(mean)
        test.append(via)

    for j in range(len(test)):
        print("means", test[j])

    err = []
    for j in range(len(ev[0])):
        via = []
        var = 0
        for i in range(len(ev)):
            sorted_list, indices = SortedList(ev[i][j])
            var = var + (sorted_list[state] - test[j][state])**2
        var = np.sqrt(var / conf)
        via.append(var)
        err.append(via)

    print(test)
    print(err)

    return test, err


def Real_eigenvalueByConf(eigenvalue, tole):
    realev_jk = []
    rindices = []

    ev = copy.deepcopy(eigenvalue)
    conf = len(ev)
    for i in range(conf):
        via = []
        ivia = []
        for j in range(len(ev[i])):
            cand_via = []
            rind_via = []
            sorted_list, indices = SortedList(ev[i][j])
            for k in range(len(ev[i][j])):
                # if (np.abs(cmath.phase(ev[i][j][k])) < tole): # imaginaries is sorted
                if (np.abs(cmath.phase(sorted_list[k])) < tole): # imaginaries is sorted
                    cand_via.append(ev[i][j][indices[k]].real)
                    rind_via.append(indices[k])
            via.append(cand_via)
            ivia.append(rind_via)
        realev_jk.append(via)
        rindices.append(ivia)

    return realev_jk, rindices



def Physical_eigenvalueByConf(eigenvalue, tole, rindices):
    physevs_jk = []
    pindices = []

    ev = copy.deepcopy(eigenvalue)
    conf = len(ev)
    for i in range(conf):
        via = []
        pvia = []
        ivia = []
        for j in range(len(ev[i])):
            cand_via = []
            dcand_via = []
            icand_via = []
            for k in range(len(ev[i][j])):
                if (ev[i][j][k] <= 1):
                    cand = (ev[i][j][k].real)
                    cand_via.append(cand)
                    icand_via.append( rindices[i][j][k] )
            via.append(cand_via)
            pvia.append(dcand_via)
            ivia.append(icand_via)
        physevs_jk.append(via)
        pindices.append(ivia)

    return physevs_jk, pindices


def PhysicalState(eigenvalue, eigenvector, leigenvector, state):

    conf = len(eigenvalue)

    ithevs_jk = []  # ith state
    ithvec_jk = []  # ith state right eigenvector
    ithlvec_jk = [] # ith state left eigenvector

    for i in range(conf):
        via = []
        vvia = []
        lvvia = []
        for j in range(len(eigenvalue[i])):
            cand_via = []
            cand_vvia = []
            cand_lvvia = []

            index = 0
            # get ith value
            np_eigenvalue = np.array(eigenvalue[i][j])
            sorted_eigenvalue = np.sort(np_eigenvalue)[::-1]
            sorted_index = np.argsort(np_eigenvalue)[::-1]
            if state < len(sorted_index):
                index = sorted_index[state]
            else:
                index = sorted_index[-1]
            print("sindex", index)

            cand_via.append(eigenvalue[i][j][index])
            cand_vvia.append(eigenvector[i][j][index])
            cand_lvvia.append(leigenvector[i][j][index])
            via.append(cand_via)
            vvia.append(cand_vvia)
            lvvia.append(cand_lvvia)
        ithevs_jk.append(via)
        ithvec_jk.append(vvia)
        ithlvec_jk.append(lvvia)


    return ithevs_jk, ithvec_jk, ithlvec_jk


def GroundState(eigenvalue, eigenvector, leigenvector, mass_mean):

    conf = len(eigenvalue)

    gsevs_jk = [] # ground state
    gsvec_jk = []  # ground state right eigenvector
    gslvec_jk = [] # ground state left eigenvector

    for i in range(conf):
        via = []
        vvia = []
        lvvia = []
        for j in range(len(eigenvalue[i])):
            cand_via = []
            cand_vvia = []
            cand_lvvia = []
            diff = np.abs(-np.log(eigenvalue[i][j][0]) - mass_mean)
            index = 0
            for k in range(1,len(eigenvalue[i][j])):
                cdiff = np.abs(-np.log(eigenvalue[i][j][k]) - mass_mean)
                if cdiff < diff:
                    index = k
                    diff = cdiff
            cand_via.append(eigenvalue[i][j][index])
            cand_vvia.append(eigenvector[i][j][index])
            cand_lvvia.append(leigenvector[i][j][index])
            via.append(cand_via)
            vvia.append(cand_vvia)
            lvvia.append(cand_lvvia)
        gsevs_jk.append(via)
        gsvec_jk.append(vvia)
        gslvec_jk.append(lvvia)


    return gsevs_jk, gsvec_jk, gslvec_jk

def EffectiveMassPlot(eigenvalue):

    effective_mass = []

    for i in range(len(eigenvalue)):
        via = []
        for j in range(len(eigenvalue[i])):
            cand_via = []
            for k in range(len(eigenvalue[i][j])):
                value = -np.log(eigenvalue[i][j][k])
                cand_via.append(value)
                # if (math.isnan(value)):
                #     pass
                # else:
                #     cand_via.append(value)
            via.append(cand_via)
        effective_mass.append(via)

    return effective_mass

def NormEnergyEigenvector(compevs_jk, compvec_jk, lcompvec_jk, exy_list, leftright):
    conf = len(compevs_jk)

    # normalization for the enregy eigenvector
    norm_jk = []
    for j in range(len(compevs_jk[0])):
        norm_i = []
        for i in range(conf):
            norm_k = []
            for k in range(len(compevs_jk[i][j])):
                normalization = 0
                for mi in range(len(compvec_jk[i][j][k])):
                    for mj in range(len(lcompvec_jk[i][j][k])):
                        if leftright == "both":
                            normalization += lcompvec_jk[i][j][k][mj] * compvec_jk[i][j][k][mi] * exy_list[i][mi+mj]
                        elif leftright == "left":
                            normalization += lcompvec_jk[i][j][k][mj] * lcompvec_jk[i][j][k][mi].conjugate() * exy_list[i][mi+mj]
                        elif leftright == "right":
                            normalization += compvec_jk[i][j][k][mj].conjugate() * compvec_jk[i][j][k][mi] * exy_list[i][mi+mj]
                norm_k.append(np.sqrt(normalization).real)
            norm_i.append(norm_k)
        norm_jk.append(norm_i)
    print(norm_jk)

    return norm_jk


def ReconstCorrelator(correlator, state):

    correlator_data = []

    for i in range(len(correlator)):
        via = []
        for k in range(len(correlator[i][state])):
            value = (correlator[i][state][k])
            via.append(value)
        correlator_data.append(via)

    return correlator_data 



#=============================================================#
#=============================================================#
# Transfer GEVP

def construct_TV(exy_list, m, shift):
    T = np.zeros((m+1,m+1))
    V = np.zeros((m+1,m+1))
    N = np.zeros((m+1,m+1))
    print(m, exy_list[1])

    # matrix T
    for i in range(m+1):
        for j in range(m+1):
            # N[i,j] = 1.0 / np.sqrt(exy_list[2*i]*exy_list[2*j])
            N[i,j] = 1.0 / exy_list[0]


    # matrix T
    for i in range(m+1):
        for j in range(m+1):
            # T[i,j] = exy_list[i+j+1] * N[i,j]
            # T[i,j] = exy_list[i+j+2] * N[i,j]
            T[i,j] = exy_list[i+j+1+shift] * N[i,j]

    # matrix V
    for i in range(m+1):
        for j in range(m+1):
            V[i,j] = exy_list[i+j] * N[i,j]

    return T, V


def construct_RegularizedTV(exy_list, m):
    T = np.zeros((m+1,m+1))
    V = np.zeros((m+1,m+1))
    N = np.zeros((m+1,m+1))
    print(m, exy_list[1])

    epsilon = 0.01

    # matrix T
    for i in range(m+1):
        for j in range(m+1):
            # N[i,j] = 1.0 / np.sqrt(exy_list[2*i]*exy_list[2*j])
            N[i,j] = 1.0 / exy_list[0]


    # matrix T
    for i in range(m+1):
        for j in range(m+1):
            T[i,j] = exy_list[i+j+1] * N[i,j]
            if i == j:
                T[i,j] = T[i,j] + epsilon

    # matrix V
    for i in range(m+1):
        for j in range(m+1):
            V[i,j] = exy_list[i+j] * N[i,j]
            if i == j:
                T[i,j] = T[i,j] + epsilon

    return T, V


def RMTThresholdRank(Y):
    # find a optimal rank based on Randam Matrix Theory (RMT) based technique

    m, n = Y.shape
    beta = m / n if m <= n else n / m

    # estimate noise level
    s_all = np.linalg.svd(Y, compute_uv=False)
    median_s = np.median(s_all)
    sigma = median_s / (np.sqrt(n)+np.sqrt(m))
    print("RMTsall",s_all)
    print("RMTsigma",sigma)

    # Marchenko-Pastur upper edge (bulk noise)
    lambda_plus = sigma * (1 + np.sqrt(beta))**2

    # compute full svd
    U, s, Vh = sp.linalg.svd(Y, full_matrices=False)

    # keep only singular values above lambda_plus
    mask = s > lambda_plus
    rank_est = np.sum(mask)

    return rank_est



def SingularValueDecompositionGEVP(T,V,svdrank):
    # SVD for T
    U, S_diag, V_t = sp.linalg.svd(T)
    size = len(S_diag)
    N_diag = S_diag / S_diag[0]
    S_list = list(N_diag)
    rank = svdrank

    # reduction
    U_r = U[:, :rank]
    S_diag_r = np.diag(S_diag[:rank])
    V_r = V_t[:rank, :].T

    # Reduced matrix
    T_r = S_diag_r
    W_r = U_r.T @ V @ V_r

    # GEVP
    eigenvalues, eigenvectors_r = sp.linalg.eig(T_r,W_r)
    leigenvalues, leigenvectors_r = sp.linalg.eig(T_r.conj().T, W_r.conj().T)

    eigenvectors = V_r @ eigenvectors_r
    leigenvectors = U_r @ leigenvectors_r
    print("Eigenvec", eigenvectors_r)

    return eigenvalues, leigenvectors, eigenvectors, S_list



def FigSingularValues(singularvalues_jk, m):

    # Figure
    hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
    jalabel = "j=" + str(m)
    ax = []
    ay = []
    for i in range(len(singularvalues_jk)):
        for r in range(len(singularvalues_jk[i][0])):
            ax.append(r)
            ay.append(singularvalues_jk[i][0][r])
    frun = PrintSingularValues(ax, ay, m)
    # plt.title('Singular values all one Gauge configurations')
    # plt.xlabel('$r$', fontsize=15)
    # plt.ylabel('$\\sigma^{(j)}_r$', fontsize=15)
    # plt.ylim(1e-18, 100)
    # plt.scatter(ax,ay, s=5, color=hcolors[0], label=jalabel)
    # plt.legend(fontsize=18)
    # plt.yscale('log')
    # plt.show()
    return 0



def ZCW(compevs_jk, compvec_jk, lcompvec_jk,exy_list, normalization_mean, time, t0):
    eigenvalues = []
    conf = len(compevs_jk)

    # normalization for the enregy eigenvector
    normR_jk = NormEnergyEigenvector(compevs_jk, compvec_jk, lcompvec_jk, exy_list, "right")
    normL_jk = NormEnergyEigenvector(compevs_jk, compvec_jk, lcompvec_jk, exy_list, "left")

    for j in range(len(compevs_jk[0])):
        via_j = []
        via_js = []
        for i in range(conf):
            via_jis = []
            for k in range(len(compevs_jk[i][j])):
                # for all
                via_jik = []
                for t in range(time):
                    rAmp = 0
                    lAmp = 0
                    for m in range(len(compvec_jk[i][j][k])):
                        rAmp += compvec_jk[i][j][k][m] * exy_list[i][m] / normR_jk[j][i][k]
                        lAmp += lcompvec_jk[i][j][k][m] * exy_list[i][m] / normL_jk[j][i][k]
                    value = rAmp * lAmp / exy_list[i][0] 
                    via_jik.append(value)
                via_jis.append(via_jik)
            via_js.append(via_jis)
        eigenvalues.append(via_js)

    return eigenvalues
    

def MakeFullCorrelator(compevs_jk, compvec_jk, lcompvec_jk,exy_list, normalization_mean, time, t0):
    eigenvalues = []
    conf = len(compevs_jk)

    # normalization for the enregy eigenvector
    normR_jk = NormEnergyEigenvector(compevs_jk, compvec_jk, lcompvec_jk, exy_list, "right")
    normL_jk = NormEnergyEigenvector(compevs_jk, compvec_jk, lcompvec_jk, exy_list, "left")

    for j in range(len(compevs_jk[0])):
        via_j = []
        via_js = []
        for i in range(conf):
            via_jis = []
            for k in range(len(compevs_jk[i][j])):
                # for all
                via_jik = []
                for t in range(time):
                    rAmp = 0
                    lAmp = 0
                    for m in range(len(compvec_jk[i][j][k])):
                        rAmp += compvec_jk[i][j][k][m] * exy_list[i][m] / normR_jk[j][i][k]
                        lAmp += lcompvec_jk[i][j][k][m] * exy_list[i][m] / normL_jk[j][i][k]
                    print("Amp", rAmp, lAmp)
                    value = compevs_jk[i][j][k]**(t-2*t0) * rAmp * lAmp * normalization_mean
                    via_jik.append(value)
                via_jis.append(via_jik)
            via_js.append(via_jis)
        eigenvalues.append(via_js)

    return eigenvalues
    
    
def FigCorrelators(correlator_all, correlator_gs):

    # Figure
    hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
    for j in range(len(correlator_all)):
        malabel = "m=" + str(j) + ": All eigenvalues"
        mglabel = "m=" + str(j) + ": Ground state"
        ax = []
        ay = []
        gx = []
        gy = []
        for i in range(len(correlator_all[j])):
            for k in range(len(correlator_all[j][i])):
                for t in range(len(correlator_all[j][i][k])):
                    ax.append(t)
                    ay.append(correlator_all[j][i][k][t])
            for k in range(len(correlator_gs[j][i])):
                for t in range(len(correlator_gs[j][i][k])):
                    gx.append(t)
                    gy.append(correlator_gs[j][i][k][t])

    for j in range(len(correlator_all)):
        jalabel = "j=" + str(j) + ": All eigenvalues"
        jglabel = "j=" + str(j) + ": Ground state"
        ax = []
        ay = []
        gx = []
        gy = []
        for i in range(1):
            for k in range(len(correlator_all[j][i])):
                for t in range(len(correlator_all[j][i][k])):
                    ax.append(t)
                    ay.append(correlator_all[j][i][k][t])
            for k in range(len(correlator_gs[j][i])):
                for t in range(len(correlator_gs[j][i][k])):
                    gx.append(t)
                    gy.append(correlator_gs[j][i][k][t])

    return 0



def GetPhysical(compevs_jk, compvec_jk, lcompvec_jk, tole):

    physevs_jk = []

    #====#
    # N_it   : Number of the iteration = size of the materix
    # N_boot : Number of the bootstrap sample
    #====#
    N_it = int(len(compevs_jk[0]))
    N_boot = int(len(compevs_jk))
    conf = len(compevs_jk)

    #====#
    # discard the spurious eigenvalues 
    # 1. if complex
    # 2. if lambda > 1
    # 
    # 1. is implemented in Real, while 2. is in Physical
    #====#
    print("Data compevs_jk")
    for i in range(len(compevs_jk)):
        for j in range(len(compevs_jk[i])):
            print(i, j, compevs_jk[i][j])


    realevs_jk, rindices = Real_eigenvalueByConf(compevs_jk, tole)

    print("Data realevs_jk")
    for i in range(len(realevs_jk)):
        for j in range(len(realevs_jk[i])):
            print(i, j, (realevs_jk[i][j]))

    print("Real indices")
    for i in range(len(rindices)):
        for j in range(len(rindices[i])):
            print(i, j, rindices[i][j])

    physevs_jk, pindices = Physical_eigenvalueByConf(realevs_jk, tole, rindices)

    print("Physical indices")
    for i in range(len(pindices)):
        for j in range(len(pindices[i])):
            print(i, j, compevs_jk[i][j][pindices[i][j][0]], pindices[i][j])

    print("Data physevs_jk")
    for i in range(len(physevs_jk)):
        for j in range(len(physevs_jk[i])):
            print(i, j, physevs_jk[i][j])


    print("Data physevs_jk")
    for i in range(len(physevs_jk)):
        for j in range(len(physevs_jk[i])):
            cflag = "none"
            if ( np.abs(compevs_jk[i][j][pindices[i][j][0]] - physevs_jk[i][j][0]) > 10**(15) ):
                cflag = "HIT!"
            print(i, j, compevs_jk[i][j][pindices[i][j][0]], physevs_jk[i][j][0], cflag)
            cflag = "none"

    print("Phys vec")
    physvec_jk = []
    lphysvec_jk = []
    for i in range(len(pindices)):
        via = []
        lvia = []
        for j in range(len(pindices[i])):
            cand_via = []
            cand_lvia = []
            for k in range(len(pindices[i][j])):
                cand_via.append(compvec_jk[i][j][pindices[i][j][k]])
                cand_lvia.append(lcompvec_jk[i][j][pindices[i][j][k]])
                for l in range(len(compvec_jk[i][j][pindices[i][j][k]])):
                    if (compvec_jk[i][j][pindices[i][j][k]][l].imag > tole): # means is sorted
                        print(i, j, compvec_jk[i][j][pindices[i][j][k]][l], "FAIL")
            via.append(cand_via)
            lvia.append(cand_lvia)
        physvec_jk.append(via)
        lphysvec_jk.append(lvia)


    print("physevs_jk")
    pmeans, perrs = Mean_eigenvalueByState(physevs_jk, 0)

    print("phys eff in func")
    for j in range(len(pmeans)):
        for k in range(len(pmeans[j])):
            print(j, k, -np.log(pmeans[j][k]))
    print("\n")


    return physevs_jk, physvec_jk, lphysvec_jk


def EigenvalueVariance_same(exy_list, ithevs_jk, ithvec_jk, ithlvec_jk):
    conf = len(ithevs_jk)

    # normalization for the enregy eigenvector
    norm_jk = NormEnergyEigenvector(ithevs_jk, ithvec_jk, ithlvec_jk, exy_list, "both")
    normR_jk = NormEnergyEigenvector(ithevs_jk, ithvec_jk, ithlvec_jk, exy_list, "both")
    normL_jk = NormEnergyEigenvector(ithevs_jk, ithvec_jk, ithlvec_jk, exy_list, "both")

    ithsqave_jk = []
    for b in range(conf):
        via2 = []
        for r in range(len(ithevs_jk[b])): # each rank
            cand_via2 = []
            for k in range(len(ithevs_jk[b][r])): # single state, only k=0
                value2 = 0
                for i in range(len(ithvec_jk[b][r][k])):
                    for j in range(len(ithlvec_jk[b][r][k])):
                        value2 += ithlvec_jk[b][r][k][j] * ithvec_jk[b][r][k][i] * exy_list[b][i+j+2] / (normL_jk[r][b][k]*normR_jk[r][b][k])
                cand_via2.append(value2.real)
            via2.append(cand_via2)
        ithsqave_jk.append(via2)

    ithave_jk = []
    for b in range(conf):
        via = []
        for r in range(len(ithevs_jk[b])): # each rank
            cand_via = []
            for k in range(len(ithevs_jk[b][r])): # single state, only k=0
                value = 0
                for i in range(len(ithvec_jk[b][r][k])):
                    for j in range(len(ithlvec_jk[b][r][k])):
                        value += ithlvec_jk[b][r][k][i] * ithvec_jk[b][r][k][j] * exy_list[b][i+j+1] / (normL_jk[r][b][k]*normR_jk[r][b][k])
                cand_via.append(value.real)
            via.append(cand_via)
        ithave_jk.append(via)

    ithvar_jk = []
    for b in range(conf):
        via3 = []
        for r in range(len(ithave_jk[b])): # each rank
            cand_via3 = []
            for k in range(len(ithave_jk[b][r])): # single state, only k = 0
                value = ithsqave_jk[b][r][k] - ithave_jk[b][r][k]**2
                cand_via3.append(value)
            via3.append(cand_via3)
        ithvar_jk.append(via3)

    return ithvar_jk

#=============================================================#
#=============================================================#
# misc

def PrintJKFile(data_list, name):
    djkfile = datapath + '/' + name + '_jk'
    d = open(djkfile, 'w')

    for i in range(len(data_list)):
        for j in range(len(data_list[0])):
            strdata = str(j) + ' ' + str(data_list[i][j]) + '\n'
            d.write(strdata)
    d.close

    return 0

def PrintHist(data_list, name, j):
    dhistfile = './eigenvalueshist_data/'+ name + 'Hist-' + str(j) + '_jk'
    d = open(dhistfile, 'w')

    for i in range(len(data_list)):
        strdata = str(data_list[i]) + '\n'
        d.write(strdata)
    d.close

    return 0


def PrintSingularValues(ax, ay, j):
    dsvfile = './singularvalues_data/Err-' + str(j) + '_jk'
    # dsvfile = './singularvalues_data/Err-' + str(j+1) + '_jk'
    d = open(dsvfile, 'w')

    for i in range(len(ax)):
        strdata = str(ax[i]) + ' ' + str(ay[i]) + '\n'
        d.write(strdata)
    d.close

    return 0


def PrintEffectiveMass(datalist, state, svdrank):
    davmfile = './diagonalized_data/mass' + str(state) + 'rank=' + str(svdrank)
    dav = open(davmfile, 'w')

    conf = len(datalist)
    mass_jk = EffectiveMassPlot(datalist)
    mass_means, mass_errs = MultiStatisticalAnal(mass_jk, conf, 0)

    for j in range(len(mass_means)):
        strdata = str(2*j+1) + ' ' + str(mass_means[j][0]) + ' ' + str(mass_errs[j][0]) + '\n'
        dav.write(strdata)
    dav.close()

    return 0

def PrintCorrelator(datalist, m, state):
    djkmfile = './diagonalized_data/corr' + str(state) + '_jk'
    davmfile = './diagonalized_data/corr' + str(state)
    djk = open(djkmfile, 'w')
    dav = open(davmfile, 'w')

    correlator_gs = datalist[m]
    conf = len(correlator_gs)
    corr_jk = ReconstCorrelator(correlator_gs, state) 

    corr_t = np.array(corr_jk).T
    for j in range(len(corr_t)):
        ave, err = BootstrapAnal(corr_t[j], conf)
        strdata = str(j) + ' ' + str(ave) + ' ' + str(err) + '\n'
        dav.write(strdata)
    djk.close()
    dav.close()

    return 0

def PrintEigenvalueVariance(means, errs, state):
    davmfile = './diagonalized_data/var' + str(state)
    dav = open(davmfile, 'w')

    for i in range(len(means)):
        strdata = str(i+1) + ' ' + str(means[i][0]) + ' ' + str(errs[i][0]) + '\n'
        dav.write(strdata)
    dav.close()

    return 0


def PrintSingleCorrelator(datalist, m, state, svdrank):
    djkmfile = './diagonalized_data/corr' + str(state) + 'rank=' + str(svdrank) + 'm=' + str(m) + '_jk'
    davmfile = './diagonalized_data/corr' + str(state) + 'rank=' + str(svdrank) + 'm=' + str(m)
    djk = open(djkmfile, 'w')
    dav = open(davmfile, 'w')

    correlator_gs = datalist[-1]
    # correlator_gs = datalist[m]
    conf = len(correlator_gs)
    corr_jk = ReconstCorrelator(correlator_gs, 0) 

    corr_t = np.array(corr_jk).T
    for j in range(len(corr_t)):
        ave, err = BootstrapAnal(corr_t[j], conf)
        strdata = str(j) + ' ' + str(ave) + ' ' + str(err) + '\n'
        dav.write(strdata)
    dav.close()

    for b in range(conf):
        for j in range(len(corr_jk[b])):
            strbdata = str(j) + ' ' + str(corr_jk[b][j]) + '\n'
            djk.write(strbdata)
    djk.close()

    return 0

def PrintSingleAmplitude(datalist, m, state, svdrank):
    djkmfile = './diagonalized_data/amp' + str(state) + 'rank=' + str(svdrank) + 'm=' + str(m) + '_jk'
    davmfile = './diagonalized_data/amp' + str(state) + 'rank=' + str(svdrank) + 'm=' + str(m)
    djk = open(djkmfile, 'w')
    dav = open(davmfile, 'w')

    correlator_gs = datalist[-1]
    # correlator_gs = datalist[m]
    conf = len(correlator_gs)
    corr_jk = ReconstCorrelator(correlator_gs, 0) 

    corr_t = np.array(corr_jk).T
    for j in range(1):
        ave, err = BootstrapAnal(corr_t[j], conf)
        strdata = str(j) + ' ' + str(ave) + ' ' + str(err) + '\n'
        dav.write(strdata)
    dav.close()

    for b in range(conf):
        for j in range(1):
            strbdata = str(j) + ' ' + str(corr_jk[b][j]) + '\n'
            djk.write(strbdata)
    djk.close()

    return 0


def PrintSingleZCW(datalist, m, state, svdrank):
    djkmfile = './diagonalized_data/zcw' + str(state) + 'rank=' + str(svdrank) + 'm=' + str(m) + '_jk'
    davmfile = './diagonalized_data/zcw' + str(state) + 'rank=' + str(svdrank) + 'm=' + str(m)
    djk = open(djkmfile, 'w')
    dav = open(davmfile, 'w')

    zcw_gs = datalist[-1]
    print("ZCW")
    conf = len(zcw_gs)
    zcw_jk = []
    for i in range(conf):
        for j in range(len(zcw_gs[i])):
            for k in range(len(zcw_gs[i][j])):
                zcw_jk.append(zcw_gs[i][j][k])
    print(zcw_jk)

    for j in range(1):
        ave, err = BootstrapAnal(zcw_jk, conf)
        strdata = str(j) + ' ' + str(ave) + ' ' + str(err) + '\n'
        dav.write(strdata)
    dav.close()

    for b in range(conf):
        strbdata = str(j) + ' ' + str(zcw_jk[b]) + '\n'
        djk.write(strbdata)
    djk.close()

    return 0


def PrintSingleEigenvalue(datalist, m, state, svdrank):
    djkmfile = './diagonalized_data/ev' + str(state) + 'rank=' + str(svdrank) + 'm=' + str(m) + '_jk'
    davmfile = './diagonalized_data/ev' + str(state) + 'rank=' + str(svdrank) + 'm=' + str(m)
    djk = open(djkmfile, 'w')
    dav = open(davmfile, 'w')

    conf = len(datalist)
    ev_jk = []
    for b in range(conf):
        value = datalist[b][-1][state]
        # value = datalist[b][m][state]
        ev_jk.append(value)

    ave, err = BootstrapAnal(ev_jk, conf)
    strdata = str(2*(m)+1) + ' ' + str(ave) + ' ' + str(err) + '\n'
    dav.write(strdata)
    dav.close()

    for b in range(conf):
        strbdata = str(b) + ' ' + str(ev_jk[b]) + '\n'
        djk.write(strbdata)
    djk.close()

    return 0


def PrintSingleEigenvalueVariance(datalist, m, state):

    conf = len(datalist)
    maxrank = len(datalist[0])
    for r in range(maxrank):
        djkmfile = './diagonalized_data/var' + str(state) + 'rank=' + str(r+1) + 'm=' + str(m) + '_jk'
        davmfile = './diagonalized_data/var' + str(state) + 'rank=' + str(r+1) + 'm=' + str(m)
        djk = open(djkmfile, 'w')
        dav = open(davmfile, 'w')

        varrth_jk = []
        for b in range(conf):
            value = datalist[b][r][0]
            varrth_jk.append(value)

        ave, err = BootstrapAnal(varrth_jk, conf)
        strdata = str(2*(m)+1) + ' ' + str(ave) + ' ' + str(err) + '\n'
        dav.write(strdata)
        dav.close()

        for b in range(conf):
            strbdata = str(b) + ' ' + str(varrth_jk[b]) + '\n'
            djk.write(strbdata)
        djk.close()

    return 0


#=============================================================#
#=============================================================#
# Read the eigenvector

def GetSingleRankEigenvector(dirname, m, IF, state, rank):
    lin = 0
    
    Rgstatejkfile = dirname + '/geigen' + IF + '=' + str(state) + 'rank=' + str(rank) + 'm=' + str(m) + 'R_jk'
    Rgstatejk = open(Rgstatejkfile,'rt');
    
    Rgstate_jk = []
    
    via = []
    via_j = []
    for string in Rgstatejk:
        data = string[:-1].split(' ')
        cvalue = complex(float(data[1]), float(data[2]))
        via_j.append(cvalue)
        if (lin % m == m - 1):
            via.append(np.array(via_j))
            Rgstate_jk.append(via)
            via = []
            via_j = []
        lin = lin + 1
    lin = 0 
    
    
    # Left
    Lgstatejkfile = dirname + '/geigen' + IF + '=' + str(state) + 'rank=' + str(rank) + 'm=' + str(m) +  'L_jk'
    Lgstatejk = open(Lgstatejkfile,'rt');
    
    Lgstate_jk = []
    
    via = []
    via_j = []
    for string in Lgstatejk:
        data = string[:-1].split(' ')
        cvalue = complex(float(data[1]), float(data[2]))
        via_j.append(cvalue)
        if (numconf[m-1] - 1 == j):
            via.append(np.array(via_j))
            Lgstate_jk.append(via)
            via = []
            via_j = []
        lin = lin + 1
    lin = 0 
    
    for i in range(len(Lgstate_jk)):
        for j in range(len(Lgstate_jk[i])):
            print(i, j, Lgstate_jk[i][j])

    return Rgstate_jk, Lgstate_jk
    
    


#=============================================================#
#=============================================================#
# Hadron matrix elements

def TwoPtCt(cdatafile, T):
    Ct_jk = []

    cjkfile = './correlator_data/' + cdatafile + '_jk'
    cjk = open(cjkfile, 'rt')

    lin = 0
    via = []
    for string in cjk:
        data = string[:-1].split(' ')
        via.append(float(data[1]))
        if (lin%T) == (T-1):
            Ct_jk.append(via)
            via = []
        lin = lin + 1

    return Ct_jk

def HeatMap(data, m):
    datasize = len(data)

    thrpt_list = [ [0 for j in range(datasize)] for i in range(datasize) ]
    for tsep in range(datasize):
        for tau in range(tsep+1):
            sigma = tsep - tau
            thrpt_list[tau][sigma] = data[tsep][0][tau]
    print(thrpt_list)

    return thrpt_list


def HadronMatrixElements(exy_list, iCt_jk, fCt_jk, IRgstate_jk, FLgstate_jk, nt, m, kinematic_factor, t0):
    J_jk = []

    conf = len(exy_list[1])

    print("conf in GraoundstateMatrixElements function:", conf)
    print("check with conf.0")
    HeatMap(exy_list, m)
    for j in range(m):
        for tau in range(j+1):
            for sigma in range(j+1):
                tsep = tau + sigma
                print(j, tau, sigma, exy_list[tsep][0][tau])

    # Groundstate Matrix Elements
    for i in range(conf):
        ivia = []
        for j in range(m):
            value = 0
            for tau in range(j+1):
                for sigma in range(j+1):
                    tsep = tau + sigma + t0 + t0
                    value += FLgstate_jk[i][j][sigma] * exy_list[tsep][i][tau+t0] * IRgstate_jk[i][j][tau] / np.sqrt(iCt_jk[i][2*t0]*fCt_jk[i][2*t0])
            value = value / kinematic_factor
            ivia.append(value)
        J_jk.append(ivia)


    return J_jk

#=============================================================#
#=============================================================#

def RealOrthonormalVector(v):
    v = np.asarray(v, dtype=float)
    if np.linalg.norm(v) == 0:
        raise ValueError("Zero vector has no well-defined orthonormal vector.")

    v = v / np.linalg.norm(v)

    # Pick a vector not parallel to v
    for i in range(len(v)):
        if abs(v[i]) < 0.9:
            w = np.zeros_like(v)
            w[i] = 1.0
            break

    # Gram–Schmidt step
    u = w - np.dot(w, v) * v
    u = u / np.linalg.norm(u)
    return u

def OrthonormalVectorSingle(v):
    v = np.asarray(v, dtype=np.complex128)
    if np.linalg.norm(v) == 0:
        raise ValueError("Zero vector has no well-defined orthonormal vector.")

    # Normalize v
    v = v / np.linalg.norm(v)

    n = len(v)

    # Choose a basis vector not parallel to v
    for i in range(n):
        if abs(v[i]) < 0.9:
            w = np.zeros(n, dtype=np.complex128)
            w[i] = 1.0
            break

    # Gram–Schmidt with Hermitian inner product
    u = w - np.vdot(v, w) * v   # vdot = conjugate(v) @ w
    u = u / np.linalg.norm(u)

    return u

def OrthonormalVector(v, tol=1e-12):
    v = np.asarray(v, dtype=np.complex128)
    n = v.size

    if np.linalg.norm(v) < tol:
        raise ValueError("Zero vector has no well-defined orthogonal complement.")

    # Normalize v
    v = v / np.linalg.norm(v)

    basis = []

    for i in range(n):
        w = np.zeros(n, dtype=np.complex128)
        w[i] = 1.0

        # Orthogonalize against v and previous basis vectors
        w = w - np.vdot(v, w) * v
        for b in basis:
            w = w - np.vdot(b, w) * b

        norm = np.linalg.norm(w)
        if norm > tol:
            basis.append(w / norm)

    return np.array(basis)[0]


def SetOrthonormalVectors(Lgstate_jk, Rgstate_jk):
    conf = len(Lgstate_jk)
    m = len(Lgstate_jk[0])

    OLgstate_jk = []
    for i in range(conf):
        via = []
        for j in range(m):
            if j == 0:
                orthovec = np.array([0.0])
            else:
                orthovec = OrthonormalVector(Lgstate_jk[i][j])
            via.append(np.array(orthovec))
        OLgstate_jk.append(via)

    ORgstate_jk = []
    for i in range(conf):
        via = []
        for j in range(m):
            if j == 0:
                orthovec = np.array([0.0])
            else:
                orthovec = OrthonormalVector(Rgstate_jk[i][j])
            via.append(np.array(orthovec))
        ORgstate_jk.append(via)

    return OLgstate_jk, ORgstate_jk

def SetConvolutionVectors(Ct_jk, Lgstate_jk, Rgstate_jk):
    conf = len(Lgstate_jk)
    mmax = len(Lgstate_jk[0])

    Lgconv_jk = []
    for i in range(conf):
        via = []
        for m in range(mmax):
            cvec = []
            for s in range(m+1):
                value = 0
                for t in range(m+1):
                    value += Ct_jk[i][s+t] * Lgstate_jk[i][m][t]
                cvec.append(value)
            via.append(np.array(cvec))
        Lgconv_jk.append(via)

    Rgconv_jk = []
    for i in range(conf):
        via = []
        for m in range(mmax):
            cvec = []
            for s in range(m+1):
                value = 0
                for t in range(m+1):
                    value += Ct_jk[i][s+t] * Rgstate_jk[i][m][t]
                cvec.append(value)
            via.append(np.array(cvec))
        Rgconv_jk.append(via)


    return Lgconv_jk, Rgconv_jk

def VectorNormalization(Ct_jk, Lvector_jk, Rvector_jk):
    norm_jk = []

    # normalization
    conf = len(Lvector_jk)
    mmax = len(Lvector_jk[0])
    for i in range(conf):
        via = []
        for m in range(mmax):
            normalization = 0
            for s in range(m+1):
                for t in range(m+1):
                    normalization += Lvector_jk[i][m][s] * Rvector_jk[i][m][t] * Ct_jk[i][s+t]
            via.append(np.sqrt(normalization))
        norm_jk.append(via)

    return norm_jk


def D1LowEnergyConstant(Ct_jk, Lgstate_jk, Rgstate_jk, nt, mmax):
    D1_jk = []

    conf = len(Ct_jk)
    # normalization
    TrueNorm_jk = VectorNormalization(Ct_jk, Lgstate_jk, Rgstate_jk)

    # convoluted vector
    Lgconv_jk, Rgconv_jk = SetConvolutionVectors(Ct_jk, Lgstate_jk, Rgstate_jk)

    # orthonormal vector
    OLgstate_jk, ORgstate_jk = SetOrthonormalVectors(Lgconv_jk, Rgconv_jk)

    # normalization
    BarNorm_jk = VectorNormalization(Ct_jk, OLgstate_jk, ORgstate_jk)

    for i in range(conf):
        via = []
        for m in range(mmax):
            true = 0
            bar = 0
            for s in range(m+1):
                for t in range(m+1):
                    true += Lgstate_jk[i][m][s] * Rgstate_jk[i][m][t] * Ct_jk[i][s+t+1]
                    bar += OLgstate_jk[i][m][s] * ORgstate_jk[i][m][t] * Ct_jk[i][s+t+1]
            true = true / (TrueNorm_jk[i][m]**2)
            bar  = bar / (BarNorm_jk[i][m]**2)
            value = (bar - true) / true
            via.append(value)
        D1_jk.append(via)

    return D1_jk

def D2LowEnergyConstant(Ct_jk, Lgstate_jk, Rgstate_jk, nt, mmax):
    D2_jk = []

    # normalization
    TrueNorm_jk = VectorNormalization(Ct_jk, Lgstate_jk, Rgstate_jk)

    # convoluted vector
    Lgconv_jk, Rgconv_jk = SetConvolutionVectors(Ct_jk, Lgstate_jk, Rgstate_jk)

    # orthonormal vector
    OLgstate_jk, ORgstate_jk = SetOrthonormalVectors(Lgconv_jk, Rgconv_jk)

    # normalization
    BarNorm_jk = VectorNormalization(Ct_jk, OLgstate_jk, ORgstate_jk)

    conf = len(Ct_jk)
    for i in range(conf):
        via = []
        for m in range(mmax):
            true = 0
            bar = 0
            for s in range(m+1):
                for t in range(m+1):
                    true += Lgstate_jk[i][m][s] * Rgstate_jk[i][m][t] * Ct_jk[i][s+t+1]
                    bar += OLgstate_jk[i][m][s] * ORgstate_jk[i][m][t] * Ct_jk[i][s+t+2]
            true = true / (TrueNorm_jk[i][m]**2)
            bar  = bar / (BarNorm_jk[i][m]**2)
            value = (bar - true**2) / (true**2)
            via.append(value)
        D2_jk.append(via)

    return D2_jk

def SingleRankEnergyVariance(Ct_jk, Lgstate_jk, Rgstate_jk, nt, mmax):
    Ene_jk = []
    Var_jk = []

    conf = len(Ct_jk)
    # normalization
    TrueNorm_jk = VectorNormalization(Ct_jk, Lgstate_jk, Rgstate_jk)

    for i in range(conf):
        viaE = []
        viaV = []
        for m in range(mmax):
            value0 = 0
            value1 = 0
            for s in range(m+1):
                for t in range(m+1):
                    value0 += Lgstate_jk[i][m][s] * Rgstate_jk[i][m][t] * Ct_jk[i][s+t+2]
                    value1 += Lgstate_jk[i][m][s] * Rgstate_jk[i][m][t] * Ct_jk[i][s+t+1]
            value0 = value0 / (TrueNorm_jk[i][m]**2)
            value1 = value1 / (TrueNorm_jk[i][m]**2)
            valueV = (value0 - value1**2) #/ (value1**2)
            valueE = value1
            print(valueE, valueV, value0)
            viaV.append(valueV)
            viaE.append(valueE)
        Ene_jk.append(viaE)
        Var_jk.append(viaV)

    return Ene_jk, Var_jk

#=============================================================#
#=============================================================#

def GetZCW(compevs_jk, compvec_jk, lcompvec_jk, exy_list, zcwtole):

    zcwevs_jk = []
    rzcwvec_jk = []
    lzcwvec_jk = []

    #====#
    # N_it   : Number of the iteration = size of the materix
    # N_boot : Number of the bootstrap sample
    #====#
    N_it = int(len(compevs_jk[0]))
    N_boot = int(len(compevs_jk))
    conf = len(compevs_jk)

    #====#
    # discard the spurious eigenvalues through the zcw test
    #====#
    for i in range(conf):
        via = []
        rvia = []
        lvia = []
        for j in range(N_it):
            cand_via = []
            cand_rvia = []
            cand_lvia = []
            for l in range(len(compevs_jk[i][j])):
                evalue = compevs_jk[i][j][l]
                revector = compvec_jk[i][j][l]
                levector = lcompvec_jk[i][j][l]
                correlator = exy_list[i]
                zcw = np.abs(ZCWtest(evalue, revector, levector, correlator))
                if zcw > zcwtole:
                    cand_via.append(evalue)
                    cand_rvia.append(revector)
                    cand_lvia.append(levector)
            via.append(cand_via)
            rvia.append(cand_rvia)
            lvia.append(cand_lvia)
        zcwevs_jk.append(via)
        rzcwvec_jk.append(rvia)
        lzcwvec_jk.append(lvia)


    return zcwevs_jk, rzcwvec_jk, lzcwvec_jk


def ZCWtest(evalue, revector, levector, correlator):
    # normalization for the enregy eigenvector
    normR = NormalizationEigenvector(revector, levector, correlator, "right")
    normL = NormalizationEigenvector(revector, levector, correlator, "left")

    rAmp = 0
    lAmp = 0
    for m in range(len(revector)):
        rAmp += revector[m] * correlator[m] / normR
        lAmp += levector[m] * correlator[m] / normL
    value = rAmp * lAmp / correlator[0]

    return value


def NormalizationEigenvector(revector, levector, correlator, leftright):
    normalization = 0
    for mi in range(len(revector)):
        for mj in range(len(levector)):
            if leftright == "both":
                normalization += levector[mj] * revector[mi] * correlator[mi+mj]
            elif leftright == "left":
                normalization += levector[mj] * levector[mi].conjugate() * correlator[mi+mj]
            elif leftright == "right":
                normalization += revector[mj] * levector[mi].conjugate() * correlator[mi+mj]

    return normalization

def NestedBootstrapEstimator(data_jk, Nboot, Nouter):
    median = statistics.median(data_jk)

    histogram = []
    for i in range(Nboot):
        np.random.seed(i)
        via = []
        for j in range(Nouter):
            bs = np.random.randint(Nouter)
            via.append(data_jk[bs])
        histogram.append(statistics.median(via))

    error = 0.5 * (np.quantile(histogram, 0.5*(1.+sp.special.erf(np.sqrt(2)))) 
                   - np.quantile(histogram, 0.5*(1.-sp.special.erf(np.sqrt(2)))))

    return histogram, median, error

def Evs2List(physevs_jk, ms):
    data_jk = []
    for i in range(len(physevs_jk)):
        data_jk.append(physevs_jk[i][ms][0])
    return data_jk

#=============================================================#
#=============================================================#

def ResidualBound_same(exy_list, ithevs_jk, ithvec_jk, ithlvec_jk):
    conf = len(ithevs_jk)

    # normalization for the enregy eigenvector
    norm_jk = NormEnergyEigenvector(ithevs_jk, ithvec_jk, ithlvec_jk, exy_list, "both")
    normR_jk = NormEnergyEigenvector(ithevs_jk, ithvec_jk, ithlvec_jk, exy_list, "both")
    normL_jk = NormEnergyEigenvector(ithevs_jk, ithvec_jk, ithlvec_jk, exy_list, "both")

    ith1st_jk = []
    for b in range(conf):
        via1 = []
        for r in range(len(ithevs_jk[b])): # each rank
            cand_via1 = []
            for k in range(len(ithevs_jk[b][r])): # single state, only k=0
                value1 = 0
                for i in range(len(ithvec_jk[b][r][k])):
                    for j in range(len(ithlvec_jk[b][r][k])):
                        value1 += ithlvec_jk[b][r][k][j] * ithvec_jk[b][r][k][i] * exy_list[b][i+j+2] / (normL_jk[r][b][k]*normR_jk[r][b][k])
                cand_via1.append(value1.real)
            via1.append(cand_via1)
        ith1st_jk.append(via1)

    ith2nd_jk = []
    for b in range(conf):
        via2 = []
        for r in range(len(ithevs_jk[b])): # each rank
            cand_via2 = []
            for k in range(len(ithevs_jk[b][r])): # single state, only k=0
                value2 = 0
                for i in range(len(ithvec_jk[b][r][k])):
                    for j in range(len(ithlvec_jk[b][r][k])):
                        value2 += ithlvec_jk[b][r][k][j] * ithvec_jk[b][r][k][i] * exy_list[b][i+j+1] / (normL_jk[r][b][k]*normR_jk[r][b][k])
                cand_via2.append(value2.real * ithevs_jk[b][r][k])
            via2.append(cand_via2)
        ith2nd_jk.append(via2)

    ith3rd_jk = []
    for b in range(conf):
        via3 = []
        for r in range(len(ithevs_jk[b])): # each rank
            cand_via3 = []
            for k in range(len(ithevs_jk[b][r])): # single state, only k=0
                value3 = ithevs_jk[b][r][k]**2
                cand_via3.append(value3.real)
            via3.append(cand_via3)
        ith3rd_jk.append(via3)

    ithrb_jk = []
    for b in range(conf):
        via = []
        for r in range(len(ith1st_jk[b])): # each rank
            cand_via = []
            for k in range(len(ith1st_jk[b][r])): # single state, only k = 0
                value = ith1st_jk[b][r][k] - 2 * ith2nd_jk[b][r][k] + ith3rd_jk[b][r][k]
                cand_via.append(value)
            via.append(cand_via)
        ithrb_jk.append(via)

    return ithrb_jk

def PrintResidualBound(means, errs, state):
    davmfile = './diagonalized_data/rb' + str(state)
    dav = open(davmfile, 'w')

    for i in range(len(means)):
        strdata = str(i+1) + ' ' + str(means[i][0]) + ' ' + str(errs[i][0]) + '\n'
        dav.write(strdata)
    dav.close()

    return 0


def PrintSingleResidualBound(datalist, m, state):

    conf = len(datalist)
    maxrank = len(datalist[0])
    for r in range(maxrank):
        djkmfile = './diagonalized_data/rb' + str(state) + 'rank=' + str(r+1) + 'm=' + str(m) + '_jk'
        davmfile = './diagonalized_data/rb' + str(state) + 'rank=' + str(r+1) + 'm=' + str(m)
        djk = open(djkmfile, 'w')
        dav = open(davmfile, 'w')

        varrth_jk = []
        for b in range(conf):
            value = datalist[b][r][0]
            varrth_jk.append(value)

        ave, err = BootstrapAnal(varrth_jk, conf)
        strdata = str(2*(m)+1) + ' ' + str(ave) + ' ' + str(err) + '\n'
        dav.write(strdata)
        dav.close()

        for b in range(conf):
            strbdata = str(b) + ' ' + str(varrth_jk[b]) + '\n'
            djk.write(strbdata)
        djk.close()

    return 0

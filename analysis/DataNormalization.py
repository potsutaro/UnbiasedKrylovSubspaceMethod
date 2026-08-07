import numpy as np
import copy
import scipy.optimize


#=============================================================#
#=============================================================#
# Normalization

def NormalizedValue(twy_list, t0, flag):
    conf = len(twy_list)
    y_transpose = np.array(twy_list).T
    normalization_mean, normalization_err = StatisticalAnal(y_transpose[2*t0], conf, flag)

    return normalization_mean, normalization_err


def TwoPtCtNormalization(twy_list, size, t0, flag):
    conf = len(twy_list)
    y_transpose = np.array(twy_list).T
    normalization_mean, normalization_err = StatisticalAnal(y_transpose[2*t0], conf, flag)

    ntwy_list = []
    for i in range(len(twy_list)):
        via = []
        for j in range(len(twy_list[0])):
            if (j < size):
                index = int((j+2*t0)%len(twy_list[0]))
                # value = twy_list[i][index] / normalization_mean
                value = twy_list[i][index] / twy_list[i][2*t0] 
                via.append(value)
        ntwy_list.append(via)

    return ntwy_list

def TwoPtCtNormalizationPlusOne(twy_list, size, t0, flag):
    conf = len(twy_list)
    y_transpose = np.array(twy_list).T
    normalization_mean, normalization_err = StatisticalAnal(y_transpose[2*t0], conf, flag)

    ntwy_list = []
    for i in range(len(twy_list)):
        via = []
        for j in range(len(twy_list[0])):
            if (j < (size+1)):
                index = int((j+2*t0)%len(twy_list[0]))
                # value = twy_list[i][index] / normalization_mean
                value = twy_list[i][index] / twy_list[i][2*t0] 
                via.append(value)
        ntwy_list.append(via)

    return ntwy_list


def ThreePtCtNormalization(thy_list, t0, flag):
    conf = len(thy_list[0])
    tsep = len(thy_list)
    nts = tsep

    bsy_list = []
    ts_list = []
    for ts in range(1,tsep):
        bsy_tsep = []
        for i in range(conf):
            via = []
            for j in range(len(thy_list[ts][0])):
                value = thy_list[ts][i][j]
                via.append(value)
            bsy_tsep.append(via)
        bsy_list.append(bsy_tsep)
        ts_list.append(ts)


    return bsy_list, ts_list


def CombPtCtNormalization(thy_list, twl_list, twr_list, size, t0, flag):
    conf = len(twl_list)
    tsep = len(thy_list)
    nts = tsep

    bsy_list = []
    for ts in range(1,tsep+1):
        bsy_tsep = []
        for i in range(conf):
            via = []
            for j in range(tsep):
                value = thy_list[ts][i][j]
                via.append(value)
            bsy_tsep.append(via)
        bsy_list.append(bsy_tsep)

    return bsy_list


#=============================================================#
#=============================================================#
# Normalization and Arrangement


def NormalizedThreePtCtArrangement(thy_list, t_min, t_max, t0, flag):
    conf = len(thy_list[0])
    tsep = len(thy_list)
    nts = tsep - t0 - t0

    thrpt_list = copy.deepcopy(thy_list)
    bsa_list = []
    bsn_list = []
    for t in range(t_min, t_max+1):
        bsa_t = []
        bsn_t = []
        for i in range(conf):
            via = []
            for ts in range(t, nts):
                value = thrpt_list[ts+t0+t0-1][i][t+t0] / thrpt_list[t+t0+t0-1][i][t+t0] # -1 to start from 0
                via.append(value)
            bsa_t.append(via)
        bsa_list.append(bsa_t)

    via_bsn = []
    for i in range(conf):
        via = []
        for t in range(t_min, t_max+1):
            value = thrpt_list[t+t0+t0-1][i][t+t0]
            via.append(value)
        via_bsn.append(via)

    bsn_transpose = np.array(via_bsn).T
    for t in range(len(bsn_transpose)):
        n_mean, n_err = StatisticalAnal(bsn_transpose[t], conf, flag)
        via_n = [n_mean, n_err]
        bsn_list.append(via_n)


    return bsa_list, bsn_list

#=============================================================#
#=============================================================#
# make correlator

def MakeFittedCorrelator(T, effective_mass_jk, amplitude_jk):
    exy_list = []

    conf = len(amplitude_jk)
    for i in range(conf):
        via = []
        for j in range(T):
            data = amplitude_jk[i] * np.exp(-effective_mass_jk[i] * j)
            # data = np.exp(-effective_mass_jk[i] * j)
            via.append(data)
        exy_list.append(via)

    return exy_list


#=============================================================#
#=============================================================#
# Statistics 

def StatisticalAnal(result, n, flag):
    ave = np.sum(np.array(result)) / n
    var = 0
    for r in result:
        if (flag == 0):
            var += (1. - 1. / n) * ((ave - r)**2)
        else:
            var += 1. / n * ((ave - r)**2)
    err = np.sqrt(var)
    return ave, err


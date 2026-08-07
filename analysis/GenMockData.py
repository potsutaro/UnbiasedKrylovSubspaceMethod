import sys
import numpy as np

#=============================================================#
#=============================================================#
# Statistics

def Jackknife(ex, av, conf):

    jk = []
    for i in range(len(ex)):
        via = []
        for j in range(len(ex[0])):
            if (conf > 1):
                val = (conf * av[j] - ex[i][j]) / (conf - 1)
            else:
                val = ex[i][j]
            via.append(val)
        jk.append(via)
        via = []

    return jk


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
        var += (1. / n) * ((ave -r)**2)
    err = np.sqrt(var)
    return ave, err



#=============================================================#
#=============================================================#
# model functions

def ExpWrapper(x, a, states):
    C = 0

    if states == 1:
        C = SingleExp(x, a)
    elif states == 2:
        # C = DoubleExp(x, a)
        C = MultiExp(x, a, states)
    elif states == 3:
        C = MultiExp(x, a, states)
    elif states == 9:
        C = AsymDoubleExp(x, a)
    else:
        C = MultiExp(x, a, states)

    return C


def SingleExp(x, a):
    C = np.exp( - 0.1 / a * x )
    return C

def DoubleExp(x, a):
    C =  np.exp( - 0.1 / a * x )
    C += 0.5 * np.exp( - 0.6 / a * x )
    # C += 1.0 * np.exp( - 0.6 / a * x )
    # C += 0.8 * np.exp( - 0.6 / a * x )
    # C += 0.5 * np.exp( - 1.5 / a * x )
    # C += np.exp( - 0.55 / a * x )
    return C

def AsymDoubleExp(x, a):
    C =  1.2 * 0.8 * np.exp( - 0.1 / a * x )
    C += 0.6 * 0.3 * np.exp( - 0.6 / a * x )
    return C


def MultiExp(x, a, states):
    C = 0
    for n in range(states):
        # zn = n+1
        aEn = 0.1*(n+1)
        # C = C + zn**2 * np.exp(-aEn / a * x)
        # zn = 1 / (n+1)
        # aEn = 0.1*(10*n+1)
        zn = 1. / np.sqrt(2.*aEn)
        C = C + zn**2 * np.exp(-aEn / a * x)
    return C


def RealProton(x):
    M_N = 0.94       # Nucleon mass in GeV
    M_R = 1.44       # Nucleon 1st excited state mass in GeV
    M_pi = 0.14      # Pion mass in GeV
    A1 = 1.0          # Amplitude of signal
    A2 = 0.5          # Amplitude of signal
    a = 3.1
    
    C_mean = A1 * np.exp(-M_N / a * x) 
    C_mean += A2 * np.exp(-M_R / a * x) 

    C_stddev = np.exp(-1.5*M_pi / a *x)

    return C_mean, C_stddev


#=============================================================#
#=============================================================#
# model functions for Thrpt

def ThrptExpWrapper(x, a, states, tsep):
    C = 0

    if states == 1:
        C = SingleThrptExp(x, a, tsep)
    elif states == 2:
        C = DoubleThrptExp(x, a, tsep)
    elif states == 3:
        C = MultiThrptExp(x, a, tsep, states)
    else:
        C = MultiThrptExp(x, a, tsep, states)

    return C


def SingleThrptExp(x, a, tsep):
    J = [[0 for j in range(2)] for i in range(2)]
    J[0][0] = 1.0
    
    C = 0

    C = C + J[0][0] * np.exp(- 0.1 / a * (tsep)) 

    return C

def DoubleThrptExp(x, a, tsep):

    J = [[0 for j in range(2)] for i in range(2)]
    J[0][0] = 1.0
    J[0][1] = 0.8
    J[1][0] = 0.8
    J[1][1] = 0.1
    
    C = 0

    C = C + J[0][0] * np.exp(- 0.1 / a * (tsep)) 

    C = C + J[1][0] * np.exp(- 1. / a * (tsep - x)) * np.exp(- 0.1 / a * x) * np.sqrt(0.5) 

    C = C + J[0][1] * np.exp(- 0.1 / a * (tsep - x)) * np.exp(- 1. / a * x) * np.sqrt(0.5)

    C = C + J[1][1] * np.exp(- 1. / a * tsep) * 0.5

    return C


def MultiThrptExp(x, a, tsep, states):


    # # matrix element
    # J = [[0 for j in range(states)] for i in range(states)]
    # for i in range(len(J)):
    #     for j in range(len(J[0])):
    #         J[i][j] = 1/(1+i+j)         # type-I
    #         if i != j:
    #             J[i][j] = J[i][j] * 0.5   # type-II
    #         #     J[i][j] = 0           # type-III
    #         #     # J[i][j] = J[i][j] * 0.1 # type-IV

    # matrix element 2
    J = [[0 for j in range(states)] for i in range(states)]
    for i in range(len(J)):
        for j in range(len(J[0])):
            J[i][j] = 1/(1+i*j)         # type-I
            if i != j:
                J[i][j] = J[i][j] * 0.5   # type-II
                # J[i][j] = 0           # type-III
                # J[i][j] = J[i][j] * 0.1 # type-IV

    # # hamiltonian
    # J = [[0 for j in range(states)] for i in range(states)]
    # for i in range(len(J)):
    #     for j in range(len(J[0])):
    #         # J[i][j] = 0.1*(i+1) 
    #         # J[i][j] = 1 / (i+1) / (j+1)
    #         # J[i][j] = 0.1 / (i+1) / (j+1)
    #         # J[i][j] = 1/(1+i+j)
    #         J[i][j] = 1/(1+i*j)
    #         # J[i][j] = 1. / (1.+2*i)
    #         # J[i][j] = (i+1) 
    #         if i != j:
    #             J[i][j] = 0


    # Energy
    aEn = [0] * states
    for i in range(states):
        aEn[i] = 0.1*(i+1)

    # Amplitude
    zn = [0] * states
    for i in range(states):
        zn[i] = 1. / np.sqrt(2.*aEn[i])


    # Correlation function
    C = 0
    for i in range(len(J)):
        for j in range(len(J[0])):
            C = C + zn[j] * zn[i] * J[i][j] * np.exp(- aEn[j] / a * (tsep - x)) * np.exp(- aEn[i] / a * x)

    return C


#=============================================================#
#=============================================================#
# Generate mock data

def GenNaiveConMock(conf, size, a, states, err):
    jk_list = []
    for i in range(conf):
        via = []
        np.random.seed(i)
        for j in range(size):
            data = ExpWrapper(j, a, states)
            # data = data + np.random.normal(0.0, data * err)
            data = data * (1.0 + err * np.random.normal(0.0, 1.0))
            via.append(data)
        jk_list.append(via)

    return jk_list


def GenNaiveExpMock(conf, size, a, states, err):
    jk_list = []
    for i in range(conf):
        via = []
        np.random.seed(i)
        for j in range(size):
            data = ExpWrapper(j, a, states)
            # data = data * (1. + err * np.random.normal(0.0, np.exp( j * 0.06 ))) # 1.5 * 0.1 / 10
            data = data * (1. + 0.2 * np.random.normal(0.0, np.exp( j * 0.06 ))) # 1.5 * 0.1 / 10
            # mean, dev = RealProton(j)
            # data = np.random.normal(loc=mean, scale=dev)
            via.append(data)
        jk_list.append(via)

    return jk_list



def GenNaiveConMockThrpt(conf, size, a, states, tsep, err):
    jk_list = []
    for i in range(conf):
        via = []
        np.random.seed(i+1)
        ref = ThrptExpWrapper(tsep, a, states, tsep)
        for j in range(size):
            data = ThrptExpWrapper(j, a, states, tsep)
            data = data + ref * err * np.random.normal(0.0, 1.0)
            # data = data * (1.0 + err * np.random.normal(0.0, 1.0))
            via.append(data)
        jk_list.append(via)

    return jk_list


#=============================================================#
#=============================================================#
# misc 

def PrintJKFile(data_list, name, directory):
    djkfile = './' + directory + '/' + name + '_jk'
    d = open(djkfile, 'w')

    for i in range(len(data_list)):
        for j in range(len(data_list[0])):
            strdata = str(j) + ' ' + str(data_list[i][j]) + '\n'
            d.write(strdata)
    d.close

    return 0


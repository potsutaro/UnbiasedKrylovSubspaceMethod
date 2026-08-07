import numpy as np
import scipy as sp

#=============================================================#
#=============================================================#
# data alignment
def DataImportWithRange(datapath, datafile, lower, upper, size):
    fitrange = [lower, upper]
    dof = int(upper - lower + 1)

    exx_list = []
    exy_list = []
    
    cor_list = []
    
    inx_list = []
    iny_list = []

    lin = 0

    jkfile = './' + datapath + '/' + datafile + '_jk'
    avfile = './' + datapath + '/' + datafile
    
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
            sig_list.append(float(data[2]))
        lin += 1

    return exx_list, exy_list
    
    




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


def CovarianceMatrix(y_list, n):

    # y-part
    ydof = len(y_list[0])
    vector_ybar = np.sum(np.array(y_list), axis=0) / n
    ontosi_ybar = np.reshape(vector_ybar, (1, ydof))
    sitoon_ybar = np.reshape(vector_ybar, (ydof, 1))

    ycov = np.matrix(np.zeros((ydof, ydof)))

    for i in range(n):
        sitoon_yi = np.reshape(np.array(y_list[i]), (ydof, 1))
        ontosi_yi = np.reshape(np.array(y_list[i]), (1, ydof))
        ycov += np.dot((sitoon_yi-sitoon_ybar),(ontosi_yi-ontosi_ybar)) / n

    return ycov

#=============================================================#
#=============================================================#
# fitting

def Prediction(p, x):
    mu = []
    stateNum = int(len(p)*0.5)
    size = len(x)
    for a in range(size):
        value = 0
        for s in range(stateNum):
            value += p[2*s] * np.exp(-p[2*s+1]*x[a])
        mu.append(value)

    return mu


def Chisquare(p, x, y, J, ycov, size):
    #=============================================================#
    #=============================================================#
    # Initialization
    dof = size

    #=============================================================#
    #=============================================================#
    # y-part
    #=============================================================#
    #=============================================================#
    # Residual vector 
    yresidual = np.array(y) - np.array(Prediction(p, x))

    #=============================================================#
    #=============================================================#
    # Choresky decomposition of the covariance matrix 
    yL = np.linalg.cholesky(ycov)

    #=============================================================#
    #=============================================================#
    # Data vector
    y_data = np.reshape(yresidual, (dof,1))
    yV_data = np.dot(yL.I, y_data)

    #=============================================================#
    #=============================================================#
    # total vector
    V_total =[]
    for j in range(dof):
        V_total.append(yV_data[j][0,0])


    return V_total

def LsqFitting(p, x, y, J, ycov, dof):
    #=============================================================#
    #=============================================================#
    # least_square fitting
    res_lsq = sp.optimize.least_squares(Chisquare, p, args=(x, y, J, ycov, dof))
    result = res_lsq.x

    return result


#=============================================================#
#=============================================================#
# misc

def SplitList(l, n):
    for idx in range(0, len(l), n):
        yield list(l[idx:idx + n])

def ResultSplitting(result_jk, s, stateNum):
    conf = len(result_jk)
    sresult_jk = []
    for i in range(conf):
        splittedResult = list(SplitList(result_jk[i], 2))
        sresult_jk.append((sorted(splittedResult, key=lambda x: x[1]))[s])

    return sresult_jk

def ResultParameter(sresult_jk):
    amp_jk = []
    mass_jk = []

    for i in range(len(sresult_jk)):
        amp_jk.append(sresult_jk[i][0])
        mass_jk.append(sresult_jk[i][1])

    return amp_jk, mass_jk

def PrintFile(targetdir, target, result_jk, state, stateNum):
    djkmfile = './fitting_data/' + targetdir + '/' + target + str(state) + 'stateNum=' + str(stateNum) + '_jk'
    davmfile = './fitting_data/' + targetdir + '/' + target + str(state) + 'stateNum=' + str(stateNum)
    djk = open(djkmfile, 'w')
    dav = open(davmfile, 'w')

    conf = len(result_jk)
    for i in range(conf):
        strdata = str(i) + ' ' + str(result_jk[i]) + '\n'
        djk.write(strdata)
    djk.close()

    ave, err = BootstrapAnal(result_jk, conf)
    strdata = str(ave) + ' ' + str(err) + '\n'
    dav.write(strdata)
    dav.close()

    return 0










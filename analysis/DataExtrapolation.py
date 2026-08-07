import numpy as np
import scipy as sp
import gvar as gv
import lsqfit

#=============================================================#
#=============================================================#
# data alignment

def DataAlignmentWithVariance(datapath, target, rankmax, rankmin, msize, state):
    iny_list = []
    inx_list = []

    # variance
    for r in range(rankmin,rankmax+1):
        jkfile = datapath + '/var' + str(state) + 'rank=' + str(r) + 'm=' + str(msize) + '_jk'
        f = open(jkfile, 'rt')
        via = []
        for string in f:
            data = string[:-1].split(' ')
            via.append(float(data[1]))
        inx_list.append(via)

    conf = len(inx_list[0])
    exx_list = []
    for i in range(conf):
        via = []
        for r in range(len(inx_list)):
            via.append(inx_list[r][i])
        exx_list.append(via)

    # target
    for r in range(rankmin,rankmax+1):
        jkfile = datapath + '/' + target + str(state) + 'rank=' + str(r) + 'm=' + str(msize) + '_jk'
        f = open(jkfile, 'rt')
        via = []
        for string in f:
            data = string[:-1].split(' ')
            via.append(float(data[1]))
        iny_list.append(via)
            
    exy_list = []
    for i in range(conf):
        via = []
        for r in range(len(iny_list)):
            via.append(iny_list[r][i])
        exy_list.append(via)

    return exx_list, exy_list, inx_list, iny_list

    
def ThrptAlignmentWithVariance(datapath, target, rankmax, rankmin, msize, state):
    #=============================================================#
    #=============================================================#
    # store oppsitely
    # variance as the y-axis, matrix element as the x-axis

    # Istate = Fstate, Irank = Frank
    iny_list = []
    inx_list = []

    # variance
    for r in range(rankmin,rankmax+1):
        jkfile = datapath + '/var' + str(state) + 'rank=' + str(r) + 'm=' + str(msize) + '_jk'
        f = open(jkfile, 'rt')
        via = []
        for string in f:
            data = string[:-1].split(' ')
            via.append(float(data[1]))
        inx_list.append(via)

    conf = len(inx_list[0])
    exy_list = []
    for i in range(conf):
        via = []
        for r in range(len(inx_list)):
            via.append(inx_list[r][i])
        exy_list.append(via)

    # target
    for r in range(rankmin,rankmax+1):
        jkfile = datapath + '/' + target
        jkfile += 'I=' + str(state) + 'F=' + str(state)
        jkfile += 'Irank=' + str(r) + 'Frank=' + str(r)
        jkfile += 'm=' + str(msize) + '_jk'
        f = open(jkfile, 'rt')
        via = []
        for string in f:
            data = string[:-1].split(' ')
            via.append(float(data[1]))
        iny_list.append(via)
            
    # store with squared
    exx_list = []   
    for i in range(conf):
        via = []
        for r in range(len(iny_list)):
            via.append(iny_list[r][i])
        exx_list.append(via)

    return exx_list, exy_list, inx_list, iny_list
    
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


def CovarianceMatrix(x_list, y_list, n):

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

    # x-part
    xdof = len(x_list[0])
    vector_xbar = np.sum(np.array(x_list), axis=0) / n
    ontosi_xbar = np.reshape(vector_xbar, (1, xdof))
    sitoon_xbar = np.reshape(vector_xbar, (xdof, 1))

    xcov = np.matrix(np.zeros((xdof, xdof)))

    for i in range(n):
        sitoon_xi = np.reshape(np.array(x_list[i]), (xdof, 1))
        ontosi_xi = np.reshape(np.array(x_list[i]), (1, xdof))
        xcov += np.dot((sitoon_xi-sitoon_xbar),(ontosi_xi-ontosi_xbar)) / n

    return xcov, ycov

#=============================================================#
#=============================================================#
# fitting

def PredictionYR(p, ranksep):
    ymu = []
    for r in range(ranksep):
        value = 0
        actp = p[ranksep:]
        varp = p[:ranksep]
        yf = np.poly1d(actp)
        value = yf(varp[r])
        ymu.append(value)
        value = 0

    return ymu


def PredictionXR(p, ranksep):
    xmu = []
    for r in range(ranksep):
        varp = p[:ranksep]
        xmu.append(varp[r])

    return xmu


def Chisquare(p, x, y, J, xcov, ycov, ranksep):
    #=============================================================#
    #=============================================================#
    # Initialization
    dof = ranksep

    #=============================================================#
    #=============================================================#
    # y-part
    #=============================================================#
    #=============================================================#
    # Residual vector 
    yresidual = np.array(y) - np.array(PredictionYR(p, ranksep))

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
    # x-part
    #=============================================================#
    #=============================================================#
    # Residual vector 
    xresidual = np.array(x) - np.array(PredictionXR(p, ranksep))

    #=============================================================#
    #=============================================================#
    # Choresky decomposition of the covariance matrix 
    xL = np.linalg.cholesky(xcov)

    #=============================================================#
    #=============================================================#
    # Data vector
    x_data = np.reshape(xresidual, (dof,1))
    xV_data = np.dot(xL.I, x_data)

    #=============================================================#
    #=============================================================#
    # Total = y-part + x-part
    #=============================================================#
    #=============================================================#
    # Total vector 
    V_total = []
    size_total = ranksep + ranksep
    for j in range(size_total):
        if j < ranksep:
            V_total.append(yV_data[j][0,0])
        else:
            V_total.append(xV_data[j-ranksep][0,0])

    return V_total
    
def Extrapolation(p, x, y, J, xcov, ycov, ranksep):
    #=============================================================#
    #=============================================================#
    # least_square fitting
    res_lsq = sp.optimize.least_squares(Chisquare, p, args=(x, y, J, xcov, ycov, ranksep))
    result = res_lsq.x

    return result

#=============================================================#
#=============================================================#
# noiseless

def NoiselessChisquare(p, x, y, J, ranksep):
    #=============================================================#
    #=============================================================#
    # Initialization
    dof = ranksep 

    #=============================================================#
    #=============================================================#
    # y-part
    #=============================================================#
    #=============================================================#
    # Residual vector 
    yresidual = np.array(y) - np.array(PredictionYR(p, ranksep))

    #=============================================================#
    #=============================================================#
    # Data vector
    y_data = np.reshape(yresidual, (dof,1))

    #=============================================================#
    #=============================================================#
    # x-part
    #=============================================================#
    #=============================================================#
    # Residual vector 
    xresidual = np.array(x) - np.array(PredictionXR(p, ranksep))

    #=============================================================#
    #=============================================================#
    # Data vector
    x_data = np.reshape(xresidual, (dof,1))

    #=============================================================#
    #=============================================================#
    # Total = y-part + x-part
    #=============================================================#
    #=============================================================#
    # Total vector 
    V_total = []
    size_total = ranksep + ranksep
    for j in range(size_total):
        if j < ranksep:
            V_total.append(y_data[j][0])
        else:
            V_total.append(x_data[j-ranksep][0])

    return V_total

    
def NoiselessExtrapolation(p, x, y, J, ranksep):
    #=============================================================#
    #=============================================================#
    # least_square fitting
    res_lsq = sp.optimize.least_squares(NoiselessChisquare, p, args=(x, y, J, ranksep))
    result = res_lsq.x

    return result


#=============================================================#
#=============================================================#
# fitting for a three-point correlation function

def PredictionYRThrpt(p, ranksep):
    ymu = []
    for r in range(ranksep):
        value = 0
        actp = p[ranksep:]
        varp = p[:ranksep]
        value = ((varp[r]-actp[1])/actp[0])**2 
        ymu.append(value)
        value = 0

    return ymu

def ChisquareThrpt(p, x, y, J, xcov, ycov, ranksep):
    #=============================================================#
    #=============================================================#
    # Initialization
    dof = ranksep

    #=============================================================#
    #=============================================================#
    # y-part
    #=============================================================#
    #=============================================================#
    # Residual vector 
    yresidual = np.array(y) - np.array(PredictionYRThrpt(p, ranksep))

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
    # x-part
    #=============================================================#
    #=============================================================#
    # Residual vector 
    xresidual = np.array(x) - np.array(PredictionXR(p, ranksep))

    #=============================================================#
    #=============================================================#
    # Choresky decomposition of the covariance matrix 
    xL = np.linalg.cholesky(xcov)

    #=============================================================#
    #=============================================================#
    # Data vector
    x_data = np.reshape(xresidual, (dof,1))
    xV_data = np.dot(xL.I, x_data)

    #=============================================================#
    #=============================================================#
    # Total = y-part + x-part
    #=============================================================#
    #=============================================================#
    # Total vector 
    V_total = []
    size_total = ranksep + ranksep
    for j in range(size_total):
        if j < ranksep:
            V_total.append(yV_data[j][0,0])
        else:
            V_total.append(xV_data[j-ranksep][0,0])

    return V_total
    
def ExtrapolationThrpt(p, x, y, J, xcov, ycov, ranksep):
    #=============================================================#
    #=============================================================#
    # least_square fitting
    res_lsq = sp.optimize.least_squares(ChisquareThrpt, p, args=(x, y, J, xcov, ycov, ranksep))
    result = res_lsq.x

    return result

#=============================================================#
#=============================================================#
# noiseless 

def NoiselessChisquareThrpt(p, x, y, J, ranksep):
    #=============================================================#
    #=============================================================#
    # Initialization
    dof = ranksep

    #=============================================================#
    #=============================================================#
    # y-part
    #=============================================================#
    #=============================================================#
    # Residual vector 
    yresidual = np.array(y) - np.array(PredictionYRThrpt(p, ranksep))

    #=============================================================#
    #=============================================================#
    # Data vector
    y_data = np.reshape(yresidual, (dof,1))

    #=============================================================#
    #=============================================================#
    # x-part
    #=============================================================#
    #=============================================================#
    # Residual vector 
    xresidual = np.array(x) - np.array(PredictionXR(p, ranksep))

    #=============================================================#
    #=============================================================#
    # Data vector
    x_data = np.reshape(xresidual, (dof,1))

    #=============================================================#
    #=============================================================#
    # Total = y-part + x-part
    #=============================================================#
    #=============================================================#
    # Total vector 
    V_total = []
    size_total = ranksep + ranksep
    for j in range(size_total):
        if j < ranksep:
            V_total.append(y_data[j][0])
        else:
            V_total.append(x_data[j-ranksep][0])

    return V_total
    
def NoiselessExtrapolationThrpt(p, x, y, J, ranksep):
    #=============================================================#
    #=============================================================#
    # least_square fitting
    res_lsq = sp.optimize.least_squares(NoiselessChisquareThrpt, p, args=(x, y, J, ranksep))
    result = res_lsq.x

    return result



#=============================================================#
#=============================================================#
# misc

def PrintDataWithVariance(targetdir, targetfile, rankmax, rankmin, msize, state, exx_list, exy_list, inx_list, iny_list):
    djkmfile = './extrapolation_data/' + targetdir + '/' + targetfile + str(state) + 'rankmax=' + str(rankmax) + 'rankmin' + str(rankmin) + 'm=' + str(msize) + '_jk'
    davmfile = './extrapolation_data/' + targetdir + '/' + targetfile + str(state) + 'rankmax=' + str(rankmax) + 'rankmin' + str(rankmin) + 'm=' + str(msize)
    djk = open(djkmfile, 'w')
    dav = open(davmfile, 'w')

    ranksep = rankmax - rankmin + 1
    sign = 1
    conf = len(exy_list)
    for i in range(conf):
        for r in range(ranksep):
            if targetfile == 'me':
                strdata = str(exx_list[i][r]) + ' ' + str(np.sqrt(exy_list[i][r])) + '\n'
            else:
                strdata = str(exx_list[i][r]) + ' ' + str(exy_list[i][r]) + '\n'
            djk.write(strdata)
    djk.close()

    for r in range(ranksep):
        xave, xerr = BootstrapAnal(inx_list[r], conf)
        yave, yerr = BootstrapAnal(iny_list[r], conf)
        strdata = str(xave) + ' ' + str(yave) + ' ' + str(xerr) + ' ' + str(yerr) + '\n'
        dav.write(strdata)
    dav.close()

    if targetfile == 'ev':
        davmfile = './extrapolation_data/' + targetdir + '/mass' + str(state) + 'rankmax=' + str(rankmax) + 'rankmin' + str(rankmin) + 'm=' + str(msize)
        dav = open(davmfile, 'w')


        for r in range(ranksep):
            mass_jk = []
            for i in range(len(iny_list[r])):
                mass_jk.append(-np.log(iny_list[r][i]))
            xave, xerr = BootstrapAnal(inx_list[r], conf)
            yave, yerr = BootstrapAnal(mass_jk, conf)
            strdata = str(xave) + ' ' + str(yave) + ' ' + str(xerr) + ' ' + str(yerr) + '\n'
            dav.write(strdata)
        dav.close()

    if targetfile == 'me':
        yave, yerr = BootstrapAnal(iny_list[-1], conf)
        sign = yave / np.abs(yave)

    return sign


def PrintTarget(targetdir, target, tarresult_jk, rankmax, rankmin, msize, state, sign):
    djkmfile = './extrapolation_data/' + targetdir + '/' + target + str(state) + 'rankmax=' + str(rankmax) + 'rankmin=' + str(rankmin) + 'm=' + str(msize) + '_jk'
    davmfile = './extrapolation_data/' + targetdir + '/' + target + str(state) + 'rankmax=' + str(rankmax) + 'rankmin=' + str(rankmin) + 'm=' + str(msize)
    djk = open(djkmfile, 'w')
    dav = open(davmfile, 'w')

    conf = len(tarresult_jk)
    for i in range(conf):
        strdata = str(i) + ' ' + str(tarresult_jk[i]) + '\n'
        djk.write(strdata)
    djk.close()

    ave, err = BootstrapAnal(tarresult_jk, conf)
    strdata = str(ave) + ' ' + str(err) + '\n'
    dav.write(strdata)
    dav.close()

    if target == 'ev':
        djkmfile = './extrapolation_data/mass/mass' + str(state) + 'rankmax=' + str(rankmax) + 'rankmin=' + str(rankmin) + 'm=' + str(msize) + '_jk'
        davmfile = './extrapolation_data/mass/mass' + str(state) + 'rankmax=' + str(rankmax) + 'rankmin=' + str(rankmin) + 'm=' + str(msize)
        djk = open(djkmfile, 'w')
        dav = open(davmfile, 'w')

        mass_jk = []
        for i in range(conf):
            mass_jk.append(-np.log(tarresult_jk[i]))

        for i in range(conf):
            strdata = str(i) + ' ' + str(mass_jk[i]) + '\n'
            djk.write(strdata)
        djk.close()

        ave, err = BootstrapAnal(mass_jk, conf)
        strdata = str(ave) + ' ' + str(err) + '\n'
        dav.write(strdata)
        dav.close()

    return 0

def PrintFitResult(targetdir, target, actresult_jk, rankmax, rankmin, msize, state, xlimave):
    davmfile = './extrapolation_data/' + targetdir + '/b' + target + str(state) + 'rankmax=' + str(rankmax) + 'rankmin=' + str(rankmin) + 'm=' + str(msize)
    dav = open(davmfile, 'w')

    scale = np.arange(0, xlimave, 0.00001)

    conf = len(actresult_jk)
    for s in scale:
        sresults = []
        for i in range(conf):
            f = np.poly1d(actresult_jk[i])
            sresults.append(f(float(s)))
        save, serr = BootstrapAnal(sresults, conf)
        strdata = str(s) + ' ' + str(save) + ' ' + str(serr) + '\n'
        dav.write(strdata)
    dav.close()


def PrintFitThrptLinear(targetdir, target, actresult_jk, rankmax, rankmin, msize, state, xlimave):
    davmfile = './extrapolation_data/' + targetdir + '/b' + target + str(state) + 'rankmax=' + str(rankmax) + 'rankmin=' + str(rankmin) + 'm=' + str(msize)
    dav = open(davmfile, 'w')

    scale = np.arange(0, xlimave, 0.00001)

    conf = len(actresult_jk)
    for s in scale:
        sresults = []
        for i in range(conf):
            value = (s - actresult_jk[i][1]) / actresult_jk[i][0]
            sresults.append(value)
        save, serr = BootstrapAnal(sresults, conf)
        strdata = str(s) + ' ' + str(save) + ' ' + str(serr) + '\n'
        dav.write(strdata)
    dav.close()



def PrintFitThrptSquareroot(targetdir, target, actresult_jk, rankmax, rankmin, msize, state, xlimave):
    davmfile = './extrapolation_data/' + targetdir + '/b' + target + str(state) + 'rankmax=' + str(rankmax) + 'rankmin=' + str(rankmin) + 'm=' + str(msize)
    dav = open(davmfile, 'w')

    scale = np.arange(0, xlimave, 0.00001)

    conf = len(actresult_jk)
    for s in scale:
        sresults = []
        for i in range(conf):
            value = actresult_jk[i][0] * np.sqrt(s) + actresult_jk[i][1]
            sresults.append(value)
        save, serr = BootstrapAnal(sresults, conf)
        strdata = str(s) + ' ' + str(save) + ' ' + str(serr) + '\n'
        dav.write(strdata)
    dav.close()























import sys
import numpy as np
import scipy.optimize
import copy

from . import DataNormalization

from config import Config

#=============================================================#
# call as a function


def main(cfg: Config):

    lower = cfg.analysis.fitini
    upper = cfg.analysis.fitfin
    datapath =  cfg.paths.corr
    datafile =  cfg.files.cdata
    flag =  cfg.statistics.stati
    t0 = cfg.analysis.normalization
    ssize = cfg.analysis.size
    T = cfg.lattice.T


    #=============================================================#
    #=============================================================#

    exx_list = []
    exy_list = []
    
    cor_list = []
    
    inx_list = []
    iny_list = []
    
    
    lin = 0
    fitrange = [lower, upper]
    dof = int(upper - lower + 1 )
    
    massfile = 'mass'
    
    
    jkfile = './' + datapath + '/' + massfile + '_jk'
    avfile = './' + datapath + '/' + massfile
    
    jkfile2 = './' + datapath + '/' + datafile + '_jk'
    avfile2 = './' + datapath + '/' + datafile 
    
    f = open(jkfile,'rt');
    f2 = open(avfile, 'rt')
    
    lin = 0
    for string in f:
        if fitrange[0] <= (lin % T) <= fitrange[1]:
            data = string[:-1].split(' ')
            inx_list.append(float(data[0]))
            iny_list.append(float(data[1]))
            if (lin % T) == fitrange[1]:
                exx_list.append(inx_list)
                exy_list.append(iny_list)
                inx_list = []
                iny_list = []
        lin += 1
    
    lin = 0
    
    sig_list = []
    
    for string in f2:
        if fitrange[0] <= (lin % T) <= fitrange[1]:
            data = string[:-1].split(' ')
            sig_list.append(float(data[2]))
        lin += 1
    
    lin = 0
    
    g = open(jkfile2,'rt');
    g2 = open(avfile2, 'rt')
    
    for string in g:
        if fitrange[0] <= (lin % T) <= fitrange[1]:
            data = string[:-1].split(' ')
            iny_list.append(float(data[1]))
            if (lin % T) == fitrange[1]:
                cor_list.append(iny_list)
                iny_list = []
        lin += 1
    
    lin = 0
    
    cors_list = []
    
    for string in g2:
        if fitrange[0] <= (lin % T) <= fitrange[1]:
            data = string[:-1].split(' ')
            cors_list.append(float(data[2]))
        lin += 1
    
    #=============================================================#
    #=============================================================#
    
    conf = len(exx_list)
    
    def CovarianceMatrix(x_list, y_list, n):
        vector_x = np.array(x_list[0])
        vector_ybar = np.sum(np.array(y_list), axis=0) / n
        
        ontosi_x = np.reshape(vector_x, (1,dof))
        ontosi_ybar = np.reshape(vector_ybar, (1,dof))
        
        sitoon_x = np.reshape(vector_x, (dof,1))
        sitoon_ybar = np.reshape(vector_ybar, (dof,1))
        
        cov = np.matrix(np.zeros((dof,dof)))
    
        for i in range(n):
            sitoon_yi = np.reshape(np.array(y_list[i]), (dof,1))
            ontosi_yi = np.reshape(np.array(y_list[i]), (1,dof))
            if (flag == 0):
                cov += np.dot((sitoon_yi-sitoon_ybar),(ontosi_yi-ontosi_ybar)) / n * (n - 1)
            else:
                cov += np.dot((sitoon_yi-sitoon_ybar),(ontosi_yi-ontosi_ybar)) / n 
    
        return cov
    
    
    def linfit(x_list, y_list, cov_S, n):
        #=============#
        # mapping
        #=============#
        z_list = []
        for j in range(len(x_list)):
            z_list.append(x_list[j])
        array_z = np.array(z_list)
    
        #=============#
        # Construct covariance matrix on each JK sample
        # uncorrelated approx. -> Make it be diagonal matrix
        #=============#
        sitoon_x = np.reshape(array_z, (dof,1))
        ontosi_x = np.reshape(array_z, (1,dof))
        sitoon_yi = np.reshape(np.array(y_list), (dof,1))
        ontosi_yi = np.reshape(np.array(y_list), (1,dof))
        # cov_i = np.matrix(np.dot((sitoon_yi-sitoon_ybar),(ontosi_yi-ontosi_ybar))) / n * (n - 1)
        # cov = copy.deepcopy(cov_S)
        # #cov_jk_i = np.matrix(np.diag(np.diag(cov - cov_i)))
        # cov_jk_i = np.matrix(np.diag(np.diag((n - 2) * n / (n - 1) / (n - 1) * cov - cov_i )))
        # #cov_jk_i = (n - 2) * n / (n - 1) / (n - 1) * cov - cov_i
        # #print(cov_jk_i)
        
        jkA = np.dot(np.power(ontosi_x, 0),(np.dot(cov_S.I, np.power(sitoon_x, 0))))
        jkh = np.dot(np.power(ontosi_x, 0),(np.dot(cov_S.I, sitoon_yi)))
        coeff = jkh / jkA
    
        return coeff[0,0]
    
    
    def StatisticalAnal(result, n):
        ave = np.sum(np.array(result)) / n
        var = 0
        for r in result:
            if (flag == 0):
                var += (1. - 1. / n) * ((ave - r)**2)
            else:
                var += 1. / n * ((ave - r)**2)
        err = np.sqrt(var)
        return ave, err
    
    def chisq(x_list, y_list, s_list, result):
        size = len(x_list)
        array_x = np.array(x_list)
        array_y = np.array(y_list)
        array_s = np.array(s_list)
        array_r = np.full(size, result)
    
        chisq = np.sum(np.power((array_y - array_r)/array_s, 2)) / (size - 1)
    
        return chisq
    
    def SingleExp(x, a, b):
        return a*np.exp(-b*x)
    
    def AsPolygon(ave, err):
        print(0, ave+err)
        print(dof-1, ave+err)
        print(dof-1, ave-err)
        print(0, ave-err)
    
    #=============================================================#
    #=============================================================#
    
    lin = 0
    g1 = open(jkfile2,'rt');
    ry_list = []
    for string in g1:
        data = string[:-1].split(' ')
        iny_list.append(float(data[1]))
        if(lin % T == (T - 1)):
            ry_list.append(iny_list)
            iny_list = []
        lin += 1
    
    lin = 0
    
    #=============================================================#
    #=============================================================#
    # normalization
    size = ssize + 1
    
    print('t0', t0)
    normalization_mean, normalization_err = DataNormalization.NormalizedValue(ry_list, t0, flag)
    ny_list = DataNormalization.TwoPtCtNormalizationPlusOne(ry_list, size, t0, flag)
    
    
    normfile = './' + datapath + '/normalization'
    nfile = open(normfile, 'w')
    normstr = str(normalization_mean) + ' ' + str(normalization_err) + '\n'
    nfile.write(normstr)
    nfile.close()
    
    
    cjkfile = './' + datapath + '/' + 'ndata_jk'
    cavfile = './' + datapath + '/' + 'ndata'
    
    c = open(cjkfile, 'w')
    c2 = open(cavfile, 'w')
     
    for i in range(len(ny_list)):
        for j in range(len(ny_list[0])):
            strdata = str(j) + " " + str(ny_list[i][j]) + "\n"
            c.write(strdata)
    c.close()
    
    ny_transpose = np.array(ny_list).T
    for i in range(len(ny_transpose)):
        nave, nerr = StatisticalAnal(ny_transpose[i], conf)
        strdata = str(i) + " " + str(nave) + " " + str(nerr) + "\n"
        c2.write(strdata)
    c2.close()
    
    #=============================================================#
    #=============================================================#
    
    mass_list = []
    
    nsize = len(ny_list[0])
    mjkfile = './' + datapath + '/nass_jk'
    mjk = open(mjkfile, 'w')
    for i in range(len(ny_list)):
        via = []
        for j in range(len(ny_list[0])):
            if (j % nsize < nsize - 1):
                mass = np.log(ny_list[i][j] / ny_list[i][j+1])
            else:
                mass = np.log(ny_list[i][j] / ny_list[i][0])
            via.append(mass)
            strdata = str(j) + ' ' + str(mass) + '\n'
            mjk.write(strdata)
        mass_list.append(via)
        via = []
    mjk.close()
    
    #=============================================================#
    #=============================================================#
    
    mavfile = './' + datapath + '/nass'
    mav = open(mavfile, 'w')
    mass_transpose = np.array(mass_list).T
    for i in range(len(mass_transpose)):
        mave, merr = StatisticalAnal(mass_transpose[i], conf)
        strdata = str(i) + ' ' + str(mave) + ' ' + str(merr) + '\n'
        mav.write(strdata)
    mav.close()


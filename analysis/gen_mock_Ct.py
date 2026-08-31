import sys
import numpy as np

from . import GenMockData

from config import Config

#=============================================================#
# call as a function

def main(cfg: Config):

    T = cfg.lattice.T
    conf = cfg.statistics.configuration
    ssize = cfg.analysis.size
    states = cfg.mock.state
    Bs = cfg.statistics.bootstrap
    err = cfg.mock.error

    #=============================================================#
    #=============================================================#

    ex_list = []
    av_list = []
    jk_list = []
    via0 = []
    via1 = []
    
    #=============================================================#
    #=============================================================#
    
    noise_list = GenMockData.GenNaiveConMock(conf, T, 1, states, err)
    # noise_list = GenMockData.GenNaiveExpMock(conf, T, 1, states, err)
    
    #=============================================================#
    #=============================================================#
    
    jk_list = noise_list
    GenMockData.PrintJKFile(jk_list, 'cdata', 'raw_data')
    
    #=============================================================#
    #=============================================================#
    
    jk_transpose = np.array(jk_list).T
    cavfile = './raw_data/cdata'
    c = open(cavfile, 'w')
    for i in range(len(jk_list[0])):
        ave, err = GenMockData.BootstrapAnal(jk_transpose[i], conf)
        # ave, err = GenMockData.JackknifeAnal(jk_transpose[i], conf)
        strdata = str(i) + ' ' + str(ave) + ' ' + str(err) + '\n'
        c.write(strdata)
    c.close()
    
    
    #=============================================================#
    #=============================================================#
    
    mass_list = []
    
    mjkfile = './raw_data/mass_jk'
    mjk = open(mjkfile, 'w')
    for i in range(len(jk_list)):
        via = []
        for j in range(len(jk_list[0])):
            if (j % T < T - 1):
                mass = np.log(jk_list[i][j] / jk_list[i][j+1])
            else:
                mass = np.log(jk_list[i][j] / jk_list[i][0])
            via.append(mass)
            strdata = str(j) + ' ' + str(mass) + '\n'
            mjk.write(strdata)
        mass_list.append(via)
        via = []
    mjk.close()
    
    #=============================================================#
    #=============================================================#
    
    mavfile = './raw_data/mass'
    mav = open(mavfile, 'w')
    mass_transpose = np.array(mass_list).T
    for i in range(len(mass_transpose)):
        mave, merr = GenMockData.BootstrapAnal(mass_transpose[i], conf)
        # mave, merr = GenMockData.JackknifeAnal(mass_transpose[i], conf)
        strdata = str(i) + ' ' + str(mave) + ' ' + str(merr) + '\n'
        mav.write(strdata)
    mav.close()
    
    
    #=============================================================#
    #=============================================================#
    
    size = ssize + 1
    
    cjkfile = './raw_data/ndata_jk'
    cavfile = './raw_data/ndata'
    
    c = open(cjkfile, 'w')
    c2 = open(cavfile, 'w')
    
    n_list = []
    for i in range(len(jk_list)):
        via = []
        for j in range(len(jk_list[i])):
            via.append(jk_list[i][j] / jk_list[i][0])
        n_list.append(via)
    
    n_transpose = np.array(n_list).T
    
    for i in range(len(n_list)):
        for j in range(size):
            strdata = str(j) + ' ' + str(n_list[i][j]) + '\n'
            c.write(strdata)
    c.close()
    
    for j in range(size):
        ave, err = GenMockData.BootstrapAnal(n_transpose[j], conf)
        # ave, err = GenMockData.JackknifeAnal(n_transpose[j], conf)
        strdata = str(j) + ' ' + str(ave) + ' ' + str(err) + '\n'
        c2.write(strdata)
    c2.close()
    
    #=============================================================#
    #=============================================================#
    # Bootstrap
    Jk = len(jk_list)
    
    bsx_list = []
    bsy_list = []
    
    for i in range(Bs):
        index = []
        np.random.seed(i)
        for j in range(Jk):
            bs = np.random.randint(Jk)
            index.append(bs)
    
        void_y = [0] * len(jk_list[0])
        via_y = np.array(void_y)
    
        for j in range(Jk):
            via_y = via_y + np.array(jk_list[index[j]])
        via_y = via_y / Jk
    
        bsy_list.append(list(via_y))
    
    GenMockData.PrintJKFile(bsy_list, 'bdata', 'correlator_data')
    
    bsy_transpose = np.array(bsy_list).T
    bavfile = './correlator_data/bdata'
    b = open(bavfile, 'w')
    for i in range(len(bsy_list[0])):
        ave, err = GenMockData.BootstrapAnal(bsy_transpose[i], conf)
        strdata = str(i) + ' ' + str(ave) + ' ' + str(err) + '\n'
        b.write(strdata)
    b.close()

#=============================================================#
# from command line

if __name__ == "__main__":

    import sys

    main(
        int(sys.argv[1]),
        int(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
        float(sys.argv[6]),
    )

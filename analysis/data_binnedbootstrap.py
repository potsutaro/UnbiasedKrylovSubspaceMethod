import sys
import numpy as np

#=============================================================#
#set the fitting range and target files

args = sys.argv

ex_list = []
av_list = []
via0 = []
via1 = []

datafile = args[1]
size = int(args[2])
conf = int(args[3])

binsize = int(args[4])

Bs = int(args[5])

f = open(datafile, 'rt')

lin = 0
for string in f:
    data = string[:-1].split(' ')
    via0.append(float(data[1]))
    if (lin % size) == (size - 1):
        ex_list.append(via0)
        via0 = []
    lin +=1
lin = 0

#=============================================================#
#=============================================================#

ex_transpose = np.array(ex_list).T
for i in range(len(ex_transpose)):
    ave = np.sum(ex_transpose[i]) / conf
    av_list.append(ave)

#=============================================================#
#=============================================================#

def Binning(ex, conf, binsize):
    bi = []
    
    if conf % binsize == 0:
        binconf = int(conf/binsize)
    else:
        print("check binsize")
        sys.exit()

    print(binconf, binsize)
    print(len(ex), len(ex[0]))
    for i in range(binconf):
        via = []
        for j in range(len(ex[0])):
            value = 0
            for k in range(binsize):
                value += ex[i*binsize + k][j]
            via.append(value/binsize)
        bi.append(via)

    return bi


def Bootstrap(ex, conf, Bs):

    jk = []

    for i in range(Bs):
        index = []
        np.random.seed(i)
        for j in range(conf):
            bs = np.random.randint(conf)
            index.append(bs)

        void_y = [0] * len(ex[0])
        via_y = np.array(void_y)

        for j in range(conf):
            via_y = via_y + np.array(ex[index[j]])
        via_y = via_y / conf

        jk.append(list(via_y))
        

    return jk

def BootstrapAnal(result, n):
    ave = np.sum(np.array(result)) / n
    var = 0
    for r in result:
        var += 1. / n * ((ave -r)**2)
    err = np.sqrt(var)
    return ave, err


def PrintJKFile(data_list, name):
    djkfile = './correlator_data/' + name + '_jk'
    d = open(djkfile, 'w')

    for i in range(len(data_list)):
        for j in range(len(data_list[0])):
            strdata = str(j) + ' ' + str(data_list[i][j]) + '\n'
            d.write(strdata)
    d.close

    return 0


#=============================================================#
#=============================================================#

bi_list = Binning(ex_list, conf, binsize)
jk_list = Bootstrap(bi_list, int(conf/binsize), Bs)
PrintJKFile(jk_list, 'cdata')

#=============================================================#
#=============================================================#

jk_transpose = np.array(jk_list).T
cavfile = './correlator_data/cdata'
c = open(cavfile, 'w')
for i in range(len(jk_list[0])):
    ave, err = BootstrapAnal(jk_transpose[i], Bs)
    strdata = str(i) + ' ' + str(ave) + ' ' + str(err) + '\n'
    c.write(strdata)
c.close()


#=============================================================#
#=============================================================#

mass_list = []

mjkfile = './correlator_data/mass_jk'
mjk = open(mjkfile, 'w')
for i in range(len(jk_list)):
    via = []
    for j in range(len(jk_list[0])):
        if (j % size < size - 1):
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

mavfile = './correlator_data/mass'
mav = open(mavfile, 'w')
mass_transpose = np.array(mass_list).T
for i in range(len(mass_transpose)):
    mave, merr = BootstrapAnal(mass_transpose[i], Bs)
    strdata = str(i) + ' ' + str(mave) + ' ' + str(merr) + '\n'
    mav.write(strdata)
mav.close()



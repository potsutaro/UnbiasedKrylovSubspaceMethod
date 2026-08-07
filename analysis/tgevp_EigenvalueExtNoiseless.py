import sys
import numpy as np

from . import DataExtrapolation

from config import Config

def main(cfg: Config):

    rankmin = cfg.extrapolation.rankmin
    rankmax = cfg.extrapolation.rankmax
    msize = cfg.extrapolation.msize
    
    #=============================================================#
    #=============================================================#
    # input parameters 
    
    kmax = 1 # polynomial for fitting
    
    state = 0 
    target = 'ev' # eigenvalue 
    
    
    exx_list, exy_list, inx_list, iny_list = DataExtrapolation.DataAlignmentWithVariance('./diagonalized_data', target, rankmax, rankmin, msize, state)
    
    sign = DataExtrapolation.PrintDataWithVariance('data', target, rankmax, rankmin, msize, state, exx_list, exy_list, inx_list, iny_list)
    
    #=============================================================#
    #=============================================================#
    # prepare the data
    
    conf = len(exx_list)
    ranksep = rankmax - rankmin + 1
    
    #=============================================================#
    #=============================================================#
    # lsqfitting with scipty
    
    tarresult_jk = []
    actresult_jk = []
    for i in range(conf):
        #=============#
        # initiate parameter 
        #=============#
        totalParaNum = ranksep + kmax + 1
        parameter = [float(0.0)] * totalParaNum
        for a in range(totalParaNum):
            if a < ranksep:
                parameter[a] = exx_list[i][a]
            else:
                parameter[a] = 0.0
    
        # print(parameter[rankmax:])
        # print(parameter[:rankmax])
        # print(DataExtrapolation.PredictionYR(parameter, rankmax))
        # print(DataExtrapolation.PredictionXR(parameter, rankmax))
    
        #=============#
        # variance extrapolation with least_square fitting 
        #=============#
        result = DataExtrapolation.NoiselessExtrapolation(parameter, exx_list[i], exy_list[i], i, ranksep)
        print(result[ranksep:])
        actresult_jk.append(result[ranksep:])
        tarresult_jk.append(result[-1])
    
    #=============================================================#
    #=============================================================#
    # output the target result 
    output = DataExtrapolation.PrintTarget('eigenvalue', target, tarresult_jk, rankmax, rankmin, msize, state, sign)
    
    #=============================================================#
    #=============================================================#
    # output the fitting band 
    xlimave, xlimerr = DataExtrapolation.BootstrapAnal(inx_list[0], conf)
    output = DataExtrapolation.PrintFitResult('eigenvalue', target, actresult_jk, rankmax, rankmin, msize, state, xlimave)


#=============================================================#
# from command line

if __name__ == "__main__":

    import sys

    main(
        int(sys.argv[1]),
        int(sys.argv[2]),
        int(sys.argv[3]),
    )






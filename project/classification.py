import numpy as np
def split_db_2to1(D,L,seed=0):
    nTrain=int(D.shape[1]*2/3)
    np.random.seed(seed)
    idx=np.random.permutation(D.shape[1])
    idxTrain=idx[:nTrain]
    idxTest=idx[nTrain:]
    
    DTR=D[:,idxTrain]
    DVAL=D[:,idxTest]
    LTR=L[idxTrain]
    LVAL=L[idxTest]
    
    return (DTR,LTR),(DVAL,LVAL)
    

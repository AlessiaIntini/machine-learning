import numpy as np


def mcol(array, shape):
    return array.reshape(shape, 1)


def mrow(array, shape):  # shape è d_array.shape[1]
    return array.reshape(1, shape)


def vcol(array):
    return array.reshape(1, array.size)


def vrow(array):
    return array.reshape(array.size, 1)


def load(name):
    f = open(name, 'r')
    l_array = np.array([], dtype=int)
    d_array = None

    for line in f:
        val = line.split(',')
        colData = mcol(np.array(val[0:6], dtype=float), shape=6)
        l_array = np.append(l_array, int(val[6]))
        if d_array is None:
            d_array = colData
        else:
            d_array = np.append(d_array, colData, axis=1)
    return d_array, l_array


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2 / 3)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)

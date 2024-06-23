import numpy as np


def vcol(x):
    return x.reshape((x.size, 1))


def vrow(x):
    return x.reshape((1, x.size))


def load(name):
    f = open(name, 'r')
    l_array = np.array([], dtype=int)
    d_array = None

    for line in f:
        val = line.split(',')
        colData = vcol(np.array(val[0:6], dtype=float))
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


def countLabel(L, title):
    num_zero = np.count_nonzero(L == 0)
    num_one = np.count_nonzero(L == 1)

    print("class in " + title)
    print(f"Numero di zeri fake : {num_zero}")
    print(f"Numero di uni genuine: {num_one}")

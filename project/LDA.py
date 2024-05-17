import numpy as np
import scipy.linalg

import ReadData as ut


def compute_Sv_Sb(D, L):
    num_classes = L.max() + 1
    # separate the data into classes
    D_c = [D[:, L == i] for i in range(num_classes)]
    # number of elements for each class
    n_c = [D_c[i].shape[1] for i in range(num_classes)]

    # mean for all the data
    mu = D.mean(1)
    mu = ut.vcol(mu)

    # mean for each class
    mu_c = [ut.vcol(D_c[i].mean(1)) for i in range(len(D_c))]

    S_w, S_b = 0, 0
    for i in range(num_classes):
        Dc = D_c[i] - mu_c[i]
        C_i = np.dot(Dc, Dc.T) / Dc.shape[1]
        S_w += n_c[i] * C_i
        diff = mu_c[i] - mu
        S_b += n_c[i] * np.dot(diff, diff.T)

    S_w /= D.shape[1]
    S_b /= D.shape[1]
    return S_w, S_b


def LDA_function(D, L, m):
    # compute Sw and Sb
    print("D", D)
    print("L", L)
    Sw, Sb = compute_Sv_Sb(D, L)
    print("Sw", Sw)
    print("Sb", Sb)
    # compute the eigenvalues and eigenvectors of Sw^-1*Sb
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]

    return W

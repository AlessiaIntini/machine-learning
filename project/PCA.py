import numpy as np

import ReadData as ut


def PercentageVariance(EigenValues):
    eigenvalues = EigenValues[::-1]
    ratio = np.array([])
    print("Eigenvalues:", eigenvalues)
    sum = np.sum(eigenvalues)
    for i in range(0, len(eigenvalues)):
        M = np.sum(eigenvalues[:i + 1])
        ratio = np.append(ratio, M / sum * 100)
        print("Percentuale di varianza spiegata da PC", i + 1, ":", M / sum * 100)
    return ratio


def PCA_function(D, m):
    mu = 0
    C = 0
    mu = D.mean(axis=1)  # è un vettore, cioè una matrice riga
    DC = D - ut.vcol(mu)  # per centrare i dati
    C = np.dot(DC, DC.T) / float(D.shape[1])  # matrice di covarianza

    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]  # matrice di proiezione
    print("P", P)
    # U,s,Vh=np.linalg.svd(C)
    # P=U[:,0:m]#matrice di proiezione
    return s, P

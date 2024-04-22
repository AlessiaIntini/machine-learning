import numpy as np

import ReadData as ut


def compute_mu_C(D):
    mu = ut.mcol(D.mean(1), D.mean(1).size)
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C


def logpdf_GAU_ND(X, mu, C):
    Y = []
    # get the number of features
    N = X.shape[0]
    # for each input data
    for x in X.T:
        x = ut.mcol(x, x.shape[0])
        # compute the constant term
        const = N * np.log(2 * np.pi)  # compute the second term
        logC = np.linalg.slogdet(C)[1]  # compute the third term
        mult = np.dot(np.dot((x - mu).T, np.linalg.inv(C)), (x - mu))[0, 0]
        # append the result of the function for this input data
        Y.append(-0.5 * (const + logC + mult))

    # return the result array
    return np.array(Y)


def log_likelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()


def apply_ML(D, L):
    print("D", D)
    print("L", L)

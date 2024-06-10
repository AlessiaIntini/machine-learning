import numpy as np


def mcol(array, shape):
    return array.reshape(shape, 1)


def mrow(array, shape):  # shape è d_array.shape[1]
    return array.reshape(1, shape)


def vcol(array):
    return array.reshape(1, array.shape[0])


def vrow(array):
    return array.reshape(array.shape[0], 1)


def logpdf_GAU_ND(X, mu, C):
    Y = []
    # get the number of features
    N = X.shape[0]
    # for each input data
    for x in X.T:
        x = mcol(x, x.shape[0])
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

# def logpdf_GMM(X,gmm):

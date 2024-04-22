import numpy as np
from matplotlib import pyplot as plt

import ReadData as ut
import plot


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
    for i in range(1, 7):
        plt.figure()
        plt.title("Feature {}".format(i))

        D0 = np.array(D[i - 1:i, L == 0])
        m_ML0, C_ML0 = compute_mu_C(D0)
        plot.plot_gaussian_density(D0, m_ML0, C_ML0)

        D1 = np.array(D[i - 1:i, L == 1])
        m_ML1, C_ML1 = compute_mu_C(D1)
        plot.plot_gaussian_density(D1, m_ML1, C_ML1)

        plt.legend(['False', 'True', 'False', 'True'])
        plt.show()
        print("feature", i)
        # Un valore di log-verosimiglianza più alto indica un adattamento migliore del modello ai dati.
        ll = log_likelihood(D0, m_ML0, C_ML0)
        print("False\n", ll)
        ll = log_likelihood(D1, m_ML1, C_ML1)
        print("True\n", ll)

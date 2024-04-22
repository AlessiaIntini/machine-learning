import matplotlib.pyplot as plt
import numpy as np

import utils as ut


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


if __name__ == '__main__':
    # D, L = ut.load('iris.csv')
    # pdfSol = np.load('llGAU.npy')
    # mu = calculate_mu(D, L)
    # C = calculate_cov(D, L)
    # cov_class_0 = C[0]
    # mu_class_0 = mu[0]
    # Y = logpdf_GAU_ND(D[:, L == 0], mu_class_0, cov_class_0)
    # print("Y", Y)
    # print("pdfSol", pdfSol)
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1, 1)) * 1.0
    C = np.ones((1, 1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(ut.mrow(XPlot, XPlot.shape[0]), m, C)))
    plt.show()
    pdfSol = np.load('solution/llGAU.npy')
    pdfGau = logpdf_GAU_ND(ut.mrow(XPlot, XPlot.shape[0]), m, C)
    print(np.abs(pdfGau - pdfSol).max())

    XND = np.load('solution/XND.npy')
    mu = np.load('solution/muND.npy')
    C = np.load('solution/CND.npy')
    pdfSol = np.load('solution/llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(np.abs(pdfGau - pdfSol).max())

    # ML estimates - XND
    m_ML, C_ML = compute_mu_C(XND)
    # print(m_ML)
    # print(C_ML)
    print(log_likelihood(XND, m_ML, C_ML))

    # ML estimates - X1D
    X1D = np.load('solution/X1D.npy')
    m_ML, C_ML = compute_mu_C(X1D)
    print("X1D", X1D)
    print("shape", X1D.shape)
    # print(m_ML)
    # print(C_ML)

    plt.figure()
    # è l'istogramma
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    # è la linea che fa il plot della gaussiana
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(ut.mrow(XPlot, XPlot.shape[0]), m_ML, C_ML)))
    plt.show()

    ll = log_likelihood(X1D, m_ML, C_ML)
    print(ll)
    # print(log_likelihood(X1D, m_ML, C_ML))
    # # Trying other values
    # print(log_likelihood(X1D, np.array([[1.0]]), np.array([[2.0]])))
    # print(log_likelihood(X1D, np.array([[0.0]]), np.array([[1.0]])))
    # print(log_likelihood(X1D, np.array([[2.0]]), np.array([[6.0]])))  # Values close to ML estimates
    #

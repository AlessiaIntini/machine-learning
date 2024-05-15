import matplotlib.pyplot as plt
import numpy as np
import scipy


def vcol(x):
    return x.reshape((x.size, 1))


def vrow(x):
    return x.reshape((1, x.size))


def load(name):
    f = open(name, 'r')
    l_array = np.array([], dtype=int)
    d_array = None
    flowerType = -1
    for line in f:
        val = line.split(',')
        colData = np.array(val[0:4], dtype=float).reshape(4, 1)
        if val[4] == 'Iris-setosa\n':
            flowerType = 0
        elif val[4] == 'Iris-versicolor\n':
            flowerType = 1
        elif val[4] == 'Iris-virginica\n':
            flowerType = 2
        l_array = np.append(l_array, flowerType)
        if d_array is None:
            d_array = colData
        else:
            d_array = np.append(d_array, colData, axis=1)
    return d_array, l_array


def PercentageVariance(EigenValues):
    eigenvalues = EigenValues[::-1]
    # print("Eigenvalues:",eigenvalues)
    sum = np.sum(eigenvalues)
    for i in range(0, len(eigenvalues)):
        M = np.sum(eigenvalues[:i + 1])
        ratio = M / sum * 100
        # print("Percentuale di varianza spiegata da PC",i+1,":",ratio)
    return M


def hist(d_array, l_array, caratteristics, label, bins=10):
    M0 = (l_array == 0)
    D0 = d_array[caratteristics, M0]
    plt.hist(D0, density=True, label='Iris-setosa', alpha=0.5, bins=bins)
    M1 = (l_array == 1)
    D1 = d_array[caratteristics, M1]
    plt.hist(D1, density=True, label='Iris-versicolor', alpha=0.5, bins=bins)
    M2 = (l_array == 2)
    D2 = d_array[caratteristics, M2]
    plt.hist(D2, density=True, label='Iris-virginica', alpha=0.5, bins=bins)
    plt.xlabel(label)
    plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    # plt.show()


def scatter(d_array, l_array, component1, component2, label1, label2):
    M0 = (l_array == 0)
    plt.scatter(d_array[component1, M0], d_array[component2, M0], label='Iris-setosa', alpha=0.5)
    M1 = (l_array == 1)
    plt.scatter(d_array[component1, M1], d_array[component2, M1], label='Iris-versicolor', alpha=0.5)
    M2 = (l_array == 2)
    plt.scatter(d_array[component1, M2], d_array[component2, M2], label='Iris-virginica', alpha=0.5)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

    plt.show()


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


def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])

    return mu, C


def logpdf_GAU_ND(X, mu, C):
    Y = []
    # get the number of features
    N = X.shape[0]
    # for each input data
    for x in X.T:
        x = vcol(x)
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


def compute_mu_c_MVG(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L == lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams


def compute_mu_C_Tied(D, L):
    labelSet = set(L)
    hParams = {}
    hMeans = {}
    CGlobal = 0
    for lab in labelSet:
        DX = D[:, L == lab]
        mu, C_class = compute_mu_C(DX)
        # DX.shape[1] è il numero di campioni di quella classe
        CGlobal += C_class * DX.shape[1]
        hMeans[lab] = mu
    # qui viene diviso per il numero totale di campioni
    CGlobal = CGlobal / D.shape[1]
    # viene semplicemente assegnato lo stesso valore di covarianza a tutte le classi
    for lab in labelSet:
        hParams[lab] = (hMeans[lab], CGlobal)
    return hParams


def compute_mu_C_Naive(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L == lab]
        mu, C = compute_mu_C(DX)
        # C moltiplicato per la matrice identità
        hParams[lab] = (mu, C * np.eye(D.shape[0]))
    return hParams


def compute_log_likelihood(D, hParams):
    S = np.zeros((len(hParams), D.shape[1]))
    for lab in range(S.shape[0]):
        S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])
    return S


def logpdf_GAU_ND(X, mu, C):
    Y = []
    # get the number of features
    N = X.shape[0]
    # for each input data
    for x in X.T:
        x = vcol(x)
        # compute the constant term
        const = N * np.log(2 * np.pi)  # compute the second term
        logC = np.linalg.slogdet(C)[1]  # compute the third term
        mult = np.dot(np.dot((x - mu).T, np.linalg.inv(C)), (x - mu))[0, 0]
        # append the result of the function for this input data
        Y.append(-0.5 * (const + logC + mult))

    # return the result array
    return np.array(Y)


def predict_labels(DVAL, TH, LLR, class1, class2):
    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    PVAL[LLR >= TH] = class2
    PVAL[LLR < TH] = class1
    return PVAL


def error_rate(PVAL, LVAL):
    return ((PVAL != LVAL).sum() / float(LVAL.size) * 100)


def compute_logPosterior(S_logLikelihood, v_prior):
    # probabilità congiunta
    SJoint = S_logLikelihood + vcol(np.log(v_prior))
    # probabilita marginale che è uguale al prodotto delle probabilità congiunte
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    # probabilità a posteriori, sottrai la probabilità marginale dalla probabilità congiunta in modo che tutti abbiamo probabilità massimo 1
    SPost = SJoint - SMarginal
    return SPost

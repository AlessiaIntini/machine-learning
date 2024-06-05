# import math
#
# import numpy as np
# from scipy.optimize import fmin_l_bfgs_b
#
#
# # Definisci la funzione
# def f(x):
#     y, z = x
#     return (y + 3) ** 2 + math.sin(y) + (z + 1) ** 2
#
#
# # Definisci il gradiente della funzione
# def fprime(x):
#     y, z = x
#     df_dy = 2 * (y + 3) + math.cos(y)  # derivata parziale rispetto a y
#     df_dz = 2 * (z + 1)  # derivata parziale rispetto a z
#     return np.array([df_dy, df_dz])
#
#
# # Punto di partenza
# x0 = np.array([0, 0])
#
# # Chiamiamo fmin_l_bfgs_b
# result = fmin_l_bfgs_b(f, x0, fprime, approx_grad=True, iprint=1)
#
# print(result)
import numpy as np
import scipy.optimize
import sklearn.datasets


def vcol(x):
    return x.reshape((x.size, 1))


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


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


def trainLogReg(DTR, LTR, lbd, prior=0.5, prior_weighted=False):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        ZTR = 2 * LTR - 1
        reg = 0.5 * lbd * np.linalg.norm(w) ** 2
        exp = (np.dot(w.T, DTR) + b)
        avg_risk = (np.logaddexp(0, -exp * ZTR)).mean()
        return reg + avg_risk

    def logreg_obj_prior_weighted(v):
        w, b = v[0:-1], v[-1]
        ZTR = 2 * LTR - 1
        reg = 0.5 * lbd * np.linalg.norm(w) ** 2
        exp = (np.dot(w.T, DTR) + b)
        avg_risk_0 = np.logaddexp(0, -exp[LTR == 0] * ZTR[LTR == 0]).mean() * (1 - prior)
        avg_risk_1 = np.logaddexp(0, -exp[LTR == 1] * ZTR[LTR == 1]).mean() * prior
        return reg + avg_risk_0 + avg_risk_1

    x0 = np.zeros(DTR.shape[0] + 1)
    xf = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj_prior_weighted if prior_weighted else logreg_obj, x0=x0,
                                      approx_grad=True, iprint=0)[0]

    return xf


def calculate_sllr(x, w, b, pi_emp=0.5):
    s = np.dot(w.T, x) + b
    sllr = s - np.log(pi_emp / (1 - pi_emp))
    return sllr


def calculate_error_rate(predictions, targets):
    # return np.mean(predictions != targets)
    return ((predictions != targets).sum() / float(targets.size) * 100)


def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = np.argsort(llr)
    llrSorted = llr[llrSorter]  # We sort the llrs
    classLabelsSorted = classLabels[llrSorter]  # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []

    nTrue = (classLabelsSorted == 1).sum()
    nFalse = (classLabelsSorted == 0).sum()
    nFalseNegative = 0  # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse

    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)

    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    # The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    # Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    # Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx + 1] != llrSorted[
            idx]:  # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])

    return np.array(PfnOut), np.array(PfpOut), np.array(
        thresholdsOut)  # we return also the corresponding thresholds


def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (
            1 - prior) * Cfp)  # We exploit broadcasting to compute all DCFs for all thresholds
    idx = np.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]


def computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=False):
    Pfn = confusionMatrix[0, 1] / (confusionMatrix[0, 1] + confusionMatrix[1, 1])
    Pfp = confusionMatrix[1, 0] / (confusionMatrix[1, 0] + confusionMatrix[0, 0])
    bayesError = prior * Pfn * Cfn + (1 - prior) * Pfp * Cfp
    if normalize:
        return bayesError / np.minimum(prior * Cfn, (1 - prior) * Cfp)
    return bayesError


def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = np.zeros((nClasses, nClasses), dtype=np.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M


def compute_minDCF_actDCF(LTR, xf, LVAL, DVAL, pi_emp=0.5, Cfn=1, Cfp=1, prior=0.5):
    w = xf[:-1]
    b = xf[-1]
    sval = np.dot(w.T, DVAL) + b
    PVAL = (sval > 0) * 1
    error_rate = calculate_error_rate(PVAL, LVAL)
    pi_emp = (LTR == 1).sum() / LTR.size
    sllr = np.array([calculate_sllr(x, w, b, pi_emp) for x in DVAL.T])
    predictions = (sllr > 0).astype(int)
    print("Error rate:", error_rate, "%")
    minDCF = compute_minDCF_binary_fast(sllr, LVAL, prior, Cfn, Cfp)
    print("minDCF:", minDCF)
    confusionMatrix = compute_confusion_matrix(predictions, LVAL)
    actDCF = computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=True)

    print("actDCF:", actDCF)


if __name__ == '__main__':
    # D, L = load('iris.csv')
    D, L = load_iris_binary()
    # D = D[:, L != 0]  # We remove setosa from D
    # L = L[L != 0]  # We remove setosa from L
    # L[L == 2] = 0  # We assign label 0 to virginica (was label 2) return D, L

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Binary logistic regression with prior = 0.5 and prior_weighted false
    print("Binary logistic regression with prior = 0.5 and prior_weighted false")
    xf = trainLogReg(DTR, LTR, 10 ** -3)
    # print(xf)
    compute_minDCF_actDCF(LTR, xf, LVAL, DVAL)

    xf = trainLogReg(DTR, LTR, 10 ** -1)
    # print(xf)
    compute_minDCF_actDCF(LTR, xf, LVAL, DVAL)

    xf = trainLogReg(DTR, LTR, 1.0)
    # print(xf)
    compute_minDCF_actDCF(LTR, xf, LVAL, DVAL)

    # Binary logistic regression with prior = 0.8 and prior_weighted true
    print("Binary logistic regression with prior = 0.8 and prior_weighted true")
    xf = trainLogReg(DTR, LTR, 10 ** -3, 0.8, True)
    compute_minDCF_actDCF(LTR, xf, LVAL, DVAL)
    # print("prior=p_emp")
    # xf = trainLogReg(DTR, LTR, 10 ** -3, 0.5, True)
    # compute_minDCF_actDCF(xf, LVAL, DVAL)

    xf = trainLogReg(DTR, LTR, 10 ** -1, 0.8, True)
    compute_minDCF_actDCF(LTR, xf, LVAL, DVAL)

    xf = trainLogReg(DTR, LTR, 1.0, 0.8, True)
    compute_minDCF_actDCF(LTR, xf, LVAL, DVAL)

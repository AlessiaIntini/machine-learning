import numpy as np
import sklearn.datasets

import SVM


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


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


if __name__ == '__main__':
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    hParams = [{'K': 1, 'C': 0.1}, {'K': 1, 'C': 1.0}, {'K': 1, 'C': 10.0},
               {'K': 10, 'C': 0.1}, {'K': 10, 'C': 1.0}, {'K': 10, 'C': 10.0}]

    for hParam in hParams:
        svm = SVM.SVM(hParam, kernel=None, prior=0.5)
        svmReturn = svm.train(DTR, LTR)
        print("primal solution", svmReturn.primal)
        # print("predict", svmReturn.predict(DVAL, labels=True))
        predictions = svmReturn.predict(DVAL, labels=True)
        print("error rate", calculate_error_rate(predictions, LVAL))
        llr = svmReturn.predict(DVAL)
        minDCF = compute_minDCF_binary_fast(llr, LVAL, 0.5, 1, 1)
        print("minDCF", minDCF)
        confusionMatrix = compute_confusion_matrix(predictions, LVAL)
        actDCF = computeDCF_Binary(confusionMatrix, 0.5, 1, 1, normalize=True)
        print("actDCF", actDCF)
        print("duality gap", svmReturn.compute_duality_gap())

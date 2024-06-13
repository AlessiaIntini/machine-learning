import numpy as np
import scipy
import json
import sklearn.datasets
from laboratori.lab10.GMM import GMM


def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)


def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]


def mcol(array, shape):
    return array.reshape(shape, 1)


def mrow(array, shape):  # shape è d_array.shape[1]
    return array.reshape(1, shape)


def vcol(array):
    return array.reshape(1, array.shape[0])


def vrow(array):
    return array.reshape(array.shape[0], 1)


def compute_mu_C(D):
    mu = mcol(D.mean(1), D.mean(1).size)
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C


def logpdf_GAU_ND_fast(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5 * x.shape[0] * np.log(np.pi * 2) - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * (
            (x - mu) * (P @ (x - mu))).sum(0)


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


def logpdf_GMM(X, gmm):
    return


def compute_logPosterior(S_logLikelihood, v_prior):
    # probabilità congiunta
    SJoint = S_logLikelihood + vcol(np.log(v_prior))
    # probabilita marginale che è uguale al prodotto delle probabilità congiunte
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    # probabilità a posteriori, sottrai la probabilità marginale dalla probabilità congiunta in modo che tutti abbiamo probabilità massimo 1
    SPost = SJoint - SMarginal
    return SPost


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


def computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=False):
    Pfn = confusionMatrix[0, 1] / (confusionMatrix[0, 1] + confusionMatrix[1, 1])
    Pfp = confusionMatrix[1, 0] / (confusionMatrix[1, 0] + confusionMatrix[0, 0])
    bayesError = prior * Pfn * Cfn + (1 - prior) * Pfp * Cfp
    if normalize:
        return bayesError / np.minimum(prior * Cfn, (1 - prior) * Cfp)
    return bayesError


def compute_Pfn_Pfp_allThresholds(llr, classLabels):
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
    # Pfn.append(1.0) # Corresponds to the np.inf threshold, all samples are assigned to class 0
    # Pfp.append(0.0) # Corresponds to the np.inf threshold, all samples are assigned to class 0
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


def compute_minDCF_binary(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (
            1 - prior) * Cfp)  # We exploit broadcasting to compute all DCFs for all thresholds
    idx = np.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]


def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = np.zeros((nClasses, nClasses), dtype=np.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M


if __name__ == '__main__':
    D = np.load('data/GMM_data_4D.npy')
    gmm_par = load_gmm('data/GMM_4D_3G_init.json')
    gmm = GMM()

    S, longdens = gmm.logpdf_GMM(D, gmm_par)
    solution_ll = np.load('data/GMM_4D_3G_init_ll.npy')

    resulGMM_EM, final_log = gmm.GMM_algorithm_EM(D, gmm_par)
    print("final_log", final_log)

    resultGMM_LBG, final_log = gmm.GMM_algorithm_LBG(D, 0.1, 4)
    print("resultGMM_LBG", resultGMM_LBG)

    solution_LBG = load_gmm('data/GMM_4D_4G_EM_LBG.json')

    print("solution", solution_LBG)

    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    covTypes = ['Full', 'Diag', 'Tied']
    nComponents = [1, 2, 4, 8, 16]
    for nc in nComponents:
        # print("nComponents ", nc)
        for covType in covTypes:
            # print(" covType is ", covType)
            gmm = GMM(alpha=0.1, nComponents=nc, psi=0.01, covType=covType)
            gmm.train_NoBinary(DTR, LTR)
            predictions = gmm.predict_NoBinary(DVAL, True)
            # print("    Error rate: %.1f%%" % calculate_error_rate(predictions, LVAL))

    D = np.load('data/ext_data_binary.npy')
    L = np.load('data/ext_data_binary_labels.npy')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    for nc in nComponents:
        print("nComponents ", nc)
        for covType in covTypes:
            print(" covType is ", covType)
            gmm = GMM(alpha=0.1, nComponents=nc, psi=0.01, covType=covType)
            gmm.train(DTR, LTR)
            llr = gmm.predict(DVAL)
            minDCF = compute_minDCF_binary(llr, LVAL, 0.5, 1, 1)
            print("minDCF", minDCF)
            predictions = gmm.predict(DVAL, labels=True)
            # th = -np.log((0.5 * 1) / ((1 - 0.5) * 1))
            # Slval = np.int32(llr > th)
            confusionMatrix = compute_confusion_matrix(predictions, LVAL)
            actDCF = computeDCF_Binary(confusionMatrix, 0.5, 1, 1, normalize=True)
            print("actDCF", actDCF)

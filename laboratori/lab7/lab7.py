import matplotlib.pyplot as plt
import numpy
import scipy


def vcol(x):
    return x.reshape((x.size, 1))


def vrow(x):
    return x.reshape((1, x.size))


def compute_posteriors(log_clas_conditional_ll, prior_array):
    logJoint = log_clas_conditional_ll + vcol(numpy.log(prior_array))
    logPost = logJoint - scipy.special.logsumexp(logJoint, 0)
    return numpy.exp(logPost)


def compute_optimal_Bayes(posterior, costMatrix):
    expectedCosts = costMatrix @ posterior
    return numpy.argmin(expectedCosts, 0)


def uniform_cost_matrix(nClasses):
    return numpy.ones((nClasses, nClasses)) - numpy.eye(nClasses)


def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = numpy.zeros((nClasses, nClasses), dtype=numpy.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M


def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -numpy.log((prior * Cfn) / ((1 - prior) * Cfp))
    return numpy.int32(llr > th)


def computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=False):
    Pfn = confusionMatrix[0, 1] / (confusionMatrix[0, 1] + confusionMatrix[1, 1])
    Pfp = confusionMatrix[1, 0] / (confusionMatrix[1, 0] + confusionMatrix[0, 0])
    bayesError = prior * Pfn * Cfn + (1 - prior) * Pfp * Cfp
    if normalize:
        return bayesError / numpy.minimum(prior * Cfn, (1 - prior) * Cfp)
    return bayesError


def computeDCF_Multiclass(confusionMatrix, prior, costMatrix, normalize=False):
    errorRates = confusionMatrix / vrow(confusionMatrix.sum(0))
    bayesError = ((errorRates * costMatrix).sum(0) * prior.ravel()).sum()
    if normalize:
        return bayesError / numpy.min(costMatrix @ vcol(prior))
    return bayesError


def compute_minDCF_binary(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    llrSorted = llr

    # concatena tutte le soglie llr con -inf e +inf
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), llrSorted, numpy.array([numpy.inf])])
    dcfMin = None
    dcfTh = None
    for th in thresholds:
        predictedLabels = numpy.int32(llr > th)
        M = compute_confusion_matrix(predictedLabels, classLabels)
        dcf = computeDCF_Binary(M, prior, Cfn, Cfp, normalize=True)
        if dcfMin is None or dcf < dcfMin:
            dcfMin = dcf
            dcfTh = th
    if returnThreshold:
        return dcfMin, dcfTh
    else:
        return dcfMin


def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = numpy.argsort(llr)
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
    llrSorted = numpy.concatenate([-numpy.array([numpy.inf]), llrSorted])

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

    return numpy.array(PfnOut), numpy.array(PfpOut), numpy.array(
        thresholdsOut)  # we return also the corresponding thresholds


def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / numpy.minimum(prior * Cfn, (
            1 - prior) * Cfp)  # We exploit broadcasting to compute all DCFs for all thresholds
    idx = numpy.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]


def plotROC(Ptp, Pfp):
    plt.figure()
    plt.plot(Pfp, Ptp)
    plt.xlabel('Pfp')
    plt.ylabel('Ptp')
    plt.title('Curva Ptp vs Pfp')
    plt.xlim(min(Pfp), max(Pfp))
    plt.ylim(min(Ptp), max(Ptp))
    plt.show()


if __name__ == '__main__':
    commedia_ll = numpy.load('data/commedia_ll.npy')
    commedia_labels = numpy.load('data/commedia_labels.npy')
    commedia_posteriors = compute_posteriors(commedia_ll, numpy.ones(3) / 3.0)
    commedia_predictions = compute_optimal_Bayes(commedia_posteriors, uniform_cost_matrix(3))

    print(compute_confusion_matrix(commedia_predictions, commedia_labels))

    # binary task optimal decisions
    print("Binary task")
    commedia_llr_binary = numpy.load('data/commedia_llr_infpar.npy')
    commedia_labels_binary = numpy.load('data/commedia_labels_infpar.npy')

    for prior, Cfn, Cfp in [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:
        print("Prior", prior, "Cfn", Cfn, "Cfp", Cfp)
        commedia_predictions_binary = compute_optimal_Bayes_binary_llr(commedia_llr_binary, prior, Cfn, Cfp)
        confusionMatrix = compute_confusion_matrix(commedia_predictions_binary, commedia_labels_binary)
        print(confusionMatrix)
        print("Compute DCF without normalization")
        print(computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp))
        print("Compute DCF with normalization")
        print(computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=True))

        minDCF, minDCFThreshold = compute_minDCF_binary(commedia_llr_binary, commedia_labels_binary, prior, Cfn,
                                                        Cfp, returnThreshold=True)
        print("minDFC", minDCF, "minThreshold", minDCFThreshold)

        pn, minDCFThreshold = compute_minDCF_binary_fast(commedia_llr_binary, commedia_labels_binary, prior, Cfn, Cfp,
                                                         returnThreshold=True)
        print("minDFC", pn, "minThreshold", minDCFThreshold)

    # ROC
    Pfn, Pfp, _ = compute_Pfn_Pfp_allThresholds_fast(commedia_llr_binary, commedia_labels_binary)
    Ptp = 1 - Pfn
    plotROC(Ptp, Pfp)

    print(commedia_llr_binary)
    # Bayes error plot
    # 1. genera valori di p~ da -3 a 3
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    # 2. calcola π~ da p~
    effPrior = 1 / (1 + numpy.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for prior in effPrior:
        # 3. calcoli la llr per ogni prior
        commedia_predictions_binary = compute_optimal_Bayes_binary_llr(commedia_llr_binary, prior, 1.0, 1.0)
        # confusion matrix
        confusionMatrix = compute_confusion_matrix(commedia_predictions_binary, commedia_labels_binary)
        # 4. calcola DCF e minDCF
        actDCF.append(computeDCF_Binary(confusionMatrix, prior, 1, 1, normalize=True))
        minDCF.append(compute_minDCF_binary_fast(commedia_llr_binary, commedia_labels_binary, prior, 1, 1))

    plt.figure(1)
    plt.plot(effPriorLogOdds, actDCF, label='DCF eps 0.001', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='minDCF eps 0.001', color='b')
    plt.ylim([0, 1.1])

    commedia_llr_binary = numpy.load('data/commedia_llr_infpar_eps1.npy')
    commedia_labels_binary = numpy.load('data/commedia_labels_infpar_eps1.npy')
    actDCF = []
    minDCF = []
    for prior in effPrior:
        # 3. calcoli la llr per ogni prior
        commedia_predictions_binary = compute_optimal_Bayes_binary_llr(commedia_llr_binary, prior, 1.0, 1.0)
        # confusion matrix
        confusionMatrix = compute_confusion_matrix(commedia_predictions_binary, commedia_labels_binary)
        # 4. calcola DCF e minDCF
        actDCF.append(computeDCF_Binary(confusionMatrix, prior, 1.0, 1.0, normalize=True))
        minDCF.append(compute_minDCF_binary_fast(commedia_llr_binary, commedia_labels_binary, prior, 1.0, 1.0))

    plt.plot(effPriorLogOdds, actDCF, label='DCF eps 1.0', color='y')
    plt.plot(effPriorLogOdds, minDCF, label='minDCF eps 1.0', color='c')
    plt.ylim([0, 1.1])

    plt.legend()
    plt.show()

    # MULTICLASS
    prior = numpy.array([0.3, 0.4, 0.3])
    costMatrix = numpy.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

    commedia_ll = numpy.load('data/commedia_ll.npy')
    commedia_labels = numpy.load('data/commedia_labels.npy')

    print('Eps 0.001')
    commedia_posteriors = compute_posteriors(commedia_ll, prior)
    commedia_predictions = compute_optimal_Bayes(commedia_posteriors, costMatrix)
    confusionMatrix = compute_confusion_matrix(commedia_predictions, commedia_labels)
    print(confusionMatrix)
    print('Emprical Bayes risk: %.3f' % computeDCF_Multiclass(
        confusionMatrix, prior, costMatrix, normalize=False))
    print('Normalized emprical Bayes risk: %.3f' % computeDCF_Multiclass(
        confusionMatrix, prior, costMatrix, normalize=True))

    commedia_llr_binary = numpy.load('data/commedia_ll_eps1.npy')
    commedia_labels_binary = numpy.load('data/commedia_labels_eps1.npy')

    print('Eps 1.0')
    commedia_posteriors = compute_posteriors(commedia_llr_binary, prior)
    commedia_predictions = compute_optimal_Bayes(commedia_posteriors, costMatrix)
    confusionMatrix = compute_confusion_matrix(commedia_predictions, commedia_labels_binary)
    print(confusionMatrix)
    print('Emprical Bayes risk: %.3f' % computeDCF_Multiclass(
        confusionMatrix, prior, costMatrix, normalize=False))
    print('Normalized emprical Bayes risk: %.3f' % computeDCF_Multiclass(
        confusionMatrix, prior, costMatrix, normalize=True))

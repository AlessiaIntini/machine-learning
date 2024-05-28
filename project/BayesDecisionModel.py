import numpy


def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -numpy.log((prior * Cfn) / ((1 - prior) * Cfp))
    return numpy.int32(llr > th)


def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = numpy.zeros((nClasses, nClasses), dtype=numpy.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M


def computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=False):
    Pfn = confusionMatrix[0, 1] / (confusionMatrix[0, 1] + confusionMatrix[1, 1])
    Pfp = confusionMatrix[1, 0] / (confusionMatrix[1, 0] + confusionMatrix[0, 0])
    bayesError = prior * Pfn * Cfn + (1 - prior) * Pfp * Cfp
    if normalize:
        return bayesError / numpy.minimum(prior * Cfn, (1 - prior) * Cfp)
    return bayesError


def compute_Pfn_Pfp_allThresholds(llr, classLabels):
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


def compute_minDCF_binary(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / numpy.minimum(prior * Cfn, (
            1 - prior) * Cfp)  # We exploit broadcasting to compute all DCFs for all thresholds
    idx = numpy.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]


def ApplyDecisionModel(ArrayApplications, DTR_mvg, DTR_Tied, DTR_naive, LVAL, m=-1):
    best_DCF_norm = {'MVG': float('inf'), 'naive': float('inf'), 'tied': float('inf')}
    best_DCF_min = {'MVG': float('inf'), 'naive': float('inf'), 'tied': float('inf')}
    best_DCF = {'MVG': float('inf'), 'naive': float('inf'), 'tied': float('inf')}
    best_m = {'MVG': None, 'naive': None, 'tied': None}
    for prior, Cfn, Cfp in ArrayApplications:
        print("Prior: {:.2f}, Cost False Negative: {:.2f}, Cost False Positive: {:.2f}".format(prior, Cfn, Cfp))
        # MVG
        print("MVG")
        prediction_matrix = compute_optimal_Bayes_binary_llr(DTR_mvg, prior, Cfn, Cfp)
        confusionMatrix = compute_confusion_matrix(prediction_matrix, LVAL)
        print("Confusion Matrix:\n", confusionMatrix)
        DCF_MVG = computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=False)
        print("DCF: {:.6f}".format(DCF_MVG))
        DCF_norm_MVG = computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=True)
        print("DCF normalized: {:.6f}".format(DCF_norm_MVG))
        minDCF_MVG, minDCFThreshold = compute_minDCF_binary(DTR_mvg, LVAL, prior, Cfn, Cfp,
                                                            returnThreshold=True)
        print("minDFC", minDCF_MVG)

        # TIED
        print("TIED")
        prediction_matrix = compute_optimal_Bayes_binary_llr(DTR_Tied, prior, Cfn, Cfp)
        confusionMatrix = compute_confusion_matrix(prediction_matrix, LVAL)
        print("Confusion Matrix:\n", confusionMatrix)
        DCF_tied = computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=False)
        print("DCF: {:.6f}".format(DCF_tied))
        DCF_norm_tied = computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=True)
        print("DCF normalized: {:.6f}".format(DCF_norm_tied))
        minDCF_tied, minDCFThreshold = compute_minDCF_binary(DTR_Tied, LVAL, prior, Cfn, Cfp,
                                                             returnThreshold=True)
        print("minDFC", minDCF_tied)

        # NAIVE
        print("NAIVE")
        prediction_matrix = compute_optimal_Bayes_binary_llr(DTR_naive, prior, Cfn, Cfp)
        confusionMatrix = compute_confusion_matrix(prediction_matrix, LVAL)
        print("Confusion Matrix:\n", confusionMatrix)
        DCF_naive = computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=False)
        print("DCF: {:.6f}".format(DCF_naive))
        DCF_norm_naive = computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=True)
        print("DCF normalized: {:.6f}".format(DCF_norm_naive))
        minDCF_naive, minDCFThreshold = compute_minDCF_binary(DTR_naive, LVAL, prior, Cfn, Cfp,
                                                              returnThreshold=True)
        print("minDFC", minDCF_naive)

        if prior == 0.1 and m != -1:
            if DCF_norm_MVG < best_DCF_norm['MVG']:
                best_DCF_norm['MVG'] = DCF_norm_MVG
                best_DCF['MVG'] = DCF_MVG
                best_DCF_min['MVG'] = minDCF_MVG
                best_m['MVG'] = m
            if DCF_norm_naive < best_DCF_norm['naive']:
                best_DCF_norm['naive'] = DCF_norm_naive
                best_DCF['naive'] = DCF_naive
                best_DCF_min['naive'] = minDCF_naive
                best_m['naive'] = m
            if DCF_norm_tied < best_DCF_norm['tied']:
                best_DCF_norm['tied'] = DCF_norm_tied
                best_DCF['tied'] = DCF_tied
                best_DCF_min['tied'] = minDCF_tied
                best_m['tied'] = m

    return best_DCF_norm, best_m, best_DCF, best_DCF_min

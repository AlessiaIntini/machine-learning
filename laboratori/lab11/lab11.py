import numpy as np
import scipy.optimize
import LLR_Clf as LLR


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


def calculate_sllr(x, w, b, pi_emp=0.5):
    s = np.dot(w.T, x) + b
    sllr = s - np.log(pi_emp / (1 - pi_emp))
    return sllr


def compute_minDCF_actDCF(xf, LVAL, DVAL, pi_emp, Cfn=1, Cfp=1, prior=0.5):
    w = xf[:-1]
    b = xf[-1]
    sval = np.dot(w.T, DVAL) + b
    th = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
    predictedLabels = np.int32(sval > th)
    # sValLLR = sval - np.log(pi_emp / (1 - pi_emp))
    minDCF = compute_minDCF_binary(sval, LVAL, prior, Cfn, Cfp)
    print("minDCF:", minDCF)
    confusionMatrix = compute_confusion_matrix(predictedLabels, LVAL)
    actDCF = computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=True)
    print("actDCF:", actDCF)
    return minDCF, actDCF


if __name__ == '__main__':
    score1 = np.load('data/scores_1.npy')
    score2 = np.load('data/scores_2.npy')
    labels = np.load('data/labels.npy')

    eval_score1 = np.load('data/eval_scores_1.npy')
    eval_score2 = np.load('data/eval_scores_2.npy')
    eval_labels = np.load('data/eval_labels.npy')

    minDCF1 = compute_minDCF_binary(score1, labels, 0.2, 1, 1)
    minDCF2 = compute_minDCF_binary(score2, labels, 0.2, 1, 1)
    # print("system1", minDCF1)
    # print("system2", minDCF2)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(score1 > th)
    Slval2 = np.int32(score2 > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, labels)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)
    # print("actDCF1:", actDCF1)
    confusionMatrix2 = compute_confusion_matrix(Slval2, labels)
    actDCF2 = computeDCF_Binary(confusionMatrix2, 0.2, 1, 1, normalize=True)
    # print("actDCF2:", actDCF2)

    SCAL1, SVAL1 = score1[::3], np.hstack([score1[1::3], score1[2::3]])
    SCAL2, SVAL2 = score2[::3], np.hstack([score2[1::3], score2[2::3]])
    labels_cal, labels_val = labels[::3], np.hstack([labels[1::3], labels[2::3]]),

    print("calibration")
    minDCF1 = compute_minDCF_binary(SCAL1, labels_cal, 0.2, 1, 1)
    minDCF2 = compute_minDCF_binary(SCAL2, labels_cal, 0.2, 1, 1)
    print("system1", minDCF1)
    print("system2", minDCF2)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(SCAL1 > th)
    Slval2 = np.int32(SCAL2 > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, labels_cal)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)
    print("actDCF1:", actDCF1)
    confusionMatrix2 = compute_confusion_matrix(Slval2, labels_cal)
    actDCF2 = computeDCF_Binary(confusionMatrix2, 0.2, 1, 1, normalize=True)
    print("actDCF2:", actDCF2)

    print("validation")
    minDCF1 = compute_minDCF_binary(SVAL1, labels_val, 0.2, 1, 1)
    minDCF2 = compute_minDCF_binary(SVAL2, labels_val, 0.2, 1, 1)
    print("system1", minDCF1)
    print("system2", minDCF2)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(SVAL1 > th)
    Slval2 = np.int32(SVAL2 > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, labels_val)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)
    print("actDCF1:", actDCF1)
    confusionMatrix2 = compute_confusion_matrix(Slval2, labels_val)
    actDCF2 = computeDCF_Binary(confusionMatrix2, 0.2, 1, 1, normalize=True)
    print("actDCF2:", actDCF2)

    SCAL1_reshape = SCAL1.reshape(1, -1)
    SCAL2_reshape = SCAL2.reshape(1, -1)
    SVAL1_reshape = SVAL1.reshape(1, -1)
    SVAL2_reshape = SVAL2.reshape(1, -1)

    print("SCAL, calibration system1")
    # raw scores SCAL system1
    print("Raw scores")
    minDCF = compute_minDCF_binary(SCAL1, labels_cal, 0.2, 1, 1)
    print("minDCF:", minDCF)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(SCAL1 > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, labels_cal)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)
    print("actDCF1:", actDCF1)
    # calibration result SCAL system1
    print("Calibration scores")
    score_labels = np.hstack(labels_cal)
    Nsamples = SCAL1.shape[0]
    p = 0.2
    llr = LLR.LinearLogisticRegression(0.1, prior=p)
    x = llr.train(SCAL1_reshape, score_labels)
    alpha, beta = x[0:-1], x[-1]
    k_scores_cal = (alpha * SCAL1_reshape + beta - np.log(p / (1 - p))).reshape(Nsamples, )
    minDCF = compute_minDCF_binary(k_scores_cal, score_labels, 0.2, 1, 1)
    print("minDCF:", minDCF)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(k_scores_cal > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, score_labels)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)
    print("actDCF1:", actDCF1)

    print("SVAL, validation system1")
    # raw scores SCAL system1
    print("Raw scores")
    minDCF = compute_minDCF_binary(SVAL1, labels_val, 0.2, 1, 1)
    print("minDCF:", minDCF)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(SVAL1 > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, labels_val)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)
    print("actDCF1:", actDCF1)
    # calibration result SCAL system1
    print("Calibration scores")
    score_labels = np.hstack(labels_val)
    Nsamples = SVAL1.shape[0]
    p = 0.2
    llr = LLR.LinearLogisticRegression(0.1, prior=p)
    x = llr.train(SVAL1_reshape, score_labels)
    alpha, beta = x[0:-1], x[-1]
    k_scores_cal = (alpha * SVAL1_reshape + beta - np.log(p / (1 - p))).reshape(Nsamples, )
    minDCF = compute_minDCF_binary(k_scores_cal, score_labels, 0.2, 1, 1)
    print("minDCF:", minDCF)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(k_scores_cal > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, score_labels)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)
    print("actDCF1:", actDCF1)

    print("SCAL, calibration system2")
    # raw scores SCAL system1
    print("Raw scores")
    minDCF = compute_minDCF_binary(SCAL2, labels_cal, 0.2, 1, 1)
    print("minDCF:", minDCF)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(SCAL2 > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, labels_cal)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)
    print("actDCF1:", actDCF1)
    # calibration result SCAL system1
    print("Calibration scores")
    score_labels = np.hstack(labels_cal)
    Nsamples = SCAL2.shape[0]
    p = 0.2
    llr = LLR.LinearLogisticRegression(0.1, prior=p)
    x = llr.train(SCAL2_reshape, score_labels)
    alpha, beta = x[0:-1], x[-1]
    k_scores_cal = (alpha * SCAL2_reshape + beta - np.log(p / (1 - p))).reshape(Nsamples, )
    minDCF = compute_minDCF_binary(k_scores_cal, score_labels, 0.2, 1, 1)
    print("minDCF:", minDCF)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(k_scores_cal > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, score_labels)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)
    print("actDCF1:", actDCF1)

    print("SVAL, validation system1")
    # raw scores SCAL system1
    print("Raw scores")
    minDCF = compute_minDCF_binary(SVAL2, labels_val, 0.2, 1, 1)
    print("minDCF:", minDCF)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(SVAL2 > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, labels_val)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)
    print("actDCF1:", actDCF1)
    # calibration result SCAL system1
    print("Calibration scores")
    score_labels = np.hstack(labels_val)
    Nsamples = SVAL2.shape[0]
    p = 0.2
    llr = LLR.LinearLogisticRegression(0.1, prior=p)
    x = llr.train(SVAL2_reshape, score_labels)
    alpha, beta = x[0:-1], x[-1]
    k_scores_cal = (alpha * SVAL2_reshape + beta - np.log(p / (1 - p))).reshape(Nsamples, )
    minDCF = compute_minDCF_binary(k_scores_cal, score_labels, 0.2, 1, 1)
    print("minDCF:", minDCF)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(k_scores_cal > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, score_labels)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)
    print("actDCF1:", actDCF1)

    # Ridimensiona i punteggi di valutazione per l'uso con il modello di calibrazione
    eval_score1_reshape = eval_score1.reshape(1, -1)
    eval_score2_reshape = eval_score2.reshape(1, -1)

    # Calcola i punteggi calibrati per i punteggi di valutazione
    k_scores_cal_eval1 = (alpha * eval_score1_reshape + beta - np.log(p / (1 - p))).reshape(eval_score1.shape[0], )
    k_scores_cal_eval2 = (alpha * eval_score2_reshape + beta - np.log(p / (1 - p))).reshape(eval_score2.shape[0], )

    # Calcola le metriche sui punteggi di valutazione calibrati
    minDCF_eval1 = compute_minDCF_binary(k_scores_cal_eval1, eval_labels, 0.2, 1, 1)
    minDCF_eval2 = compute_minDCF_binary(k_scores_cal_eval2, eval_labels, 0.2, 1, 1)

    print("System 1 - Evaluation set: minDCF calibrated:", minDCF_eval1)
    print("System 2 - Evaluation set: minDCF calibrated:", minDCF_eval2)

    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(k_scores_cal_eval1 > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, eval_labels)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)

    Slval2 = np.int32(k_scores_cal_eval2 > th)
    confusionMatrix2 = compute_confusion_matrix(Slval2, eval_labels)
    actDCF2 = computeDCF_Binary(confusionMatrix2, 0.2, 1, 1, normalize=True)

    print("System 1 - Evaluation set: actDCF calibrated:", actDCF1)
    print("System 2 - Evaluation set: actDCF calibrated:", actDCF2)

    # Calcola le metriche sui punteggi di valutazione non calibrati
    minDCF_eval1 = compute_minDCF_binary(eval_score1, eval_labels, 0.2, 1, 1)
    minDCF_eval2 = compute_minDCF_binary(eval_score2, eval_labels, 0.2, 1, 1)

    print("System 1 - Evaluation set: minDCF not calibrated:", minDCF_eval1)
    print("System 2 - Evaluation set: minDCF not calibrated:", minDCF_eval2)

    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    Slval1 = np.int32(eval_score1 > th)
    confusionMatrix1 = compute_confusion_matrix(Slval1, eval_labels)
    actDCF1 = computeDCF_Binary(confusionMatrix1, 0.2, 1, 1, normalize=True)

    Slval2 = np.int32(eval_score2 > th)
    confusionMatrix2 = compute_confusion_matrix(Slval2, eval_labels)
    actDCF2 = computeDCF_Binary(confusionMatrix2, 0.2, 1, 1, normalize=True)

    print("System 1 - Evaluation set: actDCF not calibrated:", actDCF1)
    print("System 2 - Evaluation set: actDCF not calibrated:", actDCF2)

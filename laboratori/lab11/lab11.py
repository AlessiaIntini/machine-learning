import numpy as np
import scipy.optimize
import LLR_Clf as LLR
import matplotlib.pyplot as plt


def calculate_error_rate(predictions, targets):
    # return np.mean(predictions != targets)
    return ((predictions != targets).sum() / float(targets.size) * 100)


def vrow(x):
    return x.reshape((1, x.size))


def computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=False):
    Pfn = confusionMatrix[0, 1] / (confusionMatrix[0, 1] + confusionMatrix[1, 1])
    Pfp = confusionMatrix[1, 0] / (confusionMatrix[1, 0] + confusionMatrix[0, 0])
    bayesError = prior * Pfn * Cfn + (1 - prior) * Pfp * Cfp
    if normalize:
        return bayesError / np.minimum(prior * Cfn, (1 - prior) * Cfp)
    return bayesError


def bayesPlot(S, L, left=-3, right=3, npts=21):
    effPriorLogOdds = np.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        th = -np.log((effPrior * 1) / ((1 - effPrior) * 1))
        predictedLabels = np.int32(S > th)
        confusionMatrix = compute_confusion_matrix(predictedLabels, L)
        actDCF_current = computeDCF_Binary(confusionMatrix, effPrior, 1, 1, normalize=True)
        actDCF.append(actDCF_current)
        minDCF.append(compute_minDCF_binary(S, L, effPrior, 1.0, 1.0))
    return effPriorLogOdds, actDCF, minDCF


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


def extract_train_val_folds_from_ary(X, idx):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]


if __name__ == '__main__':
    score1 = np.load('data/scores_1.npy')
    score2 = np.load('data/scores_2.npy')
    labels = np.load('data/labels.npy')
    print(score1)

    eval_score1 = np.load('data/eval_scores_1.npy')
    eval_score2 = np.load('data/eval_scores_2.npy')
    eval_labels = np.load('data/eval_labels.npy')
    SAMEFIGPLOTS = True

    if SAMEFIGPLOTS:
        fig = plt.figure(figsize=(16, 9))
        axes = fig.subplots(3, 3, sharex='all')
        axes[2, 0].axis('off')
        fig.suptitle('Single fold')
    else:
        axes = np.array([[plt.figure().gca(), plt.figure().gca(), plt.figure().gca()],
                         [plt.figure().gca(), plt.figure().gca(), plt.figure().gca()],
                         [None, plt.figure().gca(), plt.figure().gca()]])

    # initial analysis
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

    print()
    print('*** Single-fold approach ***')
    print()

    SCAL1, SVAL1 = score1[::3], np.hstack([score1[1::3], score1[2::3]])
    SCAL2, SVAL2 = score2[::3], np.hstack([score2[1::3], score2[2::3]])
    labels_cal, labels_val = labels[::3], np.hstack([labels[1::3], labels[2::3]])

    # We recompute the system performance on the calibration validation sets (SVAL1 and SVAL2)
    # System 1
    minDCF1 = compute_minDCF_binary(SVAL1, labels_val, 0.2, 1, 1)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    predictedLabels = np.int32(SVAL1 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, labels_val)
    actDCF1 = computeDCF_Binary(confusionMatrix, 0.2, 1, 1, normalize=True)
    print("System 1 - Validation set: minDCF:", minDCF1)
    print("System 1 - Validation set: actDCF:", actDCF1)

    # System 2
    minDCF2 = compute_minDCF_binary(SVAL2, labels_val, 0.2, 1, 1)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    predictedLabels = np.int32(SVAL2 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, labels_val)
    actDCF2 = computeDCF_Binary(confusionMatrix, 0.2, 1, 1, normalize=True)
    print("System 1 - Validation set: minDCF:", minDCF2)
    print("System 1 - Validation set: actDCF:", actDCF2)
    logOdds, actDCF, minDCF = bayesPlot(SVAL1, labels_val)
    axes[0, 0].plot(logOdds, minDCF, color='C0', linestyle='--', label='minDCF')
    axes[0, 0].plot(logOdds, actDCF, color='C0', linestyle='-', label='actDCF')

    logOdds, actDCF, minDCF = bayesPlot(SVAL2, labels_val)
    axes[1, 0].plot(logOdds, minDCF, color='C1', linestyle='--', label='minDCF')
    axes[1, 0].plot(logOdds, actDCF, color='C1', linestyle='-', label='actDCF')

    axes[0, 0].set_ylim(0, 0.8)
    axes[0, 0].legend()

    axes[1, 0].set_ylim(0, 0.8)
    axes[1, 0].legend()

    axes[0, 0].set_title('System 1 - calibration validation - non-calibrated scores')
    axes[1, 0].set_title('System 2 - calibration validation - non-calibrated scores')

    # System 1
    # calibrate
    logOdds, actDCF, minDCF = bayesPlot(SVAL1, labels_val)
    axes[0, 1].plot(logOdds, minDCF, color='C0', linestyle='--', label='minDCF')
    axes[0, 1].plot(logOdds, actDCF, color='C0', linestyle=':', label='actDCF (pre-cal.)')
    print('System 1')
    print('\tValidation set')
    print('\t\tminDCF(p=0.2)         : %.3f' % minDCF1)
    print('\t\tactDCF(p=0.2), no cal.: %.3f' % actDCF1)

    pT = 0.2
    llr = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=pT)
    xf = llr.train(vrow(SCAL1), labels_cal)
    w1, b1 = xf[:-1], xf[-1]

    calibrated_SVAL1 = (w1.T @ vrow(SVAL1) + b1 - np.log(pT / (1 - pT))).ravel()
    th = -np.log((pT * 1) / ((1 - pT) * 1))
    predictedLabels = np.int32(calibrated_SVAL1 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, labels_val)
    actDCF1 = computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    logOdds, actDCF, _ = bayesPlot(calibrated_SVAL1, labels_val)
    print("System 1")
    print("actDCF(p=0.2), cal:", actDCF1)

    axes[0, 1].plot(logOdds, actDCF, label='actDCF (cal.)')
    axes[0, 1].set_ylim(0.0, 0.8)
    axes[0, 1].set_title('System 1 - calibration validation')
    axes[0, 1].legend()

    # System 2
    # calibrate
    logOdds, actDCF, minDCF = bayesPlot(SVAL2, labels_val)
    axes[1, 1].plot(logOdds, minDCF, color='C1', linestyle='--', label='minDCF')
    axes[1, 1].plot(logOdds, actDCF, color='C1', linestyle=':', label='actDCF (pre-cal).')
    print('System 2')
    print('\tValidation set')
    print('\t\tminDCF(p=0.2)         : %.3f' % minDCF2)
    print('\t\tactDCF(p=0.2), no cal.: %.3f' % actDCF2)

    pT = 0.2
    llr = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=pT)
    xf = llr.train(vrow(SCAL2), labels_cal)
    w2, b2 = xf[:-1], xf[-1]

    calibrated_SVAL2 = (w2.T @ vrow(SVAL2) + b2 - np.log(pT / (1 - pT))).ravel()
    th = -np.log((pT * 1) / ((1 - pT) * 1))
    predictedLabels = np.int32(calibrated_SVAL2 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, labels_val)
    actDCF1 = computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    logOdds, actDCF, _ = bayesPlot(calibrated_SVAL2, labels_val)
    print("System 2")
    print("actDCF(p=0.2), cal:", actDCF1)

    logOdds, actDCF, _ = bayesPlot(calibrated_SVAL2, labels_val)
    axes[1, 1].plot(logOdds, actDCF, color='C1', label='actDCF (cal.)')
    axes[1, 1].set_ylim(0.0, 0.8)
    axes[1, 1].set_title('System 2 - calibration validation')
    axes[1, 1].legend()

    # Evaluation set
    calibrated_eval_scores_system1 = (w1.T @ vrow(eval_score1) + b1 - np.log(pT / (1 - pT))).ravel()
    print("evaluation set")
    minDCF1 = compute_minDCF_binary(eval_score1, eval_labels, pT, 1, 1)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    predictedLabels_cal = np.int32(calibrated_eval_scores_system1 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels_cal, eval_labels)
    actDCF1_cal = computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    predictedLabels = np.int32(eval_score1 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, eval_labels)
    actDCF1 = computeDCF_Binary(confusionMatrix, 0.2, 1, 1, normalize=True)
    print("System 1 - Evaluation set: minDCF:", minDCF1)
    print("System 1 - Evaluation set: actDCF, no cal:", actDCF1)
    print("System 1 - Evaluation set: actDCF, cal:", actDCF1_cal)

    logOdds, actDCF_precal, minDCF = bayesPlot(eval_score1, eval_labels)
    logOdds, actDCF_cal, _ = bayesPlot(calibrated_eval_scores_system1, eval_labels)  # minDCF is the same

    axes[0, 2].plot(logOdds, minDCF, color='C0', linestyle='--', label='minDCF')
    axes[0, 2].plot(logOdds, actDCF_precal, color='C0', linestyle=':', label='actDCF (pre-cal.)')
    axes[0, 2].plot(logOdds, actDCF_cal, color='C0', linestyle='-', label='actDCF (cal.)')

    axes[0, 2].set_ylim(0.0, 0.8)
    axes[0, 2].set_title('System 1 - evaluation')
    axes[0, 2].legend()

    # System 2
    calibrated_eval_scores_system2 = (w2.T @ vrow(eval_score2) + b2 - np.log(pT / (1 - pT))).ravel()
    print("evaluation set")
    minDCF2 = compute_minDCF_binary(eval_score2, eval_labels, pT, 1, 1)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    predictedLabels_cal = np.int32(calibrated_eval_scores_system2 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels_cal, eval_labels)
    actDCF2_cal = computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    predictedLabels = np.int32(eval_score2 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, eval_labels)
    actDCF2 = computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    print("System 1 - Evaluation set: minDCF:", minDCF2)
    print("System 1 - Evaluation set: actDCF, no cal:", actDCF2)
    print("System 1 - Evaluation set: actDCF, cal:", actDCF2_cal)

    logOdds, actDCF_precal, minDCF = bayesPlot(eval_score2, eval_labels)
    logOdds, actDCF_cal, _ = bayesPlot(calibrated_eval_scores_system2, eval_labels)  # minDCF is the same

    axes[1, 2].plot(logOdds, minDCF, color='C1', linestyle='--', label='minDCF')
    axes[1, 2].plot(logOdds, actDCF_precal, color='C1', linestyle=':', label='actDCF (pre-cal.)')
    axes[1, 2].plot(logOdds, actDCF_cal, color='C1', linestyle='-', label='actDCF (cal.)')

    axes[1, 2].set_ylim(0.0, 0.8)
    axes[1, 2].set_title('System 2 - evaluation')
    axes[1, 2].legend()
    plt.show()

    if SAMEFIGPLOTS:
        fig = plt.figure(figsize=(16, 9))
        axes = fig.subplots(3, 3, sharex='all')
        fig.suptitle('K-fold')
    else:
        axes = np.array([[plt.figure().gca(), plt.figure().gca(), plt.figure().gca()],
                         [plt.figure().gca(), plt.figure().gca(), plt.figure().gca()],
                         [None, plt.figure().gca(), plt.figure().gca()]])

    # K-fold analysis
    KFOLD = 5
    # system 1
    noCalibrated_score_sys1 = []
    noCalibrated_label_sys1 = []
    calibrated_score_sys1 = []
    label_sys1 = []
    logOdds, actDCF, minDCF = bayesPlot(score1, labels)
    axes[0, 0].plot(logOdds, minDCF, color='C0', linestyle='--', label='minDCF (pre-cal.)')
    axes[0, 0].plot(logOdds, actDCF, color='C0', linestyle=':', label='actDCF (pre-cal.)')

    # System 1
    for foldIdx in range(KFOLD):
        SCAL, SVAL = extract_train_val_folds_from_ary(score1, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        noCalibrated_score_sys1.append(SVAL)
        noCalibrated_label_sys1.append(LVAL)

        xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=0.2).train(vrow(SCAL), LCAL)
        w, b = xf[:-1], xf[-1]

        calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(0.2 / (1 - 0.2))).ravel()
        calibrated_score_sys1.append(calibrated_SVAL)
        # devi tener traccia anche delle etichette
        label_sys1.append(LVAL)

    calibrated_score_sys1 = np.hstack(calibrated_score_sys1)
    label_sys1 = np.hstack(label_sys1)
    noCalibrated_score_sys1 = np.hstack(noCalibrated_score_sys1)
    noCalibrated_label_sys1 = np.hstack(noCalibrated_label_sys1)

    minDCF1 = compute_minDCF_binary(noCalibrated_score_sys1, noCalibrated_label_sys1, 0.2, 1, 1)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    predictedLabels = np.int32(noCalibrated_score_sys1 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, noCalibrated_label_sys1)
    actDCF1 = computeDCF_Binary(confusionMatrix, 0.2, 1, 1, normalize=True)
    print("System 1 - Evaluation set k-fold=5 : minDCF no cal :", minDCF1)
    print("System 1 - Evaluation set k-fold=5 : actDCF no cal :", actDCF1)

    minDCF1 = compute_minDCF_binary(calibrated_score_sys1, label_sys1, pT, 1, 1)
    th = -np.log((pT * 1) / ((1 - pT) * 1))
    predictedLabels = np.int32(calibrated_score_sys1 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, label_sys1)
    actDCF1 = computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    print("System 1 - Evaluation set k-fold=5 : minDCF cal :", minDCF1)
    print("System 1 - Evaluation set k-fold=5 : actDCF cal :", actDCF1)

    logOdds, actDCF, _ = bayesPlot(calibrated_score_sys1, label_sys1)
    axes[0, 0].plot(logOdds, actDCF, color='C0', linestyle='-',
                    label='actDCF (cal.)')  # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    axes[0, 0].legend()

    axes[0, 0].set_title('System 1 - validation')
    axes[0, 0].set_ylim(0, 0.8)
    # System 2
    noCalibrated_score_sys2 = []
    noCalibrated_label_sys2 = []
    calibrated_score_sys2 = []
    label_sys2 = []
    logOdds, actDCF, minDCF = bayesPlot(score2, labels)
    axes[1, 0].plot(logOdds, minDCF, color='C1', linestyle='--', label='minDCF (pre-cal.)')
    axes[1, 0].plot(logOdds, actDCF, color='C1', linestyle=':', label='actDCF (pre-cal.)')

    for foldIdx in range(KFOLD):
        SCAL, SVAL = extract_train_val_folds_from_ary(score2, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        noCalibrated_score_sys2.append(SVAL)
        noCalibrated_label_sys2.append(LVAL)

        xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=0.2).train(vrow(SCAL), LCAL)
        w, b = xf[:-1], xf[-1]

        calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(0.2 / (1 - 0.2))).ravel()
        calibrated_score_sys2.append(calibrated_SVAL)
        # devi tener traccia anche delle etichette
        label_sys2.append(LVAL)

    calibrated_score_sys2 = np.hstack(calibrated_score_sys2)
    label_sys2 = np.hstack(label_sys2)
    noCalibrated_score_sys2 = np.hstack(noCalibrated_score_sys2)
    noCalibrated_label_sys2 = np.hstack(noCalibrated_label_sys2)

    minDCF2 = compute_minDCF_binary(noCalibrated_score_sys2, noCalibrated_label_sys2, 0.2, 1, 1)
    th = -np.log((0.2 * 1) / ((1 - 0.2) * 1))
    predictedLabels = np.int32(noCalibrated_score_sys2 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, noCalibrated_label_sys2)
    actDCF2 = computeDCF_Binary(confusionMatrix, 0.2, 1, 1, normalize=True)
    print("System 2 - Evaluation set k-fold=5 : minDCF no cal :", minDCF2)
    print("System 2 - Evaluation set k-fold=5 : actDCF no cal :", actDCF2)

    minDCF2 = compute_minDCF_binary(calibrated_score_sys2, label_sys2, pT, 1, 1)
    th = -np.log((pT * 1) / ((1 - pT) * 1))
    predictedLabels = np.int32(calibrated_score_sys2 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, label_sys2)
    actDCF1 = computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    print("System 2 - Evaluation set k-fold=5 : minDCF cal :", minDCF2)
    print("System 2 - Evaluation set k-fold=5 : actDCF cal :", actDCF2)

    logOdds, actDCF, _ = bayesPlot(calibrated_score_sys2, label_sys2)
    axes[1, 0].plot(logOdds, actDCF, color='C1', linestyle='-',
                    label='actDCF (cal.)')  # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    axes[1, 0].legend()

    axes[1, 0].set_title('System 2 - validation')
    axes[1, 0].set_ylim(0, 0.8)

    # For K-fold the final model is a new model re-trained over the whole set, using the optimal hyperparameters we selected during the k-fold procedure (in this case we have no hyperparameter, so we simply train a new model on the whole dataset)
    xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=0.2).train(vrow(eval_score1), eval_labels)
    w, b = xf[:-1], xf[-1]
    calibrated_eval_score_sys1 = (w.T @ vrow(eval_score1) + b - np.log(0.2 / (1 - 0.2))).ravel()
    minDCF1 = compute_minDCF_binary(calibrated_eval_score_sys1, eval_labels, pT, 1, 1)
    th = -np.log((pT * 1) / ((1 - pT) * 1))
    predictedLabels = np.int32(calibrated_eval_score_sys1 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, eval_labels)
    actDCF1 = computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    print("System 1 - Evaluation set k-fold=5 : minDCF cal :", minDCF1)
    print("System 1 - Evaluation set k-fold=5 : actDCF cal :", actDCF1)
    predictedLabels = np.int32(eval_score1 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, eval_labels)
    actDCF1 = computeDCF_Binary(confusionMatrix, 0.2, 1, 1, normalize=True)
    print("System 1 - Evaluation set k-fold=5 : actDCF no cal :", actDCF1)

    logOdds, actDCF_precal, minDCF = bayesPlot(eval_score1, eval_labels)
    logOdds, actDCF, _ = bayesPlot(calibrated_eval_score_sys1, eval_labels)
    axes[0, 1].plot(logOdds, actDCF, color='C0', linestyle='--',
                    label='actDCF (cal.)')  # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    axes[0, 1].plot(logOdds, actDCF_precal, color='C0', linestyle=':',
                    label='actDCF (pre cal.)')
    axes[0, 1].plot(logOdds, minDCF, color='C0', linestyle='-',
                    label='minDCF ')
    axes[0, 1].legend()
    axes[0, 1].set_title('System 1 - evaluation')
    axes[0, 1].set_ylim(0, 0.8)

    # system 2
    xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=0.2).train(vrow(eval_score2), eval_labels)
    w, b = xf[:-1], xf[-1]
    calibrated_eval_score_sys2 = (w.T @ vrow(eval_score2) + b - np.log(0.2 / (1 - 0.2))).ravel()
    minDCF2 = compute_minDCF_binary(calibrated_eval_score_sys2, eval_labels, pT, 1, 1)
    th = -np.log((pT * 1) / ((1 - pT) * 1))
    predictedLabels = np.int32(calibrated_eval_score_sys2 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, eval_labels)
    actDCF2 = computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    print("System 2- Evaluation set k-fold=5 : minDCF cal :", minDCF2)
    print("System 2 - Evaluation set k-fold=5 : actDCF cal :", actDCF2)
    predictedLabels = np.int32(eval_score2 > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, eval_labels)
    actDCF2 = computeDCF_Binary(confusionMatrix, 0.2, 1, 1, normalize=True)
    print("System 2- Evaluation set k-fold=5 : actDCF no cal :", actDCF2)

    logOdds, actDCF_precal, minDCF = bayesPlot(eval_score2, eval_labels)
    logOdds, actDCF, _ = bayesPlot(calibrated_eval_score_sys2, eval_labels)
    axes[1, 1].plot(logOdds, minDCF, color='C1', linestyle='--',
                    label='minDCF')
    axes[1, 1].plot(logOdds, actDCF_precal, color='C1', linestyle=':',
                    label='actDCF (pre cal.)')
    axes[1, 1].plot(logOdds, actDCF, color='C1', linestyle='-',
                    label='actDCF (cal.)')  # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    axes[1, 1].legend()
    axes[1, 1].set_title('System 2 - evaluation')
    axes[1, 1].set_ylim(0, 0.8)

    # fusion

    fusedScore = []
    fusedLabels = []
    pT = 0.2

    for foldIdx in range(KFOLD):
        SCAL1, SVAL1 = extract_train_val_folds_from_ary(score1, foldIdx)
        SCAL2, SVAL2 = extract_train_val_folds_from_ary(score2, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)

        SCAL = np.vstack([SCAL1, SCAL2])
        xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=0.2).train(SCAL, LCAL)
        w, b = xf[:-1], xf[-1]
        SVAL = np.vstack([SVAL1, SVAL2])
        calibrated_SVAL = (w.T @ SVAL + b - np.log(0.2 / (1 - 0.2))).ravel()
        fusedScore.append(calibrated_SVAL)
        fusedLabels.append(LVAL)

    fusedScore = np.hstack(fusedScore)
    fusedLabels = np.hstack(fusedLabels)
    minDCF = compute_minDCF_binary(fusedScore, fusedLabels, pT, 1, 1)
    th = -np.log((pT * 1) / ((1 - pT) * 1))
    predictedLabels = np.int32(fusedScore > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, fusedLabels)
    actDCF = computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    print("Fusion - Evaluation set k-fold=5 : minDCF cal :", minDCF)
    print("Fusion - Evaluation set k-fold=5 : actDCF cal :", actDCF)

    logOdds, actDCF, minDCF = bayesPlot(calibrated_score_sys1, label_sys1)
    axes[2, 1].set_title('Fusion - validation')
    axes[2, 1].plot(logOdds, minDCF, color='C0', linestyle='--', label='S1 - minDCF')
    axes[2, 1].plot(logOdds, actDCF, color='C0', linestyle='-', label='S1 - actDCF')
    logOdds, actDCF, minDCF = bayesPlot(calibrated_score_sys2, label_sys2)
    axes[2, 1].plot(logOdds, minDCF, color='C1', linestyle='--', label='S2 - minDCF')
    axes[2, 1].plot(logOdds, actDCF, color='C1', linestyle='-', label='S2 - actDCF')

    logOdds, actDCF, minDCF = bayesPlot(fusedScore, fusedLabels)
    axes[2, 1].plot(logOdds, minDCF, color='C2', linestyle='--', label='S1 + S2 - KFold - minDCF(0.2)')
    axes[2, 1].plot(logOdds, actDCF, color='C2', linestyle='-', label='S1 + S2 - KFold - actDCF(0.2)')
    axes[2, 1].set_ylim(0.0, 0.8)
    axes[2, 1].legend()

    SMatrix = np.vstack([score1, score2])
    xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=0.2).train(SMatrix, labels)
    w, b = xf[:-1], xf[-1]

    SMatrixEval = np.vstack([eval_score1, eval_score2])
    fused_eval_scores = (w.T @ SMatrixEval + b - np.log(0.2 / (1 - 0.2))).ravel()

    minDCF = compute_minDCF_binary(fused_eval_scores, eval_labels, pT, 1, 1)
    th = -np.log((pT * 1) / ((1 - pT) * 1))
    predictedLabels = np.int32(fused_eval_scores > th)
    confusionMatrix = compute_confusion_matrix(predictedLabels, eval_labels)
    actDCF = computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    print("Fusion - Evaluation set k-fold=5 : minDCF cal :", minDCF)
    print("Fusion - Evaluation set k-fold=5 : actDCF cal :", actDCF)

    # calibrated evaluation
    logOdds, actDCF, minDCF = bayesPlot(calibrated_eval_score_sys1, eval_labels)
    axes[2, 2].plot(logOdds, minDCF, color='C0', linestyle='--', label='S1 - minDCF')
    axes[2, 2].plot(logOdds, actDCF, color='C0', linestyle='-', label='S1 - actDCF')
    logOdds, actDCF, minDCF = bayesPlot(calibrated_eval_score_sys2, eval_labels)
    axes[2, 2].plot(logOdds, minDCF, color='C1', linestyle='--', label='S2 - minDCF')
    axes[2, 2].plot(logOdds, actDCF, color='C1', linestyle='-', label='S2 - actDCF')

    logOdds, actDCF, minDCF = bayesPlot(fused_eval_scores, eval_labels)  # minDCF is the same
    axes[2, 2].plot(logOdds, minDCF, color='C2', linestyle='--', label='S1 + S2 - minDCF')
    axes[2, 2].plot(logOdds, actDCF, color='C2', linestyle='-', label='S1 + S2 - actDCF')
    axes[2, 2].set_ylim(0.0, 0.8)
    axes[2, 2].set_title('Fusion - evaluation')
    axes[2, 2].legend()

    plt.show()

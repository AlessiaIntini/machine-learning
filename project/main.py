import matplotlib
import numpy as np
from sklearn.preprocessing import StandardScaler

import BayesDecisionModel as bdm
import Evaluation as e
import GaussianModel as gm
import LLR
import PCA
import ReadData as rd
from LLR import LinearLogisticRegression

if __name__ == '__main__':
    D, L = rd.load('trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = rd.split_db_2to1(D, L)
    # ########################
    # ## FEATURES ANALYSIS ###
    # ########################
    # fa.plot_features(D, L)

    # ########################
    # ## DIM REDUCTION #######
    # ########################
    # dr.Dim_red(D, L)

    # Classification

    # dr.classification(DTR, LTR, DVAL, LVAL)

    # ########################
    # ## Gausian Density ####
    # ########################
    # gd.apply_ML(D, L)

    # ########################
    # ## Gausian Models ####
    # ########################
    # features 1,2,3,4,5,6
    print("Features 1,2,3,4,5,6")
    DTR_mvg = gm.calculate_MVG(DTR, LTR, DVAL, LVAL)
    DTR_naive = gm.calculate_Naive(DTR, LTR, DVAL, LVAL)
    DTR_Tied = gm.calculate_Tied(DTR, LTR, DVAL, LVAL)

    # Correlation
    gm.correlation(DTR, LTR)

    # feature 1,2,3,4
    print("Features 1,2,3,4")
    gm.calculate_MVG(DTR[:4], LTR, DVAL[:4], LVAL)
    gm.calculate_Naive(DTR[:4], LTR, DVAL[:4], LVAL)
    gm.calculate_Tied(DTR[:4], LTR, DVAL[:4], LVAL)

    # feature 1,2
    print("Features 1,2")
    gm.calculate_MVG(DTR[:2], LTR, DVAL[:2], LVAL)
    gm.calculate_Naive(DTR[:2], LTR, DVAL[:2], LVAL)
    gm.calculate_Tied(DTR[:2], LTR, DVAL[:2], LVAL)

    # feature 3,4
    print("Features 3,4")
    gm.calculate_MVG(DTR[2:4], LTR, DVAL[2:4], LVAL)
    gm.calculate_Naive(DTR[2:4], LTR, DVAL[2:4], LVAL)
    gm.calculate_Tied(DTR[2:4], LTR, DVAL[2:4], LVAL)

    # apply PCA+MVG

    for i in range(1, 7):
        s, P = PCA.PCA_function(DTR, i)
        print("PCA+MVG, m=", i)
        DTR_PCA = np.dot(P.T, DTR)
        DVAL_PCA = np.dot(P.T, DVAL)
        DTR_mvgPCA = gm.calculate_MVG(DTR_PCA, LTR, DVAL_PCA, LVAL)
        DTR_naivePCA = gm.calculate_Naive(DTR_PCA, LTR, DVAL_PCA, LVAL)
        DTR_tiedPCA = gm.calculate_Tied(DTR_PCA, LTR, DVAL_PCA, LVAL)

    # #################################### #
    # ## Bayes Decision Model Evaluation  ##
    # #################################### #

    bdm.ApplyDecisionModel([(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)],
                           DTR_mvg, DTR_Tied, DTR_naive, LVAL)

    # Apply PCA
    best_DCF = {'MVG': float('inf'), 'naive': float('inf'), 'tied': float('inf')}
    best_m = {'MVG': None, 'naive': None, 'tied': None}
    best_DCF_norm = {'MVG': float('inf'), 'naive': float('inf'), 'tied': float('inf')}
    best_DCF_min = {'MVG': float('inf'), 'naive': float('inf'), 'tied': float('inf')}
    print("Apply PCA")
    DTR_mvgPCA = {}
    DTR_naivePCA = {}
    DTR_tiedPCA = {}

    for i in range(1, 7):
        s, P = PCA.PCA_function(DTR, i)
        print("PCA+MVG, m=", i)
        DTR_PCA = np.dot(P.T, DTR)
        DVAL_PCA = np.dot(P.T, DVAL)
        DTR_mvgPCA[i] = gm.calculate_MVG(DTR_PCA, LTR, DVAL_PCA, LVAL)
        DTR_naivePCA[i] = gm.calculate_Naive(DTR_PCA, LTR, DVAL_PCA, LVAL)
        DTR_tiedPCA[i] = gm.calculate_Tied(DTR_PCA, LTR, DVAL_PCA, LVAL)
        best_DCF_norm, best_m, best_DCF, best_DCF_min = bdm.ApplyDecisionModel(
            [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)],
            DTR_mvgPCA[i], DTR_tiedPCA[i], DTR_naivePCA[i], LVAL, i)

    print("Best DCF: ", best_DCF)
    print("Best DCF normalized: ", best_DCF_norm)
    print("Best m: ", best_m)
    print("Best min DCF: ", best_DCF_min)

    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPrior = 1 / (1 + np.exp(-effPriorLogOdds))
    DCF = []
    minDCF = []
    for prior in effPrior:
        data_prediction = bdm.compute_optimal_Bayes_binary_llr(DTR_mvgPCA[best_m['MVG']], prior, 1.0, 1.0)
        confusionMatrix = bdm.compute_confusion_matrix(data_prediction, LVAL)
        DCF.append(bdm.computeDCF_Binary(confusionMatrix, prior, 1.0, 1.0, normalize=True))
        minDCF.append(
            bdm.compute_minDCF_binary(DTR_mvgPCA[best_m['MVG']], LVAL, prior, 1.0, 1.0, returnThreshold=False))

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('MVG')
    matplotlib.pyplot.plot(effPriorLogOdds, DCF, label='DCF', color='r')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    matplotlib.pyplot.xlabel('log odds prior')
    matplotlib.pyplot.ylabel('DCF value')
    matplotlib.pyplot.xlim(-4, 4)
    matplotlib.pyplot.ylim(0.0, 0.7)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

    DCF = []
    minDCF = []
    for prior in effPrior:
        data_prediction = bdm.compute_optimal_Bayes_binary_llr(DTR_naivePCA[best_m['naive']], prior, 1.0, 1.0)
        confusionMatrix = bdm.compute_confusion_matrix(data_prediction, LVAL)
        DCF.append(bdm.computeDCF_Binary(confusionMatrix, prior, 1.0, 1.0, normalize=True))
        minDCF.append(
            bdm.compute_minDCF_binary(DTR_naivePCA[best_m['naive']], LVAL, prior, 1.0, 1.0, returnThreshold=False))

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('Naive')
    matplotlib.pyplot.plot(effPriorLogOdds, DCF, label='DCF', color='r')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    matplotlib.pyplot.xlabel('log odds prior')
    matplotlib.pyplot.ylabel('DCF value')
    matplotlib.pyplot.xlim(-4, 4)
    matplotlib.pyplot.ylim(0.0, 0.7)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

    DCF = []
    minDCF = []
    for prior in effPrior:
        data_prediction = bdm.compute_optimal_Bayes_binary_llr(DTR_tiedPCA[best_m['tied']], prior, 1.0, 1.0)
        confusionMatrix = bdm.compute_confusion_matrix(data_prediction, LVAL)
        DCF.append(bdm.computeDCF_Binary(confusionMatrix, prior, 1.0, 1.0, normalize=True))
        minDCF.append(
            bdm.compute_minDCF_binary(DTR_tiedPCA[best_m['tied']], LVAL, prior, 1.0, 1.0, returnThreshold=False))

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('Tied')
    matplotlib.pyplot.plot(effPriorLogOdds, DCF, label='DCF', color='r')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    matplotlib.pyplot.xlabel('log odds prior')
    matplotlib.pyplot.ylabel('DCF value')
    matplotlib.pyplot.xlim(-4, 4)
    matplotlib.pyplot.ylim(0.0, 0.7)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

    # #################################### #
    # ##    Linear Logistic regression    ##
    # #################################### #

    print("Linear Logistic regression")
    ldbArray = np.logspace(-4, 2, 13)
    actDCF = []
    minDCF = []
    for ldb in ldbArray:
        llr = LinearLogisticRegression(ldb, prior_weighted=False, prior=0.1)
        current_xf = llr.trainLogReg(DTR, LTR)
        current_minDCF, current_actDCF = LLR.compute_minDCF_actDCF(current_xf, LVAL, DVAL,
                                                                   pi_emp=sum(LTR == 1) / len(LTR),
                                                                   prior=0.1)
        actDCF.append(current_actDCF)
        minDCF.append(current_minDCF)

    LLR.plot_minDCF_actDCF(minDCF, actDCF, 'Linear Logistic Regression no prior_weighted', ldbArray)

    ldbArray = np.logspace(-4, 2, 13)
    actDCF = []
    minDCF = []
    for ldb in ldbArray:
        llr = LinearLogisticRegression(ldb, prior_weighted=False, prior=0.1)
        current_xf = llr.trainLogReg(DTR[:, ::50], LTR[::50])
        current_minDCF, current_actDCF = LLR.compute_minDCF_actDCF(current_xf, LVAL, DVAL,
                                                                   pi_emp=sum(LTR[::50] == 1) / len(LTR[::50]),
                                                                   prior=0.1)
        actDCF.append(current_actDCF)
        minDCF.append(current_minDCF)

    LLR.plot_minDCF_actDCF(minDCF, actDCF, 'Linear Logistic Regression no prior_weighted only 50 sample', ldbArray)

    print("Linear Logistic regression with prior_weighted=True")
    ldbArray = np.logspace(-4, 2, 13)
    actDCF = []
    minDCF = []
    for ldb in ldbArray:
        llr = LinearLogisticRegression(ldb, prior_weighted=True, prior=0.1)
        current_xf = llr.trainLogReg(DTR, LTR)
        current_minDCF, current_actDCF = LLR.compute_minDCF_actDCF(current_xf, LVAL, DVAL,
                                                                   pi_emp=sum(LTR == 1) / len(LTR),
                                                                   prior=0.1)
        actDCF.append(current_actDCF)
        minDCF.append(current_minDCF)

    LLR.plot_minDCF_actDCF(minDCF, actDCF, 'Linear Logistic Regression with prior_weighted=True', ldbArray)

    DTR_expanded = np.concatenate([DTR, np.square(DTR)], axis=0)
    DVAL_expanded = np.concatenate([DVAL, np.square(DVAL)], axis=0)
    actDCF = []
    minDCF = []
    for ldb in ldbArray:
        llr = LinearLogisticRegression(ldb, prior_weighted=False, prior=0.1)
        current_xf = llr.trainLogReg(DTR_expanded, LTR)
        current_minDCF, current_actDCF = LLR.compute_minDCF_actDCF(current_xf, LVAL, DVAL_expanded,
                                                                   pi_emp=sum(LTR == 1) / len(LTR),
                                                                   prior=0.1)
        actDCF.append(current_actDCF)
        minDCF.append(current_minDCF)

    LLR.plot_minDCF_actDCF(minDCF, actDCF, 'Linear Logistic Regression with expanded features', ldbArray)

    # apply PCA
    print("Linear Logistic regression with PCA")
    for m in range(4, 7):
        s, P = PCA.PCA_function(DTR, m)
        DTR_PCA = np.dot(P.T, DTR)
        DVAL_PCA = np.dot(P.T, DVAL)

        actDCF = []
        minDCF = []
        for ldb in ldbArray:
            llr = LinearLogisticRegression(ldb, prior_weighted=False, prior=0.1)
            current_xf = llr.trainLogReg(DTR_PCA, LTR)
            current_minDCF, current_actDCF = LLR.compute_minDCF_actDCF(current_xf, LVAL, DVAL_PCA,
                                                                       pi_emp=sum(LTR == 1) / len(LTR),
                                                                       prior=0.1)
            actDCF.append(current_actDCF)
            minDCF.append(current_minDCF)

        LLR.plot_minDCF_actDCF(minDCF, actDCF, 'Linear Logistic Regression with PCA with', ldbArray, m)

    # apply PCA and Z-normalization
    print("Linear Logistic regression with PCA and Z-normalization")
    scaler = StandardScaler().fit(DTR)
    DTR_normalized = e.znormalized_features_training(DTR)
    DVAL_normalized = e.znormalized_features_evaluation(DVAL, DTR)
    for m in range(4, 7):
        s, P = PCA.PCA_function(DTR_normalized, m)
        DTR_PCA = np.dot(P.T, DTR_normalized)
        DVAL_PCA = np.dot(P.T, DVAL_normalized)

        actDCF = []
        minDCF = []
        for ldb in ldbArray:
            llr = LinearLogisticRegression(ldb, prior_weighted=False, prior=0.1)
            current_xf = llr.trainLogReg(DTR_PCA, LTR)
            current_minDCF, current_actDCF = LLR.compute_minDCF_actDCF(current_xf, LVAL, DVAL_PCA,
                                                                       pi_emp=(LTR == 1).sum() / len(LTR),
                                                                       prior=0.1)
            actDCF.append(current_actDCF)
            minDCF.append(current_minDCF)

        LLR.plot_minDCF_actDCF(minDCF, actDCF, 'Linear Logistic Regression with Z-normalization and PCA with', ldbArray,
                               m)

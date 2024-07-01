import matplotlib
import numpy as np
from sklearn.preprocessing import StandardScaler

import BayesDecisionModel as bdm
import Evaluation as e
import GaussianModel as gm
import LLR_Clf
import PCA
import ReadData as rd
import GMM_Clf as GMM
import SVM_Clf as SVM
import plot as plt
import matplotlib.pyplot as pl
import LLR_Clf as LLR
import QLR_Clf as QLR
import Dim_reduction as dr
import FeaturesAnalysis as fa
import GaussianDensity as gd

if __name__ == '__main__':
    D, L = rd.load('Data/trainData.txt')
    evalData, evalLabels = rd.load("data/evalData.txt")
    rd.countLabel(L, 'Train Data')
    rd.countLabel(evalLabels, 'Eval Data')
    (DTR, LTR), (DVAL, LVAL) = rd.split_db_2to1(D, L)
    ########################
    ## FEATURES ANALYSIS ###
    ########################
    # fa.plot_features(D, L)
    # fa.plot_single_feature(D, L)
    # fa.plot_features_pairwise(D, L)

    ########################
    ## DIM REDUCTION #######
    ########################
    # dr.Dim_red(D, L)

    # Classification

    # dr.classification(DTR, LTR, DVAL, LVAL)

    ########################
    ## Gausian Density ####
    ########################
    # gd.apply_ML(D, L)

    ########################
    ## Gausian Models ####
    ########################
    # features 1,2,3,4,5,6
    # print("Features 1,2,3,4,5,6")
    # DTR_mvg = gm.calculate_MVG(DTR, LTR, DVAL, LVAL)
    # DTR_naive = gm.calculate_Naive(DTR, LTR, DVAL, LVAL)
    # DTR_Tied = gm.calculate_Tied(DTR, LTR, DVAL, LVAL)
    #
    # # Correlation
    # gm.correlation(DTR, LTR)
    #
    # # feature 1,2,3,4
    # print("Features 1,2,3,4")
    # gm.calculate_MVG(DTR[:4], LTR, DVAL[:4], LVAL)
    # gm.calculate_Naive(DTR[:4], LTR, DVAL[:4], LVAL)
    # gm.calculate_Tied(DTR[:4], LTR, DVAL[:4], LVAL)
    #
    # # feature 1,2
    # print("Features 1,2")
    # gm.calculate_MVG(DTR[:2], LTR, DVAL[:2], LVAL)
    # gm.calculate_Naive(DTR[:2], LTR, DVAL[:2], LVAL)
    # gm.calculate_Tied(DTR[:2], LTR, DVAL[:2], LVAL)
    #
    # # feature 3,4
    # print("Features 3,4")
    # gm.calculate_MVG(DTR[2:4], LTR, DVAL[2:4], LVAL)
    # gm.calculate_Naive(DTR[2:4], LTR, DVAL[2:4], LVAL)
    # gm.calculate_Tied(DTR[2:4], LTR, DVAL[2:4], LVAL)
    #
    # # apply PCA+MVG
    #
    # for i in range(1, 7):
    #     s, P = PCA.PCA_function(DTR, i)
    #     print("PCA+MVG, m=", i)
    #     DTR_PCA = np.dot(P.T, DTR)
    #     DVAL_PCA = np.dot(P.T, DVAL)
    #     DTR_mvgPCA = gm.calculate_MVG(DTR_PCA, LTR, DVAL_PCA, LVAL)
    #     DTR_naivePCA = gm.calculate_Naive(DTR_PCA, LTR, DVAL_PCA, LVAL)
    #     DTR_tiedPCA = gm.calculate_Tied(DTR_PCA, LTR, DVAL_PCA, LVAL)

    #################################### #
    ## Bayes Decision Model Evaluation  ##
    #################################### #

    # bdm.ApplyDecisionModel([(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)],
    #                        DTR_mvg, DTR_Tied, DTR_naive, LVAL)
    #
    # # Apply PCA
    # best_DCF = {'MVG': float('inf'), 'naive': float('inf'), 'tied': float('inf')}
    # best_m = {'MVG': None, 'naive': None, 'tied': None}
    # best_DCF_norm = {'MVG': float('inf'), 'naive': float('inf'), 'tied': float('inf')}
    # best_DCF_min = {'MVG': float('inf'), 'naive': float('inf'), 'tied': float('inf')}
    # print("Apply PCA")
    # DTR_mvgPCA = {}
    # DTR_naivePCA = {}
    # DTR_tiedPCA = {}
    #
    # for i in range(1, 7):
    #     s, P = PCA.PCA_function(DTR, i)
    #     print("PCA+MVG, m=", i)
    #     DTR_PCA = np.dot(P.T, DTR)
    #     DVAL_PCA = np.dot(P.T, DVAL)
    #     DTR_mvgPCA[i] = gm.calculate_MVG(DTR_PCA, LTR, DVAL_PCA, LVAL)
    #     DTR_naivePCA[i] = gm.calculate_Naive(DTR_PCA, LTR, DVAL_PCA, LVAL)
    #     DTR_tiedPCA[i] = gm.calculate_Tied(DTR_PCA, LTR, DVAL_PCA, LVAL)
    #     best_DCF_norm, best_m, best_DCF, best_DCF_min = bdm.ApplyDecisionModel(
    #         [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)],
    #         DTR_mvgPCA[i], DTR_tiedPCA[i], DTR_naivePCA[i], LVAL, i)
    #
    # print("Best DCF: ", best_DCF)
    # print("Best DCF normalized: ", best_DCF_norm)
    # print("Best m: ", best_m)
    # print("Best min DCF: ", best_DCF_min)
    #
    # effPriorLogOdds = np.linspace(-4, 4, 21)
    # effPrior = 1 / (1 + np.exp(-effPriorLogOdds))
    # DCF = []
    # minDCF = []
    # for prior in effPrior:
    #     data_prediction = bdm.compute_optimal_Bayes_binary_llr(DTR_mvgPCA[best_m['MVG']], prior, 1.0, 1.0)
    #     confusionMatrix = bdm.compute_confusion_matrix(data_prediction, LVAL)
    #     DCF.append(bdm.computeDCF_Binary(confusionMatrix, prior, 1.0, 1.0, normalize=True))
    #     minDCF.append(
    #         bdm.compute_minDCF_binary(DTR_mvgPCA[best_m['MVG']], LVAL, prior, 1.0, 1.0, returnThreshold=False))
    #
    # matplotlib.pyplot.figure()
    # matplotlib.pyplot.title('Full Covariance')
    # matplotlib.pyplot.plot(effPriorLogOdds, DCF, label='DCF', color='r')
    # matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    # matplotlib.pyplot.xlabel('log odds prior')
    # matplotlib.pyplot.ylabel('DCF value')
    # matplotlib.pyplot.xlim(-4, 4)
    # matplotlib.pyplot.ylim(0.0, 0.7)
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.show()
    #
    # DCF = []
    # minDCF = []
    # for prior in effPrior:
    #     data_prediction = bdm.compute_optimal_Bayes_binary_llr(DTR_naivePCA[best_m['naive']], prior, 1.0, 1.0)
    #     confusionMatrix = bdm.compute_confusion_matrix(data_prediction, LVAL)
    #     DCF.append(bdm.computeDCF_Binary(confusionMatrix, prior, 1.0, 1.0, normalize=True))
    #     minDCF.append(
    #         bdm.compute_minDCF_binary(DTR_naivePCA[best_m['naive']], LVAL, prior, 1.0, 1.0, returnThreshold=False))
    #
    # matplotlib.pyplot.figure()
    # matplotlib.pyplot.title('Naïve Bayes')
    # matplotlib.pyplot.plot(effPriorLogOdds, DCF, label='DCF', color='r')
    # matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    # matplotlib.pyplot.xlabel('log odds prior')
    # matplotlib.pyplot.ylabel('DCF value')
    # matplotlib.pyplot.xlim(-4, 4)
    # matplotlib.pyplot.ylim(0.0, 0.7)
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.show()
    #
    # DCF = []
    # minDCF = []
    # for prior in effPrior:
    #     data_prediction = bdm.compute_optimal_Bayes_binary_llr(DTR_tiedPCA[best_m['tied']], prior, 1.0, 1.0)
    #     confusionMatrix = bdm.compute_confusion_matrix(data_prediction, LVAL)
    #     DCF.append(bdm.computeDCF_Binary(confusionMatrix, prior, 1.0, 1.0, normalize=True))
    #     minDCF.append(
    #         bdm.compute_minDCF_binary(DTR_tiedPCA[best_m['tied']], LVAL, prior, 1.0, 1.0, returnThreshold=False))
    #
    # matplotlib.pyplot.figure()
    # matplotlib.pyplot.title('Tied Covariance')
    # matplotlib.pyplot.plot(effPriorLogOdds, DCF, label='DCF', color='r')
    # matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    # matplotlib.pyplot.xlabel('log odds prior')
    # matplotlib.pyplot.ylabel('DCF value')
    # matplotlib.pyplot.xlim(-4, 4)
    # matplotlib.pyplot.ylim(0.0, 0.7)
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.show()

    # #################################### #
    # ##    Linear Logistic regression    ##
    # #################################### #
    print("Binary Logistic regression##")
    print("Binary Logistic regression non prior-weighted")
    ldbArray = np.logspace(-4, 2, 13)
    actDCF = []
    minDCF = []
    minValue_minDCF = {'lambda': 0, 'weight': False, 'min': 10, 'expanded': False, 'PCA': -1, 'Z-norm': False,
                       "50-sample": False}
    minValue_actDCF = {'lambda': 0, 'weight': False, 'min': 10, 'expanded': False, 'PCA': -1, 'Z-norm': False,
                       "50-sample": False}
    for ldb in ldbArray:
        print("lambda: ", ldb)
        llr = LLR.LinearLogisticRegression(ldb, prior_weighted=False)
        current_xf = llr.trainLogReg(DTR, LTR)
        current_minDCF, current_actDCF = LLR.compute_minDCF_actDCF(current_xf, LVAL, DVAL,
                                                                   pi_emp=sum(LTR == 1) / len(LTR),
                                                                   prior=0.1)
        if current_minDCF < minValue_minDCF['min']:
            minValue_minDCF['min'] = current_minDCF
            minValue_minDCF['lambda'] = ldb
        if current_actDCF < minValue_actDCF['min']:
            minValue_actDCF['min'] = current_actDCF
            minValue_actDCF['lambda'] = ldb
        actDCF.append(current_actDCF)
        minDCF.append(current_minDCF)

    plt.plot_minDCF_actDCF(minDCF, actDCF, 'Binary Logistic regression non prior-weighted', ldbArray)
    plt.plot_minDCF_actDCF(minDCF, actDCF, 'Binary Logistic regression non prior-weighted', ldbArray, One=True)

    print("Binary Logistic regression non prior-weighted only 50 sample")
    ldbArray = np.logspace(-4, 2, 13)
    actDCF = []
    minDCF = []
    for ldb in ldbArray:
        print("lambda: ", ldb)
        llr = LLR.LinearLogisticRegression(ldb, prior_weighted=False)
        current_xf = llr.trainLogReg(DTR[:, ::50], LTR[::50])
        current_minDCF, current_actDCF = LLR.compute_minDCF_actDCF(current_xf, LVAL, DVAL,
                                                                   pi_emp=sum(LTR[::50] == 1) / len(LTR[::50]),
                                                                   prior=0.1)
        if current_minDCF < minValue_minDCF['min']:
            minValue_minDCF['min'] = current_minDCF
            minValue_minDCF['lambda'] = ldb
            minValue_minDCF['50-sample'] = True
        if current_actDCF < minValue_actDCF['min']:
            minValue_actDCF['min'] = current_actDCF
            minValue_actDCF['lambda'] = ldb
            minValue_actDCF['50-sample'] = True
        actDCF.append(current_actDCF)
        minDCF.append(current_minDCF)

    plt.plot_minDCF_actDCF(minDCF, actDCF, 'Binary Logistic regression non prior-weighted (50 samples)', ldbArray)
    plt.plot_minDCF_actDCF(minDCF, actDCF, 'Binary Logistic regression non prior-weighted (50 samples)', ldbArray,
                           One=True)
    print("Binary Logistic regression with prior-weighted pi_t=0.1")
    ldbArray = np.logspace(-4, 2, 13)
    actDCF = []
    minDCF = []
    for ldb in ldbArray:
        print("lambda: ", ldb)
        llr = LLR.LinearLogisticRegression(ldb, prior_weighted=True, prior=0.1)
        current_xf = llr.trainLogReg(DTR, LTR)
        current_minDCF, current_actDCF = LLR.compute_minDCF_actDCF(current_xf, LVAL, DVAL,
                                                                   pi_emp=sum(LTR == 1) / len(LTR),
                                                                   prior=0.1)

        if current_minDCF < minValue_minDCF['min']:
            minValue_minDCF['min'] = current_minDCF
            minValue_minDCF['lambda'] = ldb
            minValue_minDCF['weight'] = True
        if current_actDCF < minValue_actDCF['min']:
            minValue_actDCF['min'] = current_actDCF
            minValue_actDCF['lambda'] = ldb
            minValue_actDCF['weight'] = True
        actDCF.append(current_actDCF)
        minDCF.append(current_minDCF)

    plt.plot_minDCF_actDCF(minDCF, actDCF, 'Binary Logistic Regression with prior-weighted pi_t=0.1', ldbArray)
    plt.plot_minDCF_actDCF(minDCF, actDCF, 'Binary Logistic Regression with prior-weighted pi_t=0.1', ldbArray,
                           One=True)

    print("Quadratic Logistic regression non prior-weighted")
    DTR_expanded = np.concatenate([DTR, np.square(DTR)], axis=0)
    DVAL_expanded = np.concatenate([DVAL, np.square(DVAL)], axis=0)
    actDCF = []
    minDCF = []
    for ldb in ldbArray:
        print("lambda: ", ldb)
        qlr = QLR.QuadraticLogisticRegression(ldb, prior_weighted=False)
        current_xf = qlr.train(DTR, LTR)
        current_minDCF, current_actDCF = qlr.compute_minDCF_actDCF(LVAL, DVAL,
                                                                   pi_emp=sum(LTR == 1) / len(LTR), prior=0.1)
        if current_minDCF < minValue_minDCF['min']:
            minValue_minDCF['min'] = current_minDCF
            minValue_minDCF['lambda'] = ldb
            minValue_minDCF['expanded'] = True
        if current_actDCF < minValue_actDCF['min']:
            minValue_actDCF['min'] = current_actDCF
            minValue_actDCF['lambda'] = ldb
            minValue_actDCF['expanded'] = True
        actDCF.append(current_actDCF)
        minDCF.append(current_minDCF)

    plt.plot_minDCF_actDCF(minDCF, actDCF, 'Quadratic Logistic Regression with expanded features', ldbArray)
    plt.plot_minDCF_actDCF(minDCF, actDCF, 'Quadratic Logistic Regression with expanded features', ldbArray, One=True)

    # apply PCA
    print("Binary Logistic regression with PCA")
    for m in range(5, 7):
        print("PCA with m=", m)
        s, P = PCA.PCA_function(DTR, m)
        DTR_PCA = np.dot(P.T, DTR)
        DVAL_PCA = np.dot(P.T, DVAL)

        actDCF = []
        minDCF = []
        for ldb in ldbArray:
            print("lambda: ", ldb)
            llr = LLR.LinearLogisticRegression(ldb, prior_weighted=False, prior=0.1)
            current_xf = llr.trainLogReg(DTR_PCA, LTR)
            current_minDCF, current_actDCF = LLR.compute_minDCF_actDCF(current_xf, LVAL, DVAL_PCA,
                                                                       pi_emp=sum(LTR == 1) / len(LTR),
                                                                       prior=0.1)
            if current_minDCF < minValue_minDCF['min']:
                minValue_minDCF['min'] = current_minDCF
                minValue_minDCF['lambda'] = ldb
                minValue_minDCF['PCA'] = m
            if current_actDCF < minValue_actDCF['min']:
                minValue_actDCF['min'] = current_actDCF
                minValue_actDCF['lambda'] = ldb
                minValue_actDCF['PCA'] = m
            actDCF.append(current_actDCF)
            minDCF.append(current_minDCF)

        plt.plot_minDCF_actDCF(minDCF, actDCF, 'Binary Logistic Regression with PCA with', ldbArray, m)

    # apply PCA and Z-normalization
    # print("Binary Logistic regression with PCA and Z-normalization")
    # scaler = StandardScaler().fit(DTR)
    # DTR_normalized = e.znormalized_features_training(DTR)
    # DVAL_normalized = e.znormalized_features_evaluation(DVAL, DTR)
    # for m in range(4, 7):
    #     s, P = PCA.PCA_function(DTR_normalized, m)
    #     DTR_PCA = np.dot(P.T, DTR_normalized)
    #     DVAL_PCA = np.dot(P.T, DVAL_normalized)
    #
    #     actDCF = []
    #     minDCF = []
    #     for ldb in ldbArray:
    #         print("lambda: ", ldb)
    #         llr = LLR.LinearLogisticRegression(ldb, prior_weighted=False, prior=0.1)
    #         current_xf = llr.trainLogReg(DTR_PCA, LTR)
    #         current_minDCF, current_actDCF = LLR.compute_minDCF_actDCF(current_xf, LVAL, DVAL_PCA,
    #                                                                    pi_emp=(LTR == 1).sum() / len(LTR),
    #                                                                    prior=0.1)
    #         if current_minDCF < minValue_minDCF['min']:
    #             minValue_minDCF['min'] = current_minDCF
    #             minValue_minDCF['lambda'] = ldb
    #             minValue_minDCF['PCA'] = m
    #             minValue_minDCF['Z-norm'] = True
    #         if current_actDCF < minValue_actDCF['min']:
    #             minValue_actDCF['min'] = current_actDCF
    #             minValue_actDCF['lambda'] = ldb
    #             minValue_actDCF['PCA'] = m
    #             minValue_actDCF['Z-norm'] = True
    #         actDCF.append(current_actDCF)
    #         minDCF.append(current_minDCF)
    #
    #     plt.plot_minDCF_actDCF(minDCF, actDCF, 'Binary Logistic Regression with Z-normalization and PCA with', ldbArray,
    #                            m)

    print('Result LLR')
    print("Best minDCF", minValue_minDCF)
    print("Best actDCF ", minValue_actDCF)

    # ########################
    # ##      SVM          ###
    # ########################
    # print("SVM")
    # parameters = {'prior': 0.1, 'Cfn': 1, 'Cfp': 1}
    # C_array = np.logspace(-5, 0, 11)
    # minDCF_Array = []
    # actDCF_Array = []
    # minValue_minDCF = {'C': 0, 'kernel': False, 'type': '', 'min': 10, 'gamma': 20}
    # minValue_actDCF = {'C': 0, 'kernel': False, 'type': '', 'min': 10, 'gamma': 20}
    # K = 1.0
    # # linear SVM
    # print("Linear SVM")
    # for C in C_array:
    #     hParam = {'K': K, 'C': C}
    #     print(hParam)
    #     svm = SVM.SVM(hParam, kernel=None, prior=0)
    #     svmReturn = svm.train(DTR, LTR)
    #
    #     predictions = svmReturn.predict(DVAL, labels=True)
    #     print("error rate", e.error_rate(predictions, LVAL))
    #     th = -np.log((parameters['prior'] * parameters['Cfn']) / ((1 - parameters['prior']) * parameters['Cfn']))
    #     llr = svmReturn.predict(DVAL)
    #     predictedLabels = np.int32(llr > th)
    #
    #     minDCF = bdm.compute_minDCF_binary(llr, LVAL, 0.1, 1, 1)
    #     minDCF_Array.append(minDCF)
    #     print("minDCF", minDCF)
    #     confusionMatrix = bdm.compute_confusion_matrix(predictedLabels, LVAL)
    #     actDCF = bdm.computeDCF_Binary(confusionMatrix, 0.1, 1, 1, normalize=True)
    #     actDCF_Array.append(actDCF)
    #     if minDCF < minValue_minDCF['min']:
    #         minValue_minDCF['min'] = minDCF
    #         minValue_minDCF['C'] = C
    #     if actDCF < minValue_actDCF['min']:
    #         minValue_actDCF['min'] = actDCF
    #         minValue_actDCF['C'] = C
    #     print("actDCF", actDCF)
    #     primal_value, dual_value = svmReturn.compute_primal_dual_value()
    #     print("primal value", primal_value)
    #     print("dual value", dual_value)
    #     print("duality gap", svmReturn.compute_duality_gap())
    # plt.plot_minDCF_actDCF(minDCF_Array, actDCF_Array, 'SVM linear', C_array, m=0, xlabel="C", One=True)
    # plt.plot_minDCF_actDCF(minDCF_Array, actDCF_Array, 'SVM linear', C_array, m=0, xlabel="C")
    #
    # print("SVM with polinomial kernel")
    # minDCF_Array = []
    # actDCF_Array = []
    # K = 0.0
    # d = 2
    # c = 1
    # for C in C_array:
    #     hParam = {'K': K, 'C': C, 'd': d, 'c': c, 'kernel': 'Polynomial'}
    #     print(hParam)
    #     svm = SVM.SVM(hParam, kernel='Polynomial', prior=0)
    #     svmReturn = svm.train(DTR, LTR)
    #     predictions = svmReturn.predict(DVAL, labels=True)
    #     print("error rate", e.error_rate(predictions, LVAL))
    #     th = -np.log((parameters['prior'] * parameters['Cfn']) / ((1 - parameters['prior']) * parameters['Cfn']))
    #     llr = svmReturn.predict(DVAL)
    #     predictedLabels = np.int32(llr > th)
    #     minDCF = bdm.compute_minDCF_binary(llr, LVAL, 0.1, 1, 1)
    #     minDCF_Array.append(minDCF)
    #     print("minDCF", minDCF)
    #     confusionMatrix = bdm.compute_confusion_matrix(predictedLabels, LVAL)
    #     actDCF = bdm.computeDCF_Binary(confusionMatrix, 0.1, 1, 1, normalize=True)
    #     actDCF_Array.append(actDCF)
    #     if minDCF < minValue_minDCF['min']:
    #         minValue_minDCF['min'] = minDCF
    #         minValue_minDCF['C'] = C
    #         minValue_minDCF['kernel'] = True
    #         minValue_minDCF['type'] = 'Polynomial'
    #     if actDCF < minValue_actDCF['min']:
    #         minValue_actDCF['min'] = actDCF
    #         minValue_actDCF['C'] = C
    #         minValue_actDCF['kernel'] = True
    #         minValue_actDCF['type'] = 'Polynomial'
    #     print("actDCF", actDCF)
    #     print("dual value", svmReturn.dual_value)
    # plt.plot_minDCF_actDCF(minDCF_Array, actDCF_Array, 'SVM polynomial', C_array, m=0, xlabel="C", One=True)
    # plt.plot_minDCF_actDCF(minDCF_Array, actDCF_Array, 'SVM polynomial', C_array, m=0, xlabel="C")
    #
    # print("SVM with RBF kernel")
    # minDCF_matrix = []
    # actDCF_matrix = []
    # C_array = np.logspace(-3, 2, 11)
    # K = 1.0
    # gamma_Array = np.array([1e-4, 1e-3, 1e-2, 1e-1])
    # # gamma_Array = np.array([1e-2, 1e-1, 1.0])
    # for gamma in gamma_Array:
    #     minDCF_values = []
    #     actDCF_values = []
    #     for c in C_array:
    #         hParam = {'K': K, 'C': c, 'gamma': gamma, 'kernel': 'RBF'}
    #         print(hParam)
    #         svm = SVM.SVM(hParam, kernel='RBF', prior=0)
    #         svmReturn = svm.train(DTR, LTR)
    #         print(svmReturn.alpha)
    #         predictions = svmReturn.predict(DVAL, labels=True)
    #
    #         print("error rate", e.error_rate(predictions, LVAL))
    #         th = -np.log((parameters['prior'] * parameters['Cfn']) / ((1 - parameters['prior']) * parameters['Cfn']))
    #         llr = svmReturn.predict(DVAL)
    #         predictedLabels = np.int32(llr > th)
    #         minDCF = bdm.compute_minDCF_binary(llr, LVAL, parameters['prior'], parameters['Cfn'], parameters['Cfn'])
    #         minDCF_values.append(minDCF)
    #         print("minDCF", minDCF)
    #         confusionMatrix = bdm.compute_confusion_matrix(predictedLabels, LVAL)
    #         actDCF = bdm.computeDCF_Binary(confusionMatrix, parameters['prior'], parameters['Cfn'], parameters['Cfn'],
    #                                        normalize=True)
    #         actDCF_values.append(actDCF)
    #         if minDCF < minValue_minDCF['min']:
    #             minValue_minDCF['min'] = minDCF
    #             minValue_minDCF['C'] = c
    #             minValue_minDCF['kernel'] = True
    #             minValue_minDCF['type'] = 'RBF'
    #             minValue_minDCF['gamma'] = gamma
    #         if actDCF < minValue_actDCF['min']:
    #             minValue_actDCF['min'] = actDCF
    #             minValue_actDCF['C'] = c
    #             minValue_actDCF['kernel'] = True
    #             minValue_actDCF['type'] = 'RBF'
    #             minValue_actDCF['gamma'] = gamma
    #         print("actDCF", actDCF)
    #         print("dual value", svmReturn.dual_value)
    #     minDCF_matrix.append(minDCF_values)
    #     actDCF_matrix.append(actDCF_values)
    # print("Best value for SVM")
    # print("Best minDCF", minValue_minDCF)
    # print("Best actDCF ", minValue_actDCF)
    # color = ['b', 'g', 'r', 'c']
    # pl.figure()
    # pl.title("SVM RBF kernel")
    # for i, minDCF_values in enumerate(minDCF_matrix):
    #     pl.plot(C_array, minDCF_values, color=color[i], label=str(gamma_Array[i]))
    #
    # pl.xlabel('C')
    # pl.ylabel('minDCF value')
    # pl.xscale('log', base=10)
    # pl.legend()
    # pl.show()
    #
    # color = ['b', 'g', 'r', 'c']
    # pl.figure()
    # pl.title("SVM RBF kernel")
    # for i, actDCF_values in enumerate(actDCF_matrix):
    #     pl.plot(C_array, actDCF_values, color=color[i], label=str(gamma_Array[i]))
    #
    # pl.xlabel('C')
    # pl.ylabel('actDCF value')
    # pl.xscale('log', base=10)
    # pl.legend()
    # pl.show()

    # ########################
    # ##       GMM         ###
    # ########################
    covTypes = ['Full', 'Diag']
    prior = 0.1
    nComponents = [1, 2, 4, 8, 16, 32]
    minValue_minDCF = {'cn': 0, 'covType': '', 'min': 10}
    minValue_actDCF = {'cn': 0, 'covType': '', 'min': 10}
    print("GMM with prior=0.1")
    for nc in nComponents:
        print("nComponents ", nc)
        for covType in covTypes:
            print(" covType is ", covType)
            gmm = GMM.GMM(alpha=0.1, nComponents=nc, psi=0.01, covType=covType)
            gmm.train(DTR, LTR)
            llr = gmm.predict(DVAL)
            minDCF = bdm.compute_minDCF_binary(llr, LVAL, 0.1, 1, 1)
            if minDCF < minValue_minDCF['min']: minValue_minDCF = {'cn': nc, 'covType': covType, 'min': minDCF}
            print("minDCF", minDCF)
            th = -np.log((prior * 1) / ((1 - prior) * 1))
            predictedLabels = np.int32(llr > th)
            confusionMatrix = bdm.compute_confusion_matrix(predictedLabels, LVAL)
            actDCF = bdm.computeDCF_Binary(confusionMatrix, 0.1, 1, 1, normalize=True)
            if actDCF < minValue_actDCF['min']: minValue_actDCF = {'cn': nc, 'covType': covType, 'min': actDCF}
            print("actDCF", actDCF)

    print("minimum value of minDCF", minValue_minDCF)
    print("minimum value of actDCF", minValue_actDCF)

    print("GMM with different log odds")
    log_odds_values = np.linspace(-4, 4, 21)
    minDCF_values = []
    actDCF_values = []
    minValue_minDCF = {'cn': 0, 'covType': '', 'min': 10}
    minValue_actDCF = {'cn': 0, 'covType': '', 'min': 10}

    for covType in covTypes:
        print(" covType is ", covType)
        for nc in nComponents:
            print("nComponents ", nc)
            gmm = GMM.GMM(alpha=0.1, nComponents=nc, psi=0.01, covType=covType)
            gmm.train(DTR, LTR)
            for log_odds in log_odds_values:
                prior = 1 / (1 + np.exp(-log_odds))
                llr = gmm.predict(DVAL)
                minDCF = bdm.compute_minDCF_binary(llr, LVAL, prior, 1, 1)
                print("minDCF", minDCF)
                if minDCF < minValue_minDCF['min']: minValue_minDCF = {'cn': nc, 'covType': covType, 'min': minDCF}
                minDCF_values.append(minDCF)
                predictions = gmm.predict(DVAL, labels=True)
                th = -np.log((prior * 1) / ((1 - prior) * 1))
                predictedLabels = np.int32(llr > th)
                confusionMatrix = bdm.compute_confusion_matrix(predictedLabels, LVAL)
                actDCF = bdm.computeDCF_Binary(confusionMatrix, prior, 1, 1, normalize=True)
                if actDCF < minValue_actDCF['min']: minValue_actDCF = {'cn': nc, 'covType': covType, 'min': actDCF}
                print("actDCF", actDCF)
                actDCF_values.append(actDCF)
            pl.figure()
            pl.plot(log_odds_values, minDCF_values, 'b')
            pl.plot(log_odds_values, actDCF_values, 'r')
            pl.ylim([None, 0.4])
            pl.xlabel('Log Odds')
            pl.ylabel('DCF values')
            pl.title('minDCF vs Log Odds with nComponents = ' + str(nc) + ' and covType ' + covType)
            pl.legend(['minDCF', 'actDCF'])
            pl.show()
            minDCF_values = []
            actDCF_values = []

    print("minimum value of minDCF", minValue_minDCF)
    print("minimum value of actDCF", minValue_actDCF)

    ##########################
    # ##   CALIBRATION     ###
    # ########################
    # print("Calibration")
    # pT = 0.1
    # SAMEFIGPLOTS = True
    # if SAMEFIGPLOTS:
    #     fig = pl.figure(figsize=(16, 9))
    #     axes = fig.subplots(2, 2, sharex='all')
    #     fig.suptitle('Single fold')
    # else:
    #     axes = np.array([[pl.figure().gca(), pl.figure().gca(), pl.figure().gca()],
    #                      [pl.figure().gca(), pl.figure().gca(), pl.figure().gca()],
    #                      [None, pl.figure().gca(), pl.figure().gca()]])
    #
    # print("QLR")
    # qlr = QLR.QuadraticLogisticRegression(0.00031622776601683794, prior_weighted=False, prior=pT)
    # xf = qlr.train(DTR, LTR)
    # w, b = xf[:-1], xf[-1]
    # scoreQLR = qlr.calculateS(DVAL)
    # th = -np.log((pT * 1) / ((1 - pT) * 1))
    # labelQLR = qlr.predictThreshold(DVAL, th)
    #
    # minDCF = bdm.compute_minDCF_binary(scoreQLR, LVAL, pT, 1, 1)
    # confusionMatrix = bdm.compute_confusion_matrix(np.int32(scoreQLR > th), LVAL)
    # actDCF = bdm.computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    # print("minDCF no calibrated", minDCF)
    # print("actDCF no calibrated", actDCF)
    # logOdds, actDCF, minDCF = plt.bayesPlot(scoreQLR, LVAL)
    # axes[0, 0].plot(logOdds, minDCF, color='C0', linestyle=':', label='minDCF no cal')
    # axes[0, 0].plot(logOdds, actDCF, color='C0', linestyle='-.', label='actDCF no cal')

    # midpoint = len(scoreQLR) // 3
    # S_CAL, S_VAL = scoreQLR[:midpoint], scoreQLR[midpoint:]
    # L_CAL, L_VAL = LVAL[:midpoint], LVAL[midpoint:]
    #
    # # Trasforma i punteggi di calibrazione per usarli con la LLR
    # S_CAL = S_CAL.reshape(1, -1)
    # S_VAL = S_VAL.reshape(1, -1)
    #
    # xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=pT).trainLogReg(S_CAL, L_CAL)
    # w, b = xf[:-1], xf[-1]
    #
    # calibrated_SVAL = (w.T @ S_VAL + b - np.log(pT / (1 - pT))).ravel()
    #
    # minDCF_calibrated = bdm.compute_minDCF_binary(calibrated_SVAL, L_VAL, pT, 1, 1)
    # th = -np.log((pT * 1) / ((1 - pT) * 1))
    # confusionMatrix_calibrated = bdm.compute_confusion_matrix(np.int32(calibrated_SVAL > th), L_VAL)
    # actDCF_calibrated = bdm.computeDCF_Binary(confusionMatrix_calibrated, pT, 1, 1, normalize=True)
    # print("Single fold")
    # print(f"minDCF Calibrated: {minDCF_calibrated}")
    # print(f"actDCF Calibrated: {actDCF_calibrated}")
    # logOdds, actDCF, minDCF = plt.bayesPlot(calibrated_SVAL, L_VAL)
    # axes[0, 0].plot(logOdds, minDCF, color='C0', linestyle='--', label='minDCF cal')
    # axes[0, 0].plot(logOdds, actDCF, color='C0', linestyle='-', label='actDCF cal')
    # axes[0, 0].set_ylim(0.0, 0.8)
    # axes[0, 0].set_title('QLR - calibration validation')
    # axes[0, 0].legend()
    # pl.show()
    #
    # print("K-fold")
    # KFOLD = 5
    # calibrated_value_QLR = []
    # calibrated_label_QLR = []
    # for foldIdx in range(KFOLD):
    #     SCAL, SVAL = e.extract_train_val_folds_from_ary(scoreQLR, foldIdx)
    #     labels_CAL, labels_VAL = e.extract_train_val_folds_from_ary(LVAL, foldIdx)
    #     xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=pT).trainLogReg(
    #         rd.vrow(SCAL), labels_CAL)
    #     w, b = xf[:-1], xf[-1]
    #     calibrated_SVAL = (w.T @ rd.vrow(SVAL) + b - np.log(pT / (1 - pT))).ravel()
    #     calibrated_value_QLR.append(calibrated_SVAL)
    #     calibrated_label_QLR.append(labels_VAL)
    #
    # calibrated_value_QLR = np.hstack(calibrated_value_QLR)
    # calibrated_label_QLR = np.hstack(calibrated_label_QLR)
    # minDCF_calibrated = bdm.compute_minDCF_binary(calibrated_value_QLR, calibrated_label_QLR, pT, 1, 1)
    # confusionMatrix_calibrated = bdm.compute_confusion_matrix(np.int32(calibrated_value_QLR > th), calibrated_label_QLR)
    # actDCF_calibrated = bdm.computeDCF_Binary(confusionMatrix_calibrated, pT, 1, 1, normalize=True)
    # print(f"minDCF Calibrated: {minDCF_calibrated}")
    # print(f"actDCF Calibrated: {actDCF_calibrated}")
    #
    # logOdds, actDCF, minDCF = plt.bayesPlot(calibrated_value_QLR, calibrated_label_QLR)
    # axes[0, 0].plot(logOdds, minDCF, color='C0', linestyle='--', label='minDCF cal')
    # axes[0, 0].plot(logOdds, actDCF, color='C0', linestyle='-', label='actDCF cal')
    # axes[0, 0].set_ylim(0.0, 0.8)
    # axes[0, 0].set_title('QLR - calibration validation')
    # axes[0, 0].legend()
    #
    # print("SVM")
    # hParam = {'K': 1, 'C': 100.0, 'kernel': 'RBF', 'gamma': 0.1}
    # svm = SVM.SVM(hParam, kernel='RBF', prior=0)
    # svmReturn = svm.train(DTR, LTR)
    # scoreSVM = svmReturn.predict(DVAL)
    # labelSVM = svmReturn.predict(DVAL, labels=True)
    #
    # minDCF = bdm.compute_minDCF_binary(scoreSVM, LVAL, pT, 1, 1)
    # confusionMatrix = bdm.compute_confusion_matrix(np.int32(scoreSVM > th), LVAL)
    # actDCF = bdm.computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    # print("minDCF no calibrated", minDCF)
    # print("actDCF no calibrated", actDCF)
    # logOdds, actDCF, minDCF = plt.bayesPlot(scoreSVM, LVAL)
    # axes[0, 1].plot(logOdds, minDCF, color='C1', linestyle=':', label='minDCF no cal')
    # axes[0, 1].plot(logOdds, actDCF, color='C1', linestyle='-.', label='actDCF no cal')
    #
    # print("K-fold")
    # calibrated_value_SVM = []
    # calibrated_label_SVM = []
    # for foldIdx in range(KFOLD):
    #     SCAL, SVAL = e.extract_train_val_folds_from_ary(scoreSVM, foldIdx)
    #     labels_CAL, labels_VAL = e.extract_train_val_folds_from_ary(LVAL, foldIdx)
    #
    #     xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=pT).trainLogReg(
    #         rd.vrow(SCAL), labels_CAL)
    #     w, b = xf[:-1], xf[-1]
    #     calibrated_SVAL = (w.T @ rd.vrow(SVAL) + b - np.log(pT / (1 - pT))).ravel()
    #     calibrated_value_SVM.append(calibrated_SVAL)
    #     calibrated_label_SVM.append(labels_VAL)
    #
    # calibrated_value_SVM = np.hstack(calibrated_value_SVM)
    # calibrated_label_SVM = np.hstack(calibrated_label_SVM)
    # minDCF_calibrated = bdm.compute_minDCF_binary(calibrated_value_SVM, calibrated_label_SVM, pT, 1, 1)
    # confusionMatrix_calibrated = bdm.compute_confusion_matrix(np.int32(calibrated_value_SVM > th), calibrated_label_SVM)
    # actDCF_calibrated = bdm.computeDCF_Binary(confusionMatrix_calibrated, pT, 1, 1, normalize=True)
    # print(f"minDCF Calibrated: {minDCF_calibrated}")
    # print(f"actDCF Calibrated: {actDCF_calibrated}")
    #
    # logOdds, actDCF, minDCF = plt.bayesPlot(calibrated_value_SVM, calibrated_label_SVM)
    # axes[0, 1].plot(logOdds, minDCF, color='C1', linestyle='--', label='minDCF cal')
    # axes[0, 1].plot(logOdds, actDCF, color='C1', linestyle='-', label='actDCF cal')
    # axes[0, 1].set_ylim(0.0, 0.8)
    # axes[0, 1].set_title('SVM - calibration validation')
    # axes[0, 1].legend()
    #
    # print("GMM")
    # covType = 'Diag'
    # nComponents = 32
    # gmm = GMM.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=covType)
    # gmm.train(DTR, LTR)
    # scoreGMM = gmm.predict(DVAL)
    # labelGMM = gmm.predict(DVAL, labels=True)
    # minDCF = bdm.compute_minDCF_binary(scoreGMM, LVAL, pT, 1, 1)
    # confusionMatrix = bdm.compute_confusion_matrix(labelGMM, LVAL)
    # actDCF = bdm.computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    # print("minDCF no calibrated", minDCF)
    # print("actDCF no calibrated", actDCF)
    # logOdds, actDCF, minDCF = plt.bayesPlot(scoreGMM, LVAL)
    # axes[1, 0].plot(logOdds, minDCF, color='C2', linestyle='--', label='minDCF no cal')
    # axes[1, 0].plot(logOdds, actDCF, color='C2', linestyle='-.', label='actDCF no cal')
    #
    # print("K-fold")
    # calibrated_value_GMM = []
    # calibrated_label_GMM = []
    # for foldIdx in range(KFOLD):
    #     SCAL, SVAL = e.extract_train_val_folds_from_ary(scoreGMM, foldIdx)
    #     labels_CAL, labels_VAL = e.extract_train_val_folds_from_ary(LVAL, foldIdx)
    #
    #     xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=pT).trainLogReg(
    #         rd.vrow(SCAL), labels_CAL)
    #     w, b = xf[:-1], xf[-1]
    #     calibrated_SVAL = (w.T @ rd.vrow(SVAL) + b - np.log(pT / (1 - pT))).ravel()
    #     calibrated_value_GMM.append(calibrated_SVAL)
    #     calibrated_label_GMM.append(labels_VAL)
    #
    # calibrated_value_GMM = np.hstack(calibrated_value_GMM)
    # calibrated_label_GMM = np.hstack(calibrated_label_GMM)
    # minDCF_calibrated = bdm.compute_minDCF_binary(calibrated_value_GMM, calibrated_label_GMM, pT, 1, 1)
    # confusionMatrix_calibrated = bdm.compute_confusion_matrix(np.int32(calibrated_value_GMM > th), calibrated_label_GMM)
    # actDCF_calibrated = bdm.computeDCF_Binary(confusionMatrix_calibrated, pT, 1, 1, normalize=True)
    # print(f"minDCF Calibrated: {minDCF_calibrated}")
    # print(f"actDCF Calibrated: {actDCF_calibrated}")
    #
    # logOdds, actDCF, minDCF = plt.bayesPlot(calibrated_value_GMM, calibrated_label_GMM)
    # axes[1, 0].plot(logOdds, minDCF, color='C2', linestyle='--', label='minDCF cal')
    # axes[1, 0].plot(logOdds, actDCF, color='C2', linestyle='-', label='actDCF cal')
    # axes[1, 0].set_ylim(0.0, 0.4)
    # axes[1, 0].set_title('GMM - calibration validation')
    # axes[1, 0].legend()
    #
    # print("Fusion")
    # fusedScore = []
    # fusedLabels = []
    #
    # for foldIdx in range(KFOLD):
    #     SCAL_QLR, SVAL_QLR = e.extract_train_val_folds_from_ary(scoreQLR, foldIdx)
    #     SCAL_SVM, SVAL_SVM = e.extract_train_val_folds_from_ary(scoreSVM, foldIdx)
    #     SCAL_GMM, SVAL_GMM = e.extract_train_val_folds_from_ary(scoreGMM, foldIdx)
    #     labels_CAL, labels_VAL = e.extract_train_val_folds_from_ary(LVAL, foldIdx)
    #
    #     SCAL = np.vstack([SCAL_QLR, SCAL_SVM, SCAL_GMM])
    #
    #     xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=pT).trainLogReg(SCAL, labels_CAL)
    #     w, b = xf[:-1], xf[-1]
    #     SVAL = np.vstack([SVAL_QLR, SVAL_SVM, SVAL_GMM])
    #     calibrated_SVAL = (w.T @ SVAL + b - np.log(pT / (1 - pT))).ravel()
    #
    #     fusedScore.append(calibrated_SVAL)
    #     fusedLabels.append(labels_VAL)
    #
    # fusedScore = np.hstack(fusedScore)
    # fusedLabels = np.hstack(fusedLabels)
    # minDCF_calibrated = bdm.compute_minDCF_binary(fusedScore, fusedLabels, pT, 1, 1)
    # confusionMatrix_calibrated = bdm.compute_confusion_matrix(np.int32(fusedScore > th), fusedLabels)
    # actDCF_calibrated = bdm.computeDCF_Binary(confusionMatrix_calibrated, pT, 1, 1, normalize=True)
    # print(f"minDCF Calibrated: {minDCF_calibrated}")
    # print(f"actDCF Calibrated: {actDCF_calibrated}")
    #
    # # now do comparison with target application
    # logOdds, actDCF, minDCF = plt.bayesPlot(calibrated_value_QLR, calibrated_label_QLR)
    # axes[1, 1].set_title('Fusion - validation')
    # axes[1, 1].plot(logOdds, minDCF, color='C0', linestyle='--', label='QLR-minDCF')
    # axes[1, 1].plot(logOdds, actDCF, color='C0', linestyle='-', label='QLR-actDCF')
    # logOdds, actDCF, minDCF = plt.bayesPlot(calibrated_value_SVM, calibrated_label_SVM)
    # axes[1, 1].plot(logOdds, minDCF, color='C1', linestyle='--', label='SVM-minDCF')
    # axes[1, 1].plot(logOdds, actDCF, color='C1', linestyle='-', label='SVM-actDCF')
    # logOdds, actDCF, minDCF = plt.bayesPlot(calibrated_value_GMM, calibrated_label_GMM)
    # axes[1, 1].plot(logOdds, minDCF, color='C2', linestyle='--', label='GMM-minDCF')
    # axes[1, 1].plot(logOdds, actDCF, color='C2', linestyle='-', label='GMM-actDCF')
    #
    # logOdds, actDCF, minDCF = plt.bayesPlot(fusedScore, fusedLabels)
    # axes[1, 1].plot(logOdds, minDCF, color='C3', linestyle='--', label='Fusion-minDCF')
    # axes[1, 1].plot(logOdds, actDCF, color='C3', linestyle='-', label='Fusion-actDCF')
    # axes[1, 1].set_ylim(0.0, 0.45)
    # axes[1, 1].legend()
    # pl.show()
    #
    # # ########################
    # # ##    EVALUATION     ###
    # # ########################
    # print("Evaluation")
    #
    # scoreQLR_eval = qlr.calculateS(evalData)
    # labelQLR_eval = qlr.predictThreshold(evalData, th)
    #
    # print("QLR")
    # minDCF = bdm.compute_minDCF_binary(scoreQLR_eval, evalLabels, pT, 1, 1)
    # confusionMatrix = bdm.compute_confusion_matrix(np.int32(scoreQLR_eval > th), evalLabels)
    # actDCF = bdm.computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    # print("minDCF no calibrated", minDCF)
    # print("actDCF no calibrated", actDCF)
    #
    # # calibrated
    # xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=pT).trainLogReg(rd.vrow(scoreQLR), LVAL)
    # w, b = xf[:-1], xf[-1]
    # calibrated_scoreQLR_eval = (w.T @ rd.vrow(scoreQLR_eval) + b - np.log(pT / (1 - pT))).ravel()
    #
    # minDCF_calibrated_eval = bdm.compute_minDCF_binary(calibrated_scoreQLR_eval, evalLabels, pT, 1, 1)
    # confusionMatrix_calibrated = bdm.compute_confusion_matrix(np.int32(calibrated_scoreQLR_eval > th), evalLabels)
    # actDCF_calibrated_eval = bdm.computeDCF_Binary(confusionMatrix_calibrated, pT, 1, 1, normalize=True)
    # print("minDCF Calibrated: ", minDCF_calibrated_eval)
    # print("actDCF Calibrated: ", actDCF_calibrated_eval)
    #
    # print("GMM")
    # scoreGMM_eval = gmm.predict(evalData)
    # labelGMM_eval = gmm.predict(evalData, labels=True)
    # minDCF = bdm.compute_minDCF_binary(scoreGMM_eval, evalLabels, pT, 1, 1)
    # confusionMatrix = bdm.compute_confusion_matrix(labelGMM_eval, evalLabels)
    # actDCF = bdm.computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    # print("minDCF no calibrated", minDCF)
    # print("actDCF no calibrated", actDCF)
    #
    # xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=pT).trainLogReg(rd.vrow(scoreGMM), LVAL)
    # w, b = xf[:-1], xf[-1]
    # calibrated_scoreGMM_eval = (w.T @ rd.vrow(scoreGMM_eval) + b - np.log(pT / (1 - pT))).ravel()
    #
    # minDCF_calibrated_eval = bdm.compute_minDCF_binary(calibrated_scoreGMM_eval, evalLabels, pT, 1, 1)
    # confusionMatrix_calibrated = bdm.compute_confusion_matrix(np.int32(calibrated_scoreGMM_eval > th), evalLabels)
    # actDCF_calibrated_eval = bdm.computeDCF_Binary(confusionMatrix_calibrated, pT, 1, 1, normalize=True)
    # print("minDCF Calibrated: ", minDCF_calibrated_eval)
    # print("actDCF Calibrated: ", actDCF_calibrated_eval)
    #
    # print("SVM")
    # scoreSVM_eval = svmReturn.predict(evalData)
    # labelSVM_eval = svmReturn.predict(evalData, labels=True)
    #
    # minDCF = bdm.compute_minDCF_binary(scoreSVM_eval, evalLabels, pT, 1, 1)
    # confusionMatrix = bdm.compute_confusion_matrix(np.int32(scoreSVM_eval > th), evalLabels)
    # actDCF = bdm.computeDCF_Binary(confusionMatrix, pT, 1, 1, normalize=True)
    # print("minDCF no calibrated", minDCF)
    # print("actDCF no calibrated", actDCF)
    #
    # xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=pT).trainLogReg(rd.vrow(scoreSVM), LVAL)
    # w, b = xf[:-1], xf[-1]
    # calibrated_scoreSVM_eval = (w.T @ rd.vrow(scoreSVM_eval) + b - np.log(pT / (1 - pT))).ravel()
    #
    # minDCF_calibrated_eval = bdm.compute_minDCF_binary(calibrated_scoreSVM_eval, evalLabels, pT, 1, 1)
    # confusionMatrix_calibrated = bdm.compute_confusion_matrix(np.int32(calibrated_scoreSVM_eval > th), evalLabels)
    # actDCF_calibrated_eval = bdm.computeDCF_Binary(confusionMatrix_calibrated, pT, 1, 1, normalize=True)
    # print("minDCF Calibrated: ", minDCF_calibrated_eval)
    # print("actDCF Calibrated: ", actDCF_calibrated_eval)
    #
    # print("Fusion")
    # fusion_score = np.vstack([scoreQLR, scoreSVM, scoreGMM])
    #
    # xf = LLR.LinearLogisticRegression(0, prior_weighted=True, prior=pT).trainLogReg(fusion_score, LVAL)
    # w, b = xf[:-1], xf[-1]
    # score_eval = np.vstack([scoreQLR_eval, scoreSVM_eval, scoreGMM_eval])
    # calibrated_SVAL = (w.T @ score_eval + b - np.log(pT / (1 - pT))).ravel()
    #
    # minDCF_calibrated_eval = bdm.compute_minDCF_binary(calibrated_SVAL, evalLabels, pT, 1, 1)
    # confusionMatrix_calibrated = bdm.compute_confusion_matrix(np.int32(calibrated_SVAL > th), evalLabels)
    # actDCF_calibrated_eval = bdm.computeDCF_Binary(confusionMatrix_calibrated, pT, 1, 1, normalize=True)
    # print("minDCF Calibrated: ", minDCF_calibrated_eval)
    # print("actDCF Calibrated: ", actDCF_calibrated_eval)

import numpy as np

import GaussianModel as gm
import PCA
import ReadData as rd

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
    gm.calculate_MVG(DTR, LTR, DVAL, LVAL)
    gm.calculate_Naive(DTR, LTR, DVAL, LVAL)
    gm.calculate_Tied(DTR, LTR, DVAL, LVAL)

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

    # apply PCA+MVG, m=1
    s, P = PCA.PCA_function(D, 1)

    print("PCA+MVG, m=1")
    DTR_PCA = np.dot(P.T, DTR)
    DVAL_PCA = np.dot(P.T, DVAL)
    gm.calculate_MVG(DTR_PCA, LTR, DVAL_PCA, LVAL)
    gm.calculate_Naive(DTR_PCA, LTR, DVAL_PCA, LVAL)
    gm.calculate_Tied(DTR_PCA, LTR, DVAL_PCA, LVAL)

    # apply PCA+MVG, m=2
    s, P = PCA.PCA_function(D, 2)

    print("PCA+MVG, m=2")
    DTR_PCA = np.dot(P.T, DTR)
    DVAL_PCA = np.dot(P.T, DVAL)
    gm.calculate_MVG(DTR_PCA, LTR, DVAL_PCA, LVAL)
    gm.calculate_Naive(DTR_PCA, LTR, DVAL_PCA, LVAL)
    gm.calculate_Tied(DTR_PCA, LTR, DVAL_PCA, LVAL)

    # apply PCA+MVG, m=2
    s, P = PCA.PCA_function(D, 3)

    print("PCA+MVG, m=3")
    DTR_PCA = np.dot(P.T, DTR)
    DVAL_PCA = np.dot(P.T, DVAL)
    gm.calculate_MVG(DTR_PCA, LTR, DVAL_PCA, LVAL)
    gm.calculate_Naive(DTR_PCA, LTR, DVAL_PCA, LVAL)
    gm.calculate_Tied(DTR_PCA, LTR, DVAL_PCA, LVAL)

    # apply PCA+MVG, m=2
    s, P = PCA.PCA_function(D, 4)

    print("PCA+MVG, m=4")
    DTR_PCA = np.dot(P.T, DTR)
    DVAL_PCA = np.dot(P.T, DVAL)
    gm.calculate_MVG(DTR_PCA, LTR, DVAL_PCA, LVAL)
    gm.calculate_Naive(DTR_PCA, LTR, DVAL_PCA, LVAL)
    gm.calculate_Tied(DTR_PCA, LTR, DVAL_PCA, LVAL)

    # apply PCA+MVG, m=5
    s, P = PCA.PCA_function(D, 5)

    print("PCA+MVG, m=5")
    DTR_PCA = np.dot(P.T, DTR)
    DVAL_PCA = np.dot(P.T, DVAL)
    gm.calculate_MVG(DTR_PCA, LTR, DVAL_PCA, LVAL)
    gm.calculate_Naive(DTR_PCA, LTR, DVAL_PCA, LVAL)
    gm.calculate_Tied(DTR_PCA, LTR, DVAL_PCA, LVAL)

    # apply PCA+MVG, m=6
    s, P = PCA.PCA_function(D, 6)

    print("PCA+MVG, m=6")
    DTR_PCA = np.dot(P.T, DTR)
    DVAL_PCA = np.dot(P.T, DVAL)
    gm.calculate_MVG(DTR_PCA, LTR, DVAL_PCA, LVAL)
    gm.calculate_Naive(DTR_PCA, LTR, DVAL_PCA, LVAL)
    gm.calculate_Tied(DTR_PCA, LTR, DVAL_PCA, LVAL)

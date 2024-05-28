import numpy as np
import scipy

import GaussianDensity as gd
import ReadData as ut


def compute_log_likelihood(D, hParams):
    S = np.zeros((len(hParams), D.shape[1]))
    for lab in range(S.shape[0]):
        S[lab, :] = gd.logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])
    return S


def compute_mu_c_MVG(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L == lab]
        hParams[lab] = gd.compute_mu_C(DX)
    return hParams


def compute_mu_C_Tied(D, L):
    labelSet = set(L)
    hParams = {}
    hMeans = {}
    CGlobal = 0
    for lab in labelSet:
        DX = D[:, L == lab]
        mu, C_class = gd.compute_mu_C(DX)
        # DX.shape[1] è il numero di campioni di quella classe
        CGlobal += C_class * DX.shape[1]
        hMeans[lab] = mu
    # qui viene diviso per il numero totale di campioni
    CGlobal = CGlobal / D.shape[1]
    # viene semplicemente assegnato lo stesso valore di covarianza a tutte le classi
    for lab in labelSet:
        hParams[lab] = (hMeans[lab], CGlobal)
    return hParams


def compute_mu_C_Naive(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L == lab]
        mu, C = gd.compute_mu_C(DX)
        # C moltiplicato per la matrice identità
        hParams[lab] = (mu, C * np.eye(D.shape[0]))
    return hParams


def compute_logPosterior(S_logLikelihood, v_prior):
    # probabilità congiunta
    SJoint = S_logLikelihood + ut.vcol(np.log(v_prior))
    # probabilita marginale che è uguale al prodotto delle probabilità congiunte
    SMarginal = ut.vrow(scipy.special.logsumexp(SJoint, axis=0))
    # probabilità a posteriori, sottrai la probabilità marginale dalla probabilità congiunta in modo che tutti abbiamo probabilità massimo 1
    SPost = SJoint - SMarginal
    return SPost


def calculate_MVG(DTR, LTR, DVAL, LVAL):
    hParams_MVG = compute_mu_c_MVG(DTR, LTR)
    LLR = gd.logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - gd.logpdf_GAU_ND(DVAL, hParams_MVG[0][0],
                                                                                          hParams_MVG[0][1])
    PVAL = gd.predict_labels(DVAL=DVAL, TH=0, LLR=LLR, class1=0, class2=1)
    print("MVG 2-Class problem - Error rate: {:.6f}%".format(gd.error_rate(PVAL, LVAL)))
    return LLR


def calculate_Tied(DTR, LTR, DVAL, LVAL):
    hParams_Tied = compute_mu_C_Tied(DTR, LTR)
    LLR = gd.logpdf_GAU_ND(DVAL, hParams_Tied[1][0], hParams_Tied[1][1]) - gd.logpdf_GAU_ND(DVAL, hParams_Tied[0][0],
                                                                                            hParams_Tied[0][1])
    PVAL = gd.predict_labels(DVAL=DVAL, TH=0, LLR=LLR, class1=0, class2=1)
    print("Tied 2-Class problem - Error rate: {:.6f}%".format(gd.error_rate(PVAL, LVAL)))
    return LLR


def calculate_Naive(DTR, LTR, DVAL, LVAL):
    hParams_Naive = compute_mu_C_Naive(DTR, LTR)
    LLR = gd.logpdf_GAU_ND(DVAL, hParams_Naive[1][0], hParams_Naive[1][1]) - gd.logpdf_GAU_ND(DVAL, hParams_Naive[0][0],
                                                                                              hParams_Naive[0][1])
    PVAL = gd.predict_labels(DVAL=DVAL, TH=0, LLR=LLR, class1=0, class2=1)
    print("Naive 2-Class problem - Error rate: {:.6f}%".format(gd.error_rate(PVAL, LVAL)))
    return LLR


def correlation(DTR, LTR):
    hParams_MVG = compute_mu_c_MVG(DTR, LTR)

    C0 = hParams_MVG[0][1]
    C1 = hParams_MVG[1][1]

    print("C0\n", C0)
    print("C1\n", C1)

    Corr0 = C0 / (ut.vcol(C0.diagonal() ** 0.5) * ut.vrow(C0.diagonal() ** 0.5))
    Corr1 = C1 / (ut.vcol(C1.diagonal() ** 0.5) * ut.vrow(C1.diagonal() ** 0.5))
    for i in range(Corr0.shape[0]):
        row_Corr0 = ' '.join('{:<10.2f}'.format(x) for x in Corr0[i])
        print("Corr0[{}]: {} ".format(i, row_Corr0))
    print("\n")
    for i in range(Corr1.shape[0]):
        row_Corr1 = ' '.join('{:<10.2f}'.format(x) for x in Corr1[i])
        print(" Corr1[{}]: {}".format(i, row_Corr1))

    return Corr0, Corr1

import numpy as np

import utils as ut

if __name__ == '__main__':
    D, L = ut.load('iris.csv')
    (DTR, LTR), (DTE, LTE) = ut.split_db_2to1(D, L)
    Pc = 1 / 3.

    # MVG
    hParams_MVG = ut.compute_mu_c_MVG(DTR, LTR)
    for lab in [0, 1, 2]:
        print('MVG - Class', lab)
        print(hParams_MVG[lab][0])
        print(hParams_MVG[lab][1])
        print()

    # questo deve essere cacolato sul dataset di test
    S = ut.compute_log_likelihood(DTE, hParams_MVG)
    v_prior = np.ones(3) * Pc
    S_logPost = ut.compute_logPosterior(S, v_prior)
    S_solution_Joint = np.load('solution/logPosterior_MVG.npy')
    print("Max absolute error w.r.t. pre-computed solution - log-posterior matrix")
    print(np.abs(S_logPost - S_solution_Joint).max())

    # Predict labels
    PVAL = S_logPost.argmax(0)
    # questo sta stampando l'errore che c'è tra le label predette e quelle reali
    print("MVG - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))

    # Naive
    hParams_Naive = ut.compute_mu_C_Naive(DTR, LTR)
    S = ut.compute_log_likelihood(DTE, hParams_Naive)
    S_logPost = ut.compute_logPosterior(S, v_prior)
    PVAL = S_logPost.argmax(0)
    print("Naive Bayes Gaussian - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))

    # Tied
    hParams_Tied = ut.compute_mu_C_Tied(DTR, LTR)
    S = ut.compute_log_likelihood(DTE, hParams_Tied)
    S_logPost = ut.compute_logPosterior(S, v_prior)
    PVAL = S_logPost.argmax(0)
    print("Tied Gaussian - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))

    # 2-Class problem
    D = D[:, L != 0]
    L = L[L != 0]
    (DTR, LTR), (DVAL, LVAL) = ut.split_db_2to1(D, L)

    # calculate llr
    hParams_MVG = ut.compute_mu_c_MVG(DTR, LTR)
    LLR = ut.logpdf_GAU_ND(DVAL, hParams_MVG[2][0], hParams_MVG[2][1]) - ut.logpdf_GAU_ND(DVAL, hParams_MVG[1][0],
                                                                                          hParams_MVG[1][1])
    # predict labels, based on the threshold
    PVAL = ut.predict_labels(DVAL=DVAL, TH=0, LLR=LLR, class1=1, class2=2)

    # calculate error rate
    print("MVG 2-Class problem - Error rate: %.1f%%" % ut.error_rate(PVAL, LVAL))

    # Naive
    hParams_Naive = ut.compute_mu_C_Naive(DTR, LTR)
    LLR = ut.logpdf_GAU_ND(DVAL, hParams_Naive[2][0], hParams_Naive[2][1]) - ut.logpdf_GAU_ND(DVAL, hParams_Naive[1][0],
                                                                                              hParams_Naive[1][1])
    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 2
    PVAL[LLR < TH] = 1
    print("Naive 2-Class problem - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))

    # Tied
    hParams_Tied = ut.compute_mu_C_Tied(DTR, LTR)
    LLR = ut.logpdf_GAU_ND(DVAL, hParams_Tied[2][0], hParams_Tied[2][1]) - ut.logpdf_GAU_ND(DVAL, hParams_Tied[1][0],
                                                                                            hParams_Tied[1][1])
    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 2
    PVAL[LLR < TH] = 1
    print("Tied 2-Class problem - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))

import numpy as np
import scipy.optimize

import BayesDecisionModel as bdm
import Evaluation as e


class LinearLogisticRegression:
    def __init__(self, lbd, prior_weighted=False, prior=0.5):
        self.lbd = lbd
        self.prior_weighted = prior_weighted
        self.prior = prior

    def __logreg_obj(self, v):
        w, b = v[0:-1], v[-1]
        ZTR = 2 * self.LTR - 1
        reg = 0.5 * self.lbd * np.linalg.norm(w) ** 2
        exp = (np.dot(w.T, self.DTR) + b)
        avg_risk = (np.logaddexp(0, -exp * ZTR)).mean()
        return reg + avg_risk

    def __logreg_obj_prior_weighted(self, v):
        w, b = v[0:-1], v[-1]
        ZTR = 2 * self.LTR - 1

        wTrue = self.prior / (ZTR > 0).sum()
        wFalse = (1 - self.prior) / (ZTR < 0).sum()

        reg = 0.5 * self.lbd * np.linalg.norm(w) ** 2
        exp = (np.dot(w.T, self.DTR) + b)
        avg_risk_0 = (np.logaddexp(0, -exp[self.LTR == 0] * ZTR[self.LTR == 0]) * wFalse).sum()
        avg_risk_1 = (np.logaddexp(0, -exp[self.LTR == 1] * ZTR[self.LTR == 1]) * wTrue).sum()
        return reg + avg_risk_0 + avg_risk_1

    def trainLogReg(self, DTR, LTR):
        self.DTR = DTR
        self.LTR = LTR
        x0 = np.zeros(DTR.shape[0] + 1)
        self.xf = scipy.optimize.fmin_l_bfgs_b(
            func=self.__logreg_obj_prior_weighted if self.prior_weighted else self.__logreg_obj,
            x0=x0,
            approx_grad=True,
            # iprint=0
        )[0]
        return self.xf

    def predict(self, DVAL, label=False, threshold=0):
        w = self.xf[:-1]
        b = self.xf[-1]
        sval = np.dot(w.T, DVAL) + b
        if label:
            return np.int32(sval > threshold)
        else:
            return sval


def compute_minDCF_actDCF(xf, LVAL, DVAL, pi_emp, Cfn=1, Cfp=1, prior=0.5):
    w = xf[:-1]
    b = xf[-1]
    sval = np.dot(w.T, DVAL) + b
    th = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
    predictedLabels = np.int32(sval > th)
    error_rate = e.error_rate(predictedLabels, LVAL)
    print("Error rate:", error_rate, "%")
    sValLLR = sval - np.log(pi_emp / (1 - pi_emp))
    minDCF = bdm.compute_minDCF_binary(sValLLR, LVAL, prior, Cfn, Cfp)
    confusionMatrix = bdm.compute_confusion_matrix(predictedLabels, LVAL)
    actDCF = bdm.computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=True)

    print("actDCF:", actDCF)
    return minDCF, actDCF

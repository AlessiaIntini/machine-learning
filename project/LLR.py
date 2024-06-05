import matplotlib
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
        reg = 0.5 * self.lbd * np.linalg.norm(w) ** 2
        exp = (np.dot(w.T, self.DTR) + b)
        avg_risk_0 = np.logaddexp(0, -exp[self.LTR == 0] * ZTR[self.LTR == 0]).mean() * (1 - self.prior)
        avg_risk_1 = np.logaddexp(0, -exp[self.LTR == 1] * ZTR[self.LTR == 1]).mean() * self.prior
        return reg + avg_risk_0 + avg_risk_1

    def trainLogReg(self, DTR, LTR):
        self.DTR = DTR
        self.LTR = LTR
        x0 = np.zeros(DTR.shape[0] + 1)
        xf = \
            scipy.optimize.fmin_l_bfgs_b(
                func=self.__logreg_obj_prior_weighted if self.prior_weighted else self.__logreg_obj,
                x0=x0,
                approx_grad=True, iprint=0)[0]
        return xf


def calculate_sllr(x, w, b, pi_emp=0.5):
    s = np.dot(w.T, x) + b
    sllr = s - np.log(pi_emp / (1 - pi_emp))
    return sllr


def compute_minDCF_actDCF(xf, LVAL, DVAL, pi_emp, Cfn=1, Cfp=1, prior=0.5):
    # w = xf[:-1]
    # b = xf[-1]
    # sllr = np.array([calculate_sllr(x, w, b, pi_emp) for x in DVAL.T])
    # predictions = (sllr > 0).astype(int)
    # error_rate = error.error_rate(predictions, LVAL)
    # print("Error rate:", error_rate, "%")
    # minDCF = bdm.compute_minDCF_binary(sllr, LVAL, prior, Cfn, Cfp)
    # print("minDCF:", minDCF)
    # confusionMatrix = bdm.compute_confusion_matrix(predictions, LVAL)
    # actDCF = bdm.computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=True)
    # print("actDCF:", actDCF)
    w = xf[:-1]
    b = xf[-1]
    sval = np.dot(w.T, DVAL) + b
    PVAL = (sval > 0) * 1
    error_rate = e.error_rate(PVAL, LVAL)

    sllr = np.array([calculate_sllr(x, w, b, pi_emp) for x in DVAL.T])
    predictions = (sllr > 0).astype(int)
    print("Error rate:", error_rate, "%")
    minDCF = bdm.compute_minDCF_binary(sllr, LVAL, prior, Cfn, Cfp)
    print("minDCF:", minDCF)
    confusionMatrix = bdm.compute_confusion_matrix(predictions, LVAL)
    actDCF = bdm.computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=True)

    print("actDCF:", actDCF)
    return minDCF, actDCF


def plot_minDCF_actDCF(minDCF, actDCF, title, ldbArray, m=0):
    matplotlib.pyplot.figure()
    if m != 0:
        title = title + " m = " + str(m)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.scatter(ldbArray, minDCF, label='minDCF', color='b')
    matplotlib.pyplot.plot(ldbArray, minDCF, color='b')
    matplotlib.pyplot.xlabel('lambda')
    matplotlib.pyplot.ylabel('minDCF value')
    matplotlib.pyplot.xscale('log', base=10)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.scatter(ldbArray, actDCF, label='actDCF', color='r')
    matplotlib.pyplot.plot(ldbArray, actDCF, color='r')
    matplotlib.pyplot.xlabel('lambda')
    matplotlib.pyplot.ylabel('actDCF value')
    matplotlib.pyplot.xscale('log', base=10)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
